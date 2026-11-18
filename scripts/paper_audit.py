# Copyright (c) anneal developers
"""Recompute the paper and benchmark-table numbers from the committed rows.

Every check reads a committed CSV or TSV, recomputes a value, and compares it
with the number printed in ``paper/jctc/*.tex`` or a bench README. Medians and
interquartile ranges are Hyndman-Fan type 7. Fisher is the two-sided exact
test. Mann-Whitney is two-sided with the tie-corrected normal approximation;
each table names whether its runner applies the continuity correction.

A printed number agrees with a recomputed one when they differ by at most
half a unit in the printed number's last digit. A bare integer below 10 is
read as exact, so a printed p of 1 cannot stand for 0.6.

Every row file read here is checked for a code revision and a host. A file
without them, or whose revision is not in this repository, must be listed in
``KNOWN_GAPS``. A revision that was rebuilt on push is read through its alias
in ``ALIASES`` once the pushed equivalent is verified; one with no pushed
equivalent is listed in ``UNPUSHED`` with the reason. Printed numbers no
committed row backs are listed in ``UNTRACED``; their text must still be where
the list says. The revision checks need the full history and the remote
branches; a clone without them is refused, with exit status 2.

    python3 scripts/paper_audit.py            # exit 1 on any drift
    python3 scripts/paper_audit.py --report   # also print every check
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import importlib.util
import math
import os
import random
import re
import subprocess
import tarfile
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = "paper/jctc/manuscript.tex"
SI = "paper/jctc/si.tex"

REVISION_COLUMNS = ("code_revision", "anneal_head", "release_revision", "head", "harness_revision")
HOST_COLUMNS = ("host",)
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = {"aac1548a8aec13b53ad12a8d6904625081685d2e": "GMIN source/SVNREV"}
# Local revisions that rows cite, each with the pushed commit it was rebuilt as.
ALIASES = "bench/revision-aliases.csv"
# Cited revisions with no pushed equivalent. de22f437 records the ASE and GOFEE
# panel from five campaign heads and sits on the last, e09e0487, alone. The
# pushed history before it has one commit, 57735de1, that adds the two-hour
# cap, yet three heads ran capped, so the heads are not its commits one to one.
UNPUSHED = {
    "0f988569b49afe162e2f5ffd71b2ab45d0b3afec": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "349523d3f5f42b748f3af1573166fd17155ac461": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "befbc3e8": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own; the row gives eight digits",
    "bf09f7a2b7a6bd6359898db675eeafefcd20548b": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "ce1e7a46826cf51784ed26587c14b2747f5f48e1": "the GFN2 call-floor fix on f2a628c0, a local build head of the GFN2 "
                                                "Si10 and C10 floor rows; the fix was pushed on another base",
    "5351f272370dc8c0d18b06149753effc99aeabd5": "eae26977 with the call floor and the C20 example, a local build head of "
                                                "the GFN2 C20 rows; b4ddad0d on bench/covalent-cov6b has its tree",
    "b49341e308e6e98e39a2ea7bcc7fbf743f6c45ac": "the build head of the recommended arm of the ASE panel, first cited by "
                                                "70c0ad6c; no pushed branch holds it",
    "5a7f629d79be88b9d3f0cb2c6665062bda4ac625": "the build head of the cluster-preset and ASE arms of the Lennard-Jones "
                                                "ASE panel, first cited by b70e3269; no pushed branch holds it",
    "3654866310baf4ef63853130608ab65392429ff6": "the build head of the LJ55 ASE arms, first cited by 887764a2; no pushed "
                                                "branch holds it",
    "33bed75ff4e49af3628e7156e31c6e53e2fde32c": "the build head of the first Au55 ASE rows, first cited by edb2a088; no "
                                                "pushed branch holds it",
    "fc1af066a915a80eb46a3bc478caaf155cc9b29f": "the build head of the Au55 cluster-preset rerun and its ASE arms, first "
                                                "cited by 0f69a6b1; no pushed branch holds it",
    "1ce3ab64493196afd25e8690648e21157d91b882": "the template-free landing head for the GMIN panel and the Au55 ASE "
                                                "reruns, first cited by 93e17b69; no pushed branch holds it",
    "90b88235fe54abace3bde32789582eb6573ed40a": "the build head of the Morse ASE panel, first cited by c3bcdc01; no pushed "
                                                "branch holds it",
}

KNOWN_GAPS = {
    "data/sc96_78_hits.csv": "code_revision unrecorded on some rows; harness_revision unrecorded on some rows; host "
                             "empty or unknown on some rows",
    "bench/depth-posterior-ablation/results.csv": "revision unrecorded (no revision column)",
    "results/size-tiered-selection.csv": "revision and host unrecorded (no revision or host column)",
    "results/lj-funnel-stalls.csv": "revision and host unrecorded (no revision or host column)",
    "results/cross-field-hits.csv": "revision and host unrecorded (no revision or host column)",
    "bench/tip4p-water-rates/rates-a24ba760.csv": "revision unrecorded (no revision column)",
    "bench/tip4p-water-rates/rates.csv": "revision unrecorded (no revision column)",
    "paper/jctc/logs/water/NUMBERS.tsv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/hexamer-xtb680-repolish.csv": "revision unrecorded (no revision column)",
    "paper/jctc/data/water_v45b/water-v45b-perseed.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/campaign-scores.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/system_visuals/references.csv": "revision and host unrecorded (no revision or host column)",
    "bench/gad-published/summary.csv": "revision recorded; no host column",
    "bench/morse-panel/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/morse-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/fp-fingerprint/rows.csv": "revision and host unrecorded (no revision or host column)",
    "bench/tip4p-h2h/h2h-summary.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/before/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/after/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/v2/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/v3/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/gfn2/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/covalent-h2h/gfn2/false-hits.csv": "revision recorded; no host column",
    "bench/covalent-h2h/gfn2/rows.csv": "revision 5351f272 is not in this repository; revision ce1e7a46 is not in "
                                        "this repository",
    "bench/ase-gofee-h2h/results.csv": "revision recorded; no host column; head unrecorded on some rows; revision "
                                       "0f988569 is not in this repository; revision 349523d3 is not in this "
                                       "repository; revision befbc3e8 is not in this repository; revision bf09f7a2 is "
                                       "not in this repository",
    "bench/ase-gofee-h2h/summary.csv": "revision and host unrecorded (no revision or host column)",
    "bench/ase-gofee-h2h/asg2-summary.csv": "revision and host unrecorded (no revision or host column)",
    "bench/growpal-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv": "revision and host unrecorded (no revision or host column)",
    "bench/nh3-octamer/ladder.csv": "revision and host unrecorded (no revision or host column)",
    "bench/ase-au55/rows-for-cluster.csv": "revision 1ce3ab64 is not in this repository; revision fc1af066 is not in this "
                                           "repository",
    "bench/ase-au55/superseded.csv": "revision 33bed75f is not in this repository",
    "bench/ase-au55/rows.csv": "revision 1ce3ab64 is not in this repository; revision 33bed75f is not in this repository; "
                               "revision b49341e3 is not in this repository",
    "bench/ase-lj104/rows.csv": "revision 5a7f629d is not in this repository; revision b49341e3 is not in this repository",
    "bench/ase-lj150/rows.csv": "revision 5a7f629d is not in this repository; revision b49341e3 is not in this repository",
    "bench/ase-lj38/rows.csv": "revision 5a7f629d is not in this repository; revision b49341e3 is not in this repository",
    "bench/ase-lj55/rows.csv": "revision 36548663 is not in this repository; revision 5a7f629d is not in this "
                               "repository; revision b49341e3 is not in this repository",
    "bench/ase-lj98/rows.csv": "revision 5a7f629d is not in this repository; revision b49341e3 is not in this repository",
    "bench/ase-nonlj/rows.csv": "revision 90b88235 is not in this repository",
    "bench/ase-nonlj/identity.csv": "revision and host unrecorded (no revision or host column)",
    "bench/ase-nonlj/summary.csv": "revision 90b88235 is not in this repository",
    "bench/ase-recommended/summary.csv": "revision 1ce3ab64 is not in this repository; revision 36548663 is not in this "
                                         "repository; revision 5a7f629d is not in this repository; revision b49341e3 is "
                                         "not in this repository",
    "bench/gmin-panel/rows.csv": "revision 1ce3ab64 is not in this repository",
    "bench/gmin-panel/summary.csv": "revision 1ce3ab64 is not in this repository",
    "bench/morse-panel/template-free-rows.csv": "revision 1ce3ab64 is not in this repository",
    "paper/jctc/data/lj55_lj98/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/literature.tsv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv": "revision and host unrecorded (no revision or host "
                                                                "column)",
    "paper/jctc/data/lj75_gmin/lj75-derived-seeds-2000-2199.csv": "revision and host unrecorded (no revision or host "
                                                                  "column)",
    "paper/jctc/data/lj75_gmin/lj75-recommended-seeds-2000-2199.csv": "revision and host unrecorded (no revision or "
                                                                      "host column)",
}

UNTRACED = [
    ("bench/gad-published/README.md", "seed 2000 at 226282 calls and seed 2006 at 114143 calls",
     "the per-seed GAD rerun rows were not committed"),
    ("bench/gad-published/README.md", "a median of 13 generations on LJ31 (2–20) and 15.5 on LJ38 (12–24)",
     "the per-seed GAD rerun rows were not committed"),
    ("bench/gad-published/README.md", "| LJ31 | gmin | 12/12 | 47139.5 | 73509.5 |",
     "Mann-Whitney p needs the per-seed rows, which were not committed"),
    ("bench/metal-cluster-h2h/ptcu-b3d03061/README.md", "No committed row holds the −81.90409805",
     "the earlier Pt6Cu17 energy has no committed row"),
    ("bench/water-h2o20-goat/README.md", "Seed 16001, 28000 calls, same wrapper as the table above, scored −2214.506788 eV",
     "the seed 16001 rows were not committed"),
]


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, str, str]] = []
        self.read: dict[str, list[dict[str, str]]] = {}
        self.text_cache: dict[str, str] = {}

    def record(self, ok: bool, where: str, label: str, shown: object, value: object) -> None:
        self.results.append(("OK" if ok else "DRIFT", where, label, str(shown), str(value)))

    def rows(self, path: str, delimiter: str = ",") -> list[dict[str, str]]:
        if path not in self.read:
            with (ROOT / path).open(newline="") as handle:
                lines = [line for line in handle if line.strip() and not line.startswith("#")]
            self.read[path] = list(csv.DictReader(lines, delimiter=delimiter))
        return self.read[path]

    def text(self, path: str) -> str:
        if path not in self.text_cache:
            self.text_cache[path] = (ROOT / path).read_text()
        return self.text_cache[path]

    def cell(self, where: str, label: str, shown: str, value: object) -> None:
        self.record(agrees(shown, value), where, label, shown, value)

    def said(self, path: str, fragment: str) -> None:
        self.record(collapse(fragment) in collapse(self.text(path)), path, "prose", fragment, "present")

    def drifts(self) -> list[tuple[str, str, str, str, str]]:
        return [r for r in self.results if r[0] != "OK"]


def collapse(text: str) -> str:
    return " ".join(text.split())


def type7(values: list[float], p: float) -> float:
    ordered = sorted(float(v) for v in values)
    h = (len(ordered) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def median(values: list[float]) -> float:
    return type7(values, 0.5)


def iqr(values: list[float]) -> float:
    return type7(values, 0.75) - type7(values, 0.25)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, summing tables no more likely than the observed."""
    hits = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(a: list[float], b: list[float], continuity: bool, degenerate: float | None = 1.0) -> float | None:
    """Two-sided Mann-Whitney p, normal approximation with the tie correction.

    ``degenerate`` is returned when every value ties: 1.0 for the runners that
    write 1, None for those that write scipy's nan as a blank."""
    n1, n2 = len(a), len(b)
    pooled = sorted(list(a) + list(b))
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in a) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return degenerate
    dev = abs(u - n1 * n2 / 2.0) - (0.5 if continuity else 0.0)
    return min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))


def parse(shown: str) -> tuple[float, float, bool] | None:
    """Value, half a unit in the last printed digit, and whether it is a bare integer."""
    s = shown.strip().strip("$").replace("\\(", "").replace("\\)", "").strip()
    s = s.replace("−", "-").replace(",", "").replace("_", "").replace("\u2009", "")
    m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+))\s*\\times\s*10\^\{?([-+]?\d+)\}?", s)
    if m is None:
        m = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+))(?:[eE]([-+]?\d+))?", s)
    if m is None:
        return None
    mantissa, exponent = m.group(1), int(m.group(2) or 0)
    places = len(mantissa.split(".")[1]) if "." in mantissa else 0
    bare = "." not in mantissa and m.group(2) is None
    return float(mantissa) * 10.0**exponent, 0.5 * 10.0 ** (exponent - places), bare


def agrees(shown: str, value: object) -> bool:
    shown = shown.strip()
    if value is None:
        return shown in ("", "-", "--", "–", "—")
    if isinstance(value, str):
        return shown == value
    parsed = parse(shown)
    if parsed is None:
        return False
    x, half, bare = parsed
    v = float(value)
    if bare and abs(x) < 10:
        return abs(x - v) <= 1e-9 * max(1.0, abs(v))
    return abs(x - v) <= half * (1 + 1e-9) + 1e-12 * abs(v)


def md_table(audit: Audit, path: str, header_start: str, after: int = 0) -> list[tuple[int, dict[str, str]]]:
    """Rows of the first pipe table at or after line ``after`` whose header starts with ``header_start``."""
    lines = audit.text(path).splitlines()
    for i, line in enumerate(lines):
        if i + 1 < after or not line.startswith("| " + header_start):
            continue
        header = [c.strip() for c in line.strip().strip("|").split("|")]
        out = []
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            cells = [c.strip() for c in lines[j].strip()[1:-1].split("|")]
            out.append((j + 1, dict(zip(header, cells))))
            j += 1
        return out
    raise SystemExit(f"{path}: no table starting '{header_start}' after line {after}")


def tex_table(audit: Audit, path: str, label: str) -> list[list[str]]:
    text = audit.text(path)
    start = text.index("\\label{" + label + "}")
    body = text[text.index("\\midrule", start) + len("\\midrule"): text.index("\\bottomrule", start)]
    return [[c.strip() for c in row.split("&")] for row in body.split("\\\\") if row.strip()]


def hits_of(rows: list[dict[str, str]], key: str = "hit") -> int:
    return sum(r[key].strip().lower() in ("1", "true", "yes") for r in rows)


def pick(rows: list[dict[str, str]], **where: str) -> list[dict[str, str]]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def frac(rows: list[dict[str, str]], key: str = "hit") -> str:
    return f"{hits_of(rows, key)}/{len(rows)}"


# --------------------------------------------------------------------------- manuscript

CLUSTER = {"LJ\\(_{38}\\)": 38, "LJ\\(_{55}\\)": 55, "LJ\\(_{75}\\)": 75, "LJ\\(_{98}\\)": 98, "LJ\\(_{104}\\)": 104}


def manuscript_ablation(audit: Audit) -> None:
    first = audit.rows("bench/depth-posterior-ablation/results.csv")
    second = audit.rows("results/size-tiered-selection.csv")
    budgets = {38: 400000, 55: 1000000, 75: 4000000, 98: 4000000, 104: 4000000}
    counts = {}
    for row in tex_table(audit, TEX, "tab:ablation"):
        size = CLUSTER[row[0]]
        arms = {}
        for arm in ("posterior", "uniform"):
            group = pick(first, system=f"lj{size}", arm=arm)
            if size >= 75:
                group += pick(second, n=str(size), arm=arm)
            arms[arm] = group
            for r in group:
                audit.cell(TEX, f"ablation lj{size} {arm} seed {r['seed']} budget", str(budgets[size]), float(r.get("budget") or budgets[size]))
        where = f"{TEX} tab:ablation lj{size}"
        audit.cell(where, "budget", row[1], budgets[size])
        audit.cell(where, "posterior", row[2], frac(arms["posterior"]))
        audit.cell(where, "uniform", row[3], frac(arms["uniform"]))
        p = fisher(hits_of(arms["posterior"]), len(arms["posterior"]), hits_of(arms["uniform"]), len(arms["uniform"]))
        audit.cell(where, "Fisher p", row[4], p)
        counts[size] = (hits_of(arms["posterior"]), len(arms["posterior"]), hits_of(arms["uniform"]), len(arms["uniform"]))
    p75, p98, p104 = counts[75], counts[98], counts[104]
    audit.said(TEX, f"the posterior hits LJ\\(_{{75}}\\) on {p75[0]} of {p75[1]} and the uniform draw hits {p75[2]} of {p75[3]}.")
    audit.said(TEX, f"Both hit LJ\\(_{{98}}\\) on {p98[0]} of {p98[1]}.")
    audit.said(TEX, f"The posterior hits LJ\\(_{{104}}\\) on {p104[0]} of {p104[1]} and the uniform draw hits {p104[2]} of {p104[3]}.")
    audit.said(TEX, f"LJ\\(_{{75}}\\) is {p75[0]} of {p75[1]} against {p75[2]} of {p75[3]},")
    audit.said(TEX, f"LJ\\(_{{104}}\\) is {p104[0]} of {p104[1]} against {p104[2]} of {p104[3]}.")


def selection_tables(audit: Audit) -> None:
    """The arm-selection tables beside the two result files of tab:ablation.

    The depth-posterior README takes the median over hits; the size-tiered
    table counts a miss as the budget."""
    budgets = {38: 400000, 55: 1000000, 75: 4000000, 98: 4000000, 104: 4000000}
    first = audit.rows("bench/depth-posterior-ablation/results.csv")
    second = audit.rows("results/size-tiered-selection.csv")
    for path, head, rows, size_key, calls_key, hits_only in (
            ("bench/depth-posterior-ablation/README.md", "Size | Arm", first, "Size", "force_calls_to_hit", True),
            ("results/size-tiered-selection.md", "Cluster | Arm", second, "Cluster", "calls_to_hit", False)):
        for line, row in md_table(audit, path, head):
            size = int(row[size_key].removeprefix("LJ"))
            if rows is first:
                g = pick(rows, system=f"lj{size}", arm=row["Arm"])
            else:
                g = pick(rows, n=str(size), arm=row["Arm"])
            calls = [float(r[calls_key]) if r["hit"] == "1" else float(budgets[size]) for r in g
                     if r["hit"] == "1" or not hits_only]
            where = f"{path}:{line} lj{size} {row['Arm']}"
            audit.cell(where, "hits", row["Hits"], frac(g))
            audit.cell(where, "median calls", row["Median calls"], median(calls))
            audit.cell(where, "IQR", row["IQR"], iqr(calls))
            audit.cell(where, "wall sum", row["Wall sum (s)"], sum(float(r["wall_s"]) for r in g))


def manuscript_retry(audit: Audit) -> None:
    rows = audit.rows("results/lj-funnel-stalls.csv")
    retry98 = pick(rows, phase="retry", n="98")
    after98 = pick(rows, phase="after", n="98")
    audit.said(TEX, f"LJ\\(_{{98}}\\) hits on {frac(retry98).replace('/', ' of ')} inside \\(4\\times 10^{{6}}\\) calls.")
    audit.said(TEX, f"the eighth-leave retry hits {frac(after98).replace('/', ' of ')}.")
    miss = [r for r in after98 if r["hit"] == "0"]
    audit.cell(TEX, "retry: the hold-only miss is seed 6404", "6404", miss[0]["seed"] if len(miss) == 1 else None)
    audit.said(TEX, f"Seed 6404 stays at a gap of \\({miss[0]['gap']}\\) above \\(-543.665361\\).")
    for size in ("38", "55", "75"):
        audit.cell(TEX, f"retry off below 98: lj{size} on 6400-6405", "6/6", frac(pick(rows, phase="after", n=size)))
    retry104 = pick(rows, phase="retry", n="104")
    audit.said(TEX, f"The retry rows stored for that leave hit {hits_of(retry104)} of {len(retry104)} on LJ\\(_{{104}}\\)")
    audit.said(TEX, f"{frac(retry98).replace('/', ' of ')} on LJ\\(_{{98}}\\).")


GMIN_PANEL_REVISION = "c2de652dca0bc75e2b0cb4afffac91fd656404e4"
GMIN_PANEL_README = "bench/gmin-panel/README.md"


def mann_whitney_u(a: list[float], b: list[float]) -> float:
    n1, n2 = len(a), len(b)
    pooled = sorted(list(a) + list(b))
    n = n1 + n2
    rank: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        i = j + 1
    return sum(rank[v] for v in a) - n1 * (n1 + 1) / 2.0


def gmin_ranked_calls(rows: list[dict[str, str]], budget: int) -> list[float] | None:
    if any(r["hit"] == "0" and int(r["charged"]) < budget for r in rows):
        return None
    return [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in rows]


def gmin_ranked_wall(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[list[float], list[float]] | None:
    l_miss = [float(r["wall_s"]) for r in left if r["hit"] == "0"]
    r_miss = [float(r["wall_s"]) for r in right if r["hit"] == "0"]
    if l_miss and r_miss:
        return None
    l_hit = [float(r["wall_s"]) for r in left if r["hit"] == "1"]
    r_hit = [float(r["wall_s"]) for r in right if r["hit"] == "1"]
    if any(m < h for m in l_miss for h in r_hit) or any(m < h for m in r_miss for h in l_hit):
        return None
    return (
        [float(r["wall_s"]) if r["hit"] == "1" else math.inf for r in left],
        [float(r["wall_s"]) if r["hit"] == "1" else math.inf for r in right],
    )


def gmin_panel_groups(audit: Audit) -> tuple[object, dict[tuple[str, str], list[dict[str, str]]]]:
    """The latest panel row of each system, arm and seed, in seed order, after row validation."""
    spec = importlib.util.spec_from_file_location("gmin_panel", ROOT / "scripts/gmin_panel/run_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    rows = audit.rows("bench/gmin-panel/rows.csv")
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    latest: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    bad_rows: list[str] = []
    for index, r in enumerate(rows, start=2):
        if r["code"] == "gmin" and r["code_revision"] != panel.GMIN_REV:
            continue
        system = r["system"]
        budget = panel.PAPER_BUDGETS.get(system)
        ref = references.get(system)
        if budget is None or ref is None or r["hit"] not in ("0", "1") or int(r["budget"]) != budget:
            bad_rows.append(f"line {index}")
            continue
        hit = r["hit"] == "1"
        best = float(r["best"])
        evals, charged = int(r["evals_to_hit"]), int(r["charged"])
        wall, pot, outside = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
        ok = (
            hit == (best <= ref + panel.HIT_TOL)
            and (not hit or (abs(best - ref) < panel.HIT_TOL and 0 < evals <= charged <= budget))
            and (hit or (evals == budget and charged == budget))
            and 0.0 <= pot <= wall + 2e-6
            and outside >= 0.0
            and abs(max(wall - pot, 0.0) - outside) <= 2e-6
        )
        if not ok:
            bad_rows.append(f"line {index} {system} {r['code']} {r['seed']}")
        latest.setdefault((system, r["code"]), {})[int(r["seed"])] = r
    audit.cell("bench/gmin-panel/rows.csv", "row invariants", " ".join(bad_rows) or "ok", "ok")
    groups = {}
    for system in panel.PAPER_BUDGETS:
        for code in ("anneal", "gmin"):
            by_seed = latest.get((system, code), {})
            group = [by_seed[s] for s in sorted(by_seed)]
            where = f"bench/gmin-panel/rows.csv {system} {code}"
            audit.cell(where, "seeds", " ".join(r["seed"] for r in group), " ".join(map(str, panel.SEEDS)))
            audit.cell(where, "budget", " ".join(sorted({r["budget"] for r in group})), str(panel.PAPER_BUDGETS[system]))
            groups[(system, code)] = group
        pair_sessions = {r["session"] for code in ("anneal", "gmin") for r in groups[(system, code)]}
        audit.record(len(pair_sessions) >= 1, f"bench/gmin-panel/rows.csv {system}", "paired session count is nonzero",
                     "at least one", str(len(pair_sessions)))
    hosts = {r["host"] for group in groups.values() for r in group}
    audit.cell("bench/gmin-panel/rows.csv", "host", " ".join(sorted(hosts)), "bench")
    return panel, groups


def gmin_panel_summary(audit: Audit, panel: object, groups: dict) -> None:
    summary = audit.rows("bench/gmin-panel/summary.csv")
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    audit.cell("bench/gmin-panel/summary.csv", "systems",
               " ".join(r["system"] for r in summary), " ".join(panel.PAPER_BUDGETS))
    for line in summary:
        system = line["system"]
        budget = panel.PAPER_BUDGETS[system]
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        where = f"bench/gmin-panel/summary.csv {system}"
        a_calls = [float(r["evals_to_hit"]) for r in a]
        g_calls = [float(r["evals_to_hit"]) for r in g]
        a_hits = [float(r["evals_to_hit"]) for r in a if r["hit"] == "1"]
        g_hits = [float(r["evals_to_hit"]) for r in g if r["hit"] == "1"]
        rc_a, rc_g = gmin_ranked_calls(a, budget), gmin_ranked_calls(g, budget)
        rw = gmin_ranked_wall(a, g)
        audit.cell(where, "budget", line["budget"], float(budget))
        audit.cell(where, "reference", line["reference"], references[system])
        audit.cell(where, "anneal_hits", line["anneal_hits"], float(hits_of(a)))
        audit.cell(where, "anneal_n", line["anneal_n"], float(len(a)))
        audit.cell(where, "gmin_hits", line["gmin_hits"], float(hits_of(g)))
        audit.cell(where, "gmin_n", line["gmin_n"], float(len(g)))
        audit.cell(where, "fisher_p", line["fisher_p"], fisher(hits_of(a), len(a), hits_of(g), len(g)))
        audit.cell(where, "anneal_evals_median", line["anneal_evals_median"], median(a_calls))
        audit.cell(where, "gmin_evals_median", line["gmin_evals_median"], median(g_calls))
        audit.cell(where, "anneal_evals_iqr", line["anneal_evals_iqr"], iqr(a_calls))
        audit.cell(where, "gmin_evals_iqr", line["gmin_evals_iqr"], iqr(g_calls))
        audit.cell(where, "anneal_evals_median_to_hit", line["anneal_evals_median_to_hit"],
                   median(a_hits) if a_hits else None)
        audit.cell(where, "gmin_evals_median_to_hit", line["gmin_evals_median_to_hit"],
                   median(g_hits) if g_hits else None)
        audit.cell(where, "mannwhitney_u_calls", line["mannwhitney_u_calls"],
                   mann_whitney_u(rc_a, rc_g) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "mannwhitney_p_calls", line["mannwhitney_p_calls"],
                   mann_whitney(rc_a, rc_g, continuity=True) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "anneal_wall_median_s", line["anneal_wall_median_s"], median([float(r["wall_s"]) for r in a]))
        audit.cell(where, "gmin_wall_median_s", line["gmin_wall_median_s"], median([float(r["wall_s"]) for r in g]))
        audit.cell(where, "mannwhitney_u_wall", line["mannwhitney_u_wall"],
                   mann_whitney_u(*rw) if rw is not None else None)
        audit.cell(where, "mannwhitney_p_wall", line["mannwhitney_p_wall"],
                   mann_whitney(*rw, continuity=True) if rw is not None else None)
        for arm, group in (("anneal", a), ("gmin", g)):
            audit.cell(where, f"{arm}_pot_median_s", line[f"{arm}_pot_median_s"], median([float(r["pot_s"]) for r in group]))
            audit.cell(where, f"{arm}_outside_median_s", line[f"{arm}_outside_median_s"],
                       median([float(r["outside_s"]) for r in group]))
            audit.cell(where, f"{arm}_best_min", line[f"{arm}_best_min"], min(float(r["best"]) for r in group))
        audit.cell(where, "anneal_revision", line["anneal_revision"], " ".join(sorted({r["code_revision"] for r in a})))
        audit.cell(where, "gmin_revision", line["gmin_revision"], " ".join(sorted({r["code_revision"] for r in g})))
        audit.cell(where, "harness_revision", line["harness_revision"], " ".join(sorted({r["harness_revision"] for r in a + g})))
        audit.cell(where, "host", line["host"], "bench")
        audit.cell(where, "sessions", line["sessions"], " ".join(sorted({r["session"] for r in a + g})))
        audit.cell(where, "seeds", line["seeds"], " ".join(r["seed"] for r in a))


def gmin_panel_line(audit: Audit, where: str, cells: list[str], groups: dict, system: str, arm: str,
                    budget: int) -> None:
    """One table line: Budget, Arm, Hits, Median calls, IQR, Wall, Inside, Outside, Fisher p, MW p calls, MW p wall."""
    group = groups[(system, arm)]
    other = groups[(system, "gmin" if arm == "anneal" else "anneal")]

    def col(rows: list[dict[str, str]], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    shown_hits = re.sub(r"[~\s]*\\rerun\{\d+\}", "", cells[2])
    audit.cell(where, "hits", shown_hits, frac(group))
    audit.cell(where, "median calls", cells[3], median(col(group, "evals_to_hit")))
    audit.cell(where, "IQR calls", cells[4], iqr(col(group, "evals_to_hit")))
    audit.cell(where, "wall median", cells[5], median(col(group, "wall_s")))
    audit.cell(where, "inside median", cells[6], median(col(group, "pot_s")))
    audit.cell(where, "outside median", cells[7], median(col(group, "outside_s")))
    if arm == "anneal":
        rc_a, rc_g = gmin_ranked_calls(group, budget), gmin_ranked_calls(other, budget)
        rw = gmin_ranked_wall(group, other)
        audit.cell(where, "budget", cells[0], float(budget))
        audit.cell(where, "Fisher p", cells[8], fisher(hits_of(group), len(group), hits_of(other), len(other)))
        audit.cell(where, "MW p calls", cells[9],
                   mann_whitney(rc_a, rc_g, continuity=True) if rc_a is not None and rc_g is not None else None)
        audit.cell(where, "MW p wall", cells[10], mann_whitney(*rw, continuity=True) if rw is not None else None)
    else:
        for label, cell in zip(("budget", "Fisher p", "MW p calls", "MW p wall"), (cells[0], *cells[8:])):
            audit.cell(where, label, cell, None)


def gmin_panel_readme(audit: Audit, panel: object, groups: dict) -> None:
    rows = audit.rows("bench/gmin-panel/rows.csv")
    first_session = "20261117T035457Z"
    old_au = pick(rows, system="au55", code="gmin", session=first_session)
    new_session = groups[("au55", "gmin")][0]["session"]
    audit.said(GMIN_PANEL_README, GMIN_PANEL_REVISION)
    audit.said(GMIN_PANEL_README, panel.GMIN_REV)
    audit.said(GMIN_PANEL_README, f"adds the energy {dict((r[0], r[3]) for r in panel.SYSTEMS)['lj103']:.6f}")
    r0 = 2.884
    scale = r0 / 2.0 ** (1.0 / 6.0)
    rad = (2.0 + (3.0 * 55 / 17.77153175) ** (1.0 / 3.0)) * r0
    audit.said(GMIN_PANEL_README, f"(`RADIUS {rad:.6f}`), `STEP {0.40 * scale:.6f} 0.51` and `AVOID {4.0 * scale:.6f} 20 F`")
    audit.said(GMIN_PANEL_README, "`TEMPERATURE 0.1` eV from Wales's `examples/GMIN/MGupta.tar.bz2`")
    audit.said(GMIN_PANEL_README,
               f"The earlier Au55 rows of session `{first_session}` ran the alloy block of "
               "`scripts/metal_h2h/enter_gold.sh` (`RADIUS 50.0`, `TEMPERATURE 1.0`, `STEP 0.40 0.51` and "
               "`NEWRESTART 40` without `AVOID ... F`), which left `AVOIDRESEEDT` true, reseeded into a "
               f"57.7 Å cube after 40 stagnant steps, and hit {frac(old_au)}; those rows stay in `rows.csv`, "
               f"and session `{new_session}` reran both arms of Au55 under the Gupta input above "
               f"({frac(groups[('au55', 'anneal')])} against {frac(groups[('au55', 'gmin')])}).")
    s0 = [r for r in rows if r["session"] == first_session]
    gaps = []
    for prev, cur in zip(s0, s0[1:]):
        t0 = datetime.datetime.strptime(prev["utc"], "%Y-%m-%dT%H:%M:%SZ")
        t1 = datetime.datetime.strptime(cur["utc"], "%Y-%m-%dT%H:%M:%SZ")
        gap = (t1 - t0).total_seconds() - float(cur["wall_s"])
        if gap > 5.0:
            gaps.append((gap, cur))
    audit.cell(GMIN_PANEL_README, "VM suspend count", str(len(gaps)), "3")
    if len(gaps) == 3:
        g0, g1, g2 = (g for g, _ in gaps)
        r0_, r1_, r2_ = (r for _, r in gaps)
        audit.said(GMIN_PANEL_README, f"lie {g0:.0f} s, {g1:.0f} s and {g2:.0f} s after the row before plus their own wall.")
        audit.said(GMIN_PANEL_README,
                   f"On LJ75 GMIN seed {r0_['seed']} (wall {float(r0_['wall_s']):.1f} s, outside {float(r0_['outside_s']):.1f} s) "
                   f"and LJ98 anneal seed {r2_['seed']} (wall {float(r2_['wall_s']):.1f} s, outside {float(r2_['outside_s']):.1f} s)")
        other_out = [float(r["outside_s"]) for r in groups[("lj98", "gmin")] if r["seed"] != r1_["seed"]]
        post = next(r for r in groups[("lj98", "gmin")] if r["seed"] == "2005")
        audit.said(GMIN_PANEL_README,
                   f"On LJ98 GMIN seed {r1_['seed']} (wall {float(r1_['wall_s']):.1f} s), `pot_s` stayed at "
                   f"{float(r1_['pot_s']):.1f} s while `outside_s` rose to {float(r1_['outside_s']):.1f} s against "
                   f"{min(other_out):.1f}-{max(other_out):.1f} s on the other eleven seeds, and LJ98 GMIN seed 2005 "
                   f"right after the third resume took {float(post['wall_s']):.1f} s (inside {float(post['pot_s']):.1f} s)")


def manuscript_gmin(audit: Audit) -> None:
    panel, groups = gmin_panel_groups(audit)
    gmin_panel_summary(audit, panel, groups)
    arms = {"this work": "anneal", "anneal": "anneal", "GMIN": "gmin"}
    for path, label_index, lines in (
        (TEX, 1, [(None, row) for row in tex_table(audit, TEX, "tab:gmin")]),
        (GMIN_PANEL_README, 0,
         [(n, list(row.values())) for n, row in md_table(audit, GMIN_PANEL_README, "System | Budget | Arm")]),
    ):
        by_label = {labels[label_index]: system for system, labels in panel.LABELS.items()}
        shown = []
        system = None
        for line, cells in lines:
            if cells[0]:
                system = by_label.get(cells[0])
            arm = arms.get(cells[2])
            where = f"{path}:{line} {cells[0] or system} {cells[2]}" if line else f"{path} tab:gmin {system} {cells[2]}"
            if system is None or arm is None or len(cells) != 12:
                audit.record(False, where, "table line", " & ".join(cells), "a panel system and arm")
                continue
            shown.append((system, arm))
            gmin_panel_line(audit, where, cells[1:], groups, system, arm, panel.PAPER_BUDGETS[system])
        want = [(system, arm) for system in panel.PAPER_BUDGETS for arm in ("anneal", "gmin")]
        audit.cell(path, "gmin panel lines", " ".join(f"{s}/{a}" for s, a in shown), " ".join(f"{s}/{a}" for s, a in want))
    gmin_panel_readme(audit, panel, groups)
    gmin_panel_prose(audit, panel, groups)


def gmin_panel_prose(audit: Audit, panel: object, groups: dict) -> None:
    def name(system: str) -> str:
        label = panel.LABELS[system][1]
        return label.replace(", \\(\\rho=", " at \\(\\rho=")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(groups[(system, arm)])} of {len(groups[(system, arm)])}"

    references = {row[0]: row[3] for row in panel.SYSTEMS}
    systems = list(panel.PAPER_BUDGETS)
    ahead = [s for s in systems if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")])]
    behind = [s for s in systems if hits_of(groups[(s, "anneal")]) < hits_of(groups[(s, "gmin")])]
    tied = [s for s in systems if s not in ahead and s not in behind]
    significant = [s for s in systems if fisher(hits_of(groups[(s, "anneal")]), len(groups[(s, "anneal")]),
                                                hits_of(groups[(s, "gmin")]), len(groups[(s, "gmin")])) < 0.05]
    total = len(systems)
    audit.said(TEX, f"this setting hits more seeds on {len(ahead)} of {total}, the two tie on {len(tied)}, and GMIN hits "
                    f"more on {len(behind)}.")

    def ps(system: str) -> str:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return panel.show_p(fisher(hits_of(a), len(a), hits_of(g), len(g)), tex=True).replace("\\(", "").replace("\\)", "")

    audit.said(TEX, "GMIN hits " + ", and ".join(
        f"{name(s)} on {of(s, 'gmin')} against {of(s, 'anneal')}, with \\(p={ps(s)}\\)" for s in behind) + ".")
    audit.said(TEX, f"This setting hits more seeds than GMIN on {len(ahead)} of the {total} clusters.")
    seed_span = f"{panel.SEEDS[0]}--{panel.SEEDS[-1]}"
    rad = (2.0 + (3.0 * 55 / 17.77153175) ** (1.0 / 3.0)) * 2.884
    anneal_revs = {r["code_revision"] for s in systems for r in groups[(s, "anneal")]}
    shown_rev = sorted(anneal_revs)[0][:8] if len(anneal_revs) == 1 else "mixed"
    audit.said(TEX, f"Revisions: anneal \\texttt{{{shown_rev}}}, GMIN \\texttt{{{panel.GMIN_REV[:8]}}}.")
    audit.said(TEX, f"Seeds {seed_span}, one core, budget in force calls.")
    audit.said(TEX, f"Both arms use seeds {seed_span}, the budget in the table and the start this code writes for the seed.")
    audit.said(TEX, f"GMIN's nearest-neighbor container \\({rad:.6f}\\)~\\AA, step and taboo distances scaled by "
                    "\\(r_0/2^{1/6}\\) (\\(r_0=2.884\\)~\\AA), and \\(T=0.1\\)~eV from the Wales Gupta example.")
    audit.said(TEX,
               f"\\({references['lj102']:.6f}\\) at LJ\\(_{{102}}\\), "
               f"\\({references['lj103']:.6f}\\) at LJ\\(_{{103}}\\), "
               f"\\({references['lj150']:.6f}\\) at LJ\\(_{{150}}\\), "
               f"\\({references['au55']:.8f}\\)~eV at Au\\(_{{55}}\\), "
               f"\\({references['m55r6']:.6f}\\) and \\({references['m55r14']:.6f}\\) at M\\(_{{55}}\\) (\\(\\rho=6,14\\)), "
               f"and \\({references['m80r6']:.6f}\\) and \\({references['m80r14']:.6f}\\) at M\\(_{{80}}\\) (\\(\\rho=6,14\\)).")
    audit.said(TEX, "The hit is a quenched energy that differs by less than 0.001 from the Cambridge Cluster Database minimum:")
    audit.said(TEX, "GMIN hits more on " + ", and on ".join(f"{name(s)}, {of(s, 'gmin')} against {of(s, 'anneal')}"
                                                       for s in behind) + ".")
    audit.said(TEX, f"The two tie on {' and '.join(name(s) for s in tied)}.")
    ours = [s for s in significant if s in ahead]
    theirs = [s for s in significant if s in behind]
    audit.said(TEX, f"Fisher's test on hits is below 0.05 on {len(significant)} clusters: {len(ours)} for this work "
                    f"and {' and '.join(name(s) for s in theirs)} for GMIN.")
    counts = " and ".join(f"{of(s, 'gmin')} against {of(s, 'anneal')}" for s in behind)
    audit.said(TEX, f"Against GMIN on seeds {seed_span}, this setting hits more seeds on {len(ahead)} of {total} clusters, "
                    f"and GMIN hits more on {' and on '.join(name(s) for s in behind)}, {counts}.")
    audit.said(TEX, "Section~\\ref{sec:gmin} scores it on the clusters of Table~\\ref{tab:gmin}.")
    audit.said(TEX, f"GMIN hits more seeds on {len(behind)} of the {total}, {series(name(s) for s in behind)}.")

    def show_p(p: float) -> str:
        return panel.show_p(p, tex=True)

    def p_of(system: str) -> float:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return fisher(hits_of(a), len(a), hits_of(g), len(g))

    def col(system: str, arm: str, key: str) -> float:
        return median([float(r[key]) for r in groups[(system, arm)]])

    for system in behind:
        budget = panel.PAPER_BUDGETS[system]
        audit.cell(TEX, f"{system}: this setting's median calls are the full budget", str(budget),
                   col(system, "anneal", "evals_to_hit"))
        audit.cell(TEX, f"{system}: the full budget is 4e6", "4000000", float(budget))
    m80, lj150 = "m80r6", "lj150"
    audit.cell(TEX, "the GMIN losses are M80 rho=6 and LJ150", f"{lj150} {m80}", " ".join(behind))
    audit.said(TEX, f"On {name(m80)} GMIN reaches the minimum in a median of "
                    f"{panel.show_calls(col(m80, 'gmin', 'evals_to_hit'))} calls and "
                    f"{panel.show_s(col(m80, 'gmin', 'wall_s'))}~s, and this setting spends the full "
                    f"\\(4\\times 10^{{6}}\\) calls in a median of {panel.show_s(col(m80, 'anneal', 'wall_s'))}~s.")
    audit.said(TEX, f"On {name(lj150)} the median calls are {panel.show_calls(col(lj150, 'gmin', 'evals_to_hit'))} for "
                    f"GMIN and the full \\(4\\times 10^{{6}}\\) for this setting, and Fisher's \\(p\\) is "
                    f"{show_p(p_of(lj150))}.")
    audit.said(TEX, f"GMIN hits more seeds than this setting on {len(behind)} of the {total} clusters of "
                    "Table~\\ref{tab:gmin}.")
    for system in behind:
        audit.said(TEX, f"On {name(system)} GMIN hits {of(system, 'gmin')} and this setting {of(system, 'anneal')}, "
                        f"and Fisher's \\(p\\) is {show_p(p_of(system))}.")
    both = {f"{hits_of(groups[(s, 'anneal')])}/{len(groups[(s, 'anneal')])}" for s in tied}
    audit.cell(TEX, "the tied clusters share one count", "1", float(len(both)))
    audit.said(TEX, f"The two tie on {' and '.join(name(s) for s in tied)}, where both hit "
                    f"{next(iter(both)).replace('/', ' of ')}.")
    gmin_panel_drivers(audit)


def series(items) -> str:
    """A list with the serial comma: ``a``, ``a and b``, ``a, b, and c``."""
    items = list(items)
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def morse_pair(rho: float, r: float) -> float:
    """The Doye-Wales Morse pair with unit well depth and unit pair minimum."""
    e = math.exp(rho * (1.0 - r))
    return e * (e - 2.0)


def gmin_panel_drivers(audit: Audit) -> None:
    """The driver settings Section sec:gmin names, read from the driver sources."""
    metal = audit.text("examples/h2h_metal.rs")
    morse = audit.text("examples/h2h_morse.rs")
    stride = re.search(r"cfg\.basin_stride = .*?\.unwrap_or\((\d+)\)", metal, re.S)
    audit.cell("examples/h2h_metal.rs", "basin stride", "8", float(stride.group(1)) if stride else None)
    copper = re.search(r'"au55" => \(\s*55,\s*[-\d.]+,\s*[\d_]+,\s*Kind::CopperGold \{ n_copper: (\d+) \}', metal)
    audit.cell("examples/h2h_metal.rs", "Au55 copper atoms", "0", float(copper.group(1)) if copper else None)
    audit.record("second_neighbour_energy() <" not in morse, "examples/h2h_morse.rs",
                 "long-ranged threshold gate is removed", "absent",
                 "absent" if "second_neighbour_energy() <" not in morse else "present")
    audit.said(TEX, "The gold driver gives it the copper--gold species list with no copper atom, applies the basin "
                    "bias of equation~\\ref{eq:delta} to every downhill or accepted proposal and to one other proposal "
                    f"in {stride.group(1) if stride else '?'}, and decides the rest on the bare energy.")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap driver prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")


ASE_PANEL = ("lj38", "lj55", "lj98", "lj104", "lj150", "au55")
MORSE_PANEL = ("m38r6", "m38r14", "m55r6", "m55r14")
MORSE_REVISION = "9a30f3190b3bb7040c675a904f8388eb68224151"
ASE_ARMS = ("ase_bh", "ase_mh")
PANEL_SEEDS = " ".join(str(s) for s in range(2000, 2012))


def load_script(name: str, path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ase_groups(audit: Audit) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The rows of each system and arm of the two ASE panels, in seed order.

    Au55 takes the recommended rows of rows.csv and the other arms of
    rows-for-cluster.csv, the rerun of the corrected cluster preset."""
    groups = {}

    def campaign(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
        rows = [r for r in rows if r["system"] == system and r["arm"] == arm]
        grouped: dict[tuple[str, str], tuple[int, dict[int, dict[str, str]]]] = {}
        for index, row in enumerate(rows):
            key = (row.get("code_revision", ""), row.get("cap_s", ""))
            last, seeds = grouped.get(key, (-1, {}))
            seeds[int(row["seed"])] = row
            grouped[key] = (index, seeds)
        full = [(last, seeds) for last, seeds in grouped.values() if tuple(sorted(seeds)) == tuple(range(2000, 2012))]
        if full:
            _last, seeds = max(full, key=lambda item: item[0])
            return [seeds[s] for s in range(2000, 2012)]
        latest: dict[int, dict[str, str]] = {}
        for row in rows:
            latest[int(row["seed"])] = row
        return [latest[s] for s in sorted(latest)]

    for system in ASE_PANEL:
        rows = audit.rows(f"bench/ase-{system}/rows.csv")
        others = audit.rows("bench/ase-au55/rows-for-cluster.csv") if system == "au55" else rows
        for arm in ("anneal", "recommended", *ASE_ARMS):
            source = rows if arm == "recommended" else others
            groups[(system, arm)] = campaign(source, system, arm)
    morse = pick(audit.rows("bench/ase-nonlj/rows.csv"), code_revision=MORSE_REVISION)
    for system in MORSE_PANEL:
        for arm in ("for_cluster", "recommended", *ASE_ARMS):
            groups[(system, arm)] = pick(morse, system=system, arm=arm)
    for (system, arm), group in groups.items():
        group.sort(key=lambda r: int(r["seed"]))
        where = f"ase panel {system} {arm}"
        audit.cell(where, "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
        audit.cell(where, "host", " ".join(sorted({r["host"] for r in group})), "bench")
    return groups


def ase_cap(audit: Audit, groups: dict, system: str) -> float:
    """The one wall cap of both ASE arms, checked to be three times the slowest cluster-preset wall."""
    base = "for_cluster" if system in MORSE_PANEL else "anneal"
    caps = sorted({r["cap_s"] for arm in ASE_ARMS for r in groups[(system, arm)]})
    slowest = max(float(r["wall_s"]) for r in groups[(system, base)])
    audit.cell(f"ase panel {system}", "one ASE wall cap", "1", float(len(caps)))
    audit.record(abs(float(caps[0]) - 3.0 * slowest) <= 1e-6, f"ase panel {system}", "cap is three times the slowest "
                 f"{base} wall", caps[0], f"{3.0 * slowest:.6f}")
    return float(caps[0])


def manuscript_ase(audit: Audit) -> None:
    """Tables tab:ase and tab:ase-morse, the summaries they come from, and every sentence that reads them."""
    panel = load_script("ase_panel_summary", "scripts/ase_panel/summarize.py")
    nonlj = load_script("ase_nonlj_summary", "scripts/ase_nonlj/summarize.py")
    gmin_panel, gmin = gmin_panel_groups(audit)
    groups = ase_groups(audit)

    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    def stops(system: str, arm: str, stop: str = "wall") -> int:
        return sum(r["stop"] == stop for r in groups[(system, arm)])

    def p(system: str, arm: str) -> float:
        a, b = groups[(system, "recommended")], groups[(system, arm)]
        return fisher(hits_of(a), len(a), hits_of(b), len(b))

    def gap(system: str, arm: str, reference: float) -> float:
        return median([float(r["best"]) for r in groups[(system, arm)]]) - reference

    def show_p(value: float) -> str:
        return gmin_panel.show_p(value, tex=True)

    references = {s: v[2] for s, v in panel.SYSTEMS.items()} | {s: v[3] for s, v in nonlj.SPEC.items()}
    budgets = {s: v[1] for s, v in panel.SYSTEMS.items()} | {s: int(v[2]) for s, v in nonlj.SPEC.items()}
    for system, group in groups.items():
        audit.cell(f"ase panel {system[0]} {system[1]}", "budget",
                   " ".join(sorted({r["budget"] for r in group})), str(budgets[system[0]]))
    by_atoms = {38: 400_000, 55: 1_000_000}
    for system in (*ASE_PANEL, *MORSE_PANEL):
        atoms = panel.SYSTEMS[system][0] if system in panel.SYSTEMS else nonlj.SPEC[system][0]
        audit.cell(f"ase panel {system}", "budget by atom count", str(by_atoms.get(atoms, 4_000_000)),
                   float(budgets[system]))
    audit.said(TEX, "on seeds 2000--2011, at \\(4\\times 10^{5}\\) calls for 38 atoms, \\(1\\times 10^{6}\\) for 55, "
                    "and \\(4\\times 10^{6}\\) for 98 atoms and above, on one core of one host.")
    audit.said("scripts/ase_panel/run_one.py", 'ASE_VERSION = "3.29.0"')
    audit.said("bench/ase-nonlj/README.md", "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults")
    audit.said(TEX, "Environment 3.29.0 at its constructor defaults")
    audit.said("scripts/ase_panel/run_one.py", "Au55 is Gupta::copper_gold through asg_eval serve, the same potential "
                                               "the anneal arms evaluate.")

    # tab:ase and the combined recommended summary.
    revision = sorted({r["code_revision"] for s in ASE_PANEL for r in groups[(s, "recommended")]})
    audit.record(len(revision) >= 1, TEX, "tab:ase recommended revisions", "at least one", str(len(revision)))
    rev_ase_text = ", ".join(revision)
    audit.said(TEX, f"This work is anneal \\texttt{{{rev_ase_text}}}.")
    by_label = {label: s for s, label in panel.TEX_LABELS.items()}
    shown = []
    for cells in tex_table(audit, TEX, "tab:ase"):
        system = by_label.get(cells[0])
        where = f"{TEX} tab:ase {system}"
        if system is None or len(cells) != 13:
            audit.record(False, where, "table line", " & ".join(cells), "an ASE panel system")
            continue
        shown.append(system)
        audit.cell(where, "budget", cells[1], float(budgets[system]))
        audit.cell(where, "cap", cells[2], ase_cap(audit, groups, system))
        for cell, arm in zip(cells[3:6], ("recommended", *ASE_ARMS)):
            audit.cell(where, f"{arm} hits", cell, frac(groups[(system, arm)]))
        for cell, arm in zip(cells[6:8], ASE_ARMS):
            audit.cell(where, f"{arm} wall stops", cell, float(stops(system, arm)))
        for cell, arm in zip(cells[8:11], ("recommended", *ASE_ARMS)):
            audit.cell(where, f"{arm} median wall", cell,
                       median([float(r["wall_s"]) for r in groups[(system, arm)]]))
        for cell, arm in zip(cells[11:13], ASE_ARMS):
            audit.cell(where, f"Fisher p against {arm}", cell, p(system, arm))
    audit.cell(TEX, "tab:ase lines", " ".join(shown), " ".join(ASE_PANEL))
    summary = "bench/ase-recommended/summary.csv"
    for line in audit.rows(summary):
        system, arm = line["system"], line["arm"]
        where = f"{summary} {system} {arm}"
        audit.cell(where, "hits", line["hits"], float(hits(system, arm)))
        audit.cell(where, "median best", line["median_best"], gap(system, arm, 0.0))
        audit.cell(where, "gap", line["gap_to_reference"], gap(system, arm, references[system]))
        if arm == "recommended":
            audit.cell(where, "fisher_p_bh", line["fisher_p_bh"], p(system, "ase_bh"))
            audit.cell(where, "fisher_p_mh", line["fisher_p_mh"], p(system, "ase_mh"))

    # tab:ase-morse and the Morse summary.
    revision = sorted({r["code_revision"] for s in MORSE_PANEL for r in groups[(s, "recommended")]})
    audit.record(len(revision) >= 1, TEX, "tab:ase-morse recommended revisions", "at least one", str(len(revision)))
    rev_morse_short = revision[0][:8]
    audit.said(TEX, f"This work is anneal \\texttt{{{rev_morse_short}}} at the length scale of the Morse pair.")
    audit.said("bench/ase-nonlj/README.md",
               "Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.")
    audit.said("examples/ase_panel.rs", "Config::recommended_with_scales(n, length, well)")
    by_label = {label: s for s, label in nonlj.TEX_LABELS.items()}
    shown = []
    for cells in tex_table(audit, TEX, "tab:ase-morse"):
        system = by_label.get(cells[0])
        where = f"{TEX} tab:ase-morse {system}"
        if system is None or len(cells) != 11:
            audit.record(False, where, "table line", " & ".join(cells), "a Morse panel system")
            continue
        shown.append(system)
        audit.cell(where, "budget", cells[1], float(budgets[system]))
        audit.cell(where, "cap", cells[2], ase_cap(audit, groups, system))
        for cell, arm in zip(cells[3:6], ("recommended", *ASE_ARMS)):
            audit.cell(where, f"{arm} hits", cell, frac(groups[(system, arm)]))
        for cell, arm in zip(cells[6:9], ("recommended", *ASE_ARMS)):
            audit.cell(where, f"{arm} median wall", cell,
                       median([float(r["wall_s"]) for r in groups[(system, arm)]]))
        for cell, arm in zip(cells[9:11], ASE_ARMS):
            audit.cell(where, f"Fisher p against {arm}", cell, p(system, arm))
    audit.cell(TEX, "tab:ase-morse lines", " ".join(shown), " ".join(MORSE_PANEL))
    summary = "bench/ase-nonlj/summary.csv"
    for line in pick(audit.rows(summary), code_revision=MORSE_REVISION):
        system, arm = line["system"], line["arm"]
        where = f"{summary} {system} {arm}"
        audit.cell(where, "hits", line["hits"], float(hits(system, arm)))
        audit.cell(where, "gap", line["gap_to_reference"], gap(system, arm, references[system]))
        if arm == "recommended":
            audit.cell(where, "fisher_p_bh", line["fisher_p_bh"], p(system, "ase_bh"))
            audit.cell(where, "fisher_p_mh", line["fisher_p_mh"], p(system, "ase_mh"))
    identity = audit.rows("bench/ase-nonlj/identity.csv")
    energy = max(float(r["abs_energy_diff"]) for r in identity)
    force = max(float(r["max_force_diff"]) for r in identity)
    audit.cell(TEX, "Morse energy agreement", "6.8e-12", energy)
    audit.cell(TEX, "Morse force agreement", "3.6e-12", force)
    audit.cell(TEX, "Morse agreement at the seed-2000 start", "2000", " ".join(sorted({r["seed"] for r in identity})))
    audit.said(TEX, "agrees with this code at the seed-2000 start to \\(6.8\\times 10^{-12}\\) in energy and "
                    "\\(3.6\\times 10^{-12}\\) in force.")

    # Stops: no run of this setting at a cap, the caps of each ASE arm.
    for system in (*ASE_PANEL, *MORSE_PANEL):
        audit.cell(f"ase panel {system}", "recommended wall stops", "0", float(stops(system, "recommended")))
        uncapped = sum(not r["cap_s"] for a in ("recommended", *ASE_ARMS) for r in groups[(system, a)])
        audit.cell(f"ase panel {system}", "table runs without a cap", "0", float(uncapped))
    audit.said(TEX, "Each run stops at a hit, at the budget, or at a wall cap. The cap of the basin-hopping and "
                    "minima-hopping runs is three times the slowest wall of the panel's cluster-preset arm on that "
                    "cluster.")
    audit.said(TEX, "No run of this setting reached its cap.")
    lj = ASE_PANEL[:5]

    def label(system: str) -> str:
        if system in panel.TEX_LABELS:
            return panel.TEX_LABELS[system]
        return nonlj.TEX_LABELS[system].replace(", \\(\\rho=", " at \\(\\rho=")

    mh_runs = sum(len(groups[(s, "ase_mh")]) for s in ASE_PANEL)
    mh_walls = sum(stops(s, "ase_mh") for s in ASE_PANEL)
    audit.cell(TEX, "tab:ase minima-hopping runs at the cap", str(mh_runs), float(mh_walls))
    bh = [(s, stops(s, "ase_bh")) for s in ASE_PANEL if stops(s, "ase_bh")]
    audit.said(TEX, f"Minima hopping stopped at the cap on all {mh_runs} runs of Table~\\ref{{tab:ase}}, and basin "
                    f"hopping on {sum(n for _, n in bh)}: {series(f'{n} on {label(s)}' for s, n in bh)}.")
    spent = [s for s in ASE_PANEL if stops(s, "ase_bh", "budget") == len(groups[(s, "ase_bh")])]
    for s in spent:
        audit.cell(TEX, f"{s} basin hopping charged the budget on every seed", "4000000",
                   float(min(int(r["charged"]) for r in groups[(s, "ase_bh")])))
    audit.said(TEX, f"On {series(label(s) for s in spent)} basin hopping spent the full \\(4\\times 10^{{6}}\\) calls "
                    "on every seed.")
    runs = []
    for s in lj:
        if runs and runs[-1][1] == hits(s, "recommended"):
            runs[-1][0].append(s)
        else:
            runs.append(([s], hits(s, "recommended")))
    audit.said(TEX, "This setting hits " + series(f"{' and '.join(label(s) for s in names)} on {n} of 12"
                                                  for names, n in runs) + ".")
    bh_hits = [s for s in ASE_PANEL if hits(s, "ase_bh")]
    mh_hits = [s for s in ASE_PANEL if hits(s, "ase_mh")]
    audit.cell(TEX, "basin hopping hits one cluster", "lj55", " ".join(bh_hits))
    audit.cell(TEX, "minima hopping hits none", "", " ".join(mh_hits))
    audit.said(TEX, f"Basin hopping hits {label('lj55')} on {hits('lj55', 'ase_bh')} of 12 and no other cluster, and "
                    "minima hopping hits none.")
    below = [s for s in lj if max(p(s, a) for a in ASE_ARMS) < 0.05]
    audit.said(TEX, f"Fisher's test against each ASE arm is below 0.05 on all {len(below)} Lennard-Jones clusters.")
    audit.cell(TEX, "Fisher below 0.05 on every Lennard-Jones cluster", " ".join(lj), " ".join(below))
    au_hits = [hits("au55", a) for a in ("anneal", "recommended", *ASE_ARMS)]
    audit.cell(TEX, "no arm of the ASE panel hits Au55", "0", float(sum(au_hits)))
    audit.said(TEX, f"No arm hits {label('au55')}.")
    au_ref = references["au55"]
    au_gaps = [f"{gap('au55', a, au_ref):.3f}" for a in ("recommended", *ASE_ARMS)]
    audit.said(TEX, f"There the median best energy lies {au_gaps[0]}~eV above the reference for this setting, "
                    f"{au_gaps[1]}~eV for basin hopping, and {au_gaps[2]}~eV for minima hopping.")
    morse_walls = all(r["stop"] == "wall" for s in MORSE_PANEL for a in ASE_ARMS for r in groups[(s, a)])
    audit.record(morse_walls, TEX, "every Morse ASE run stopped at the cap", "every", "every" if morse_walls else "not every")
    audit.said(TEX, "Every basin-hopping and minima-hopping run of Table~\\ref{tab:ase-morse} stopped at the cap.")
    m = [hits(s, "recommended") for s in MORSE_PANEL]
    audit.cell(TEX, "M38 at both ranges and M55 at rho=6 share one count", "1", float(len(set(m[:3]))))
    morse_sentence = (f"{label('m38r6').split(' at ')[0]} at \\(\\rho=6\\) and \\(\\rho=14\\) and {label('m55r6')} on "
                      f"{m[0]} of 12")
    audit.said(TEX, f"This setting hits {morse_sentence}, and {label('m55r14')} on {m[3]} of 12.")
    morse_ase = {hits(s, a) for s in MORSE_PANEL for a in ASE_ARMS}
    audit.cell(TEX, "Morse ASE hits", "0", float(max(morse_ase)))
    audit.said(TEX, f"Both ASE arms hit {max(morse_ase)} of 12 on all {len(MORSE_PANEL)}.")
    morse_p = [{show_p(p(s, a)) for a in ASE_ARMS} for s in MORSE_PANEL]
    audit.cell(TEX, "one Fisher p on the first three Morse clusters", "1", float(len(set.union(*morse_p[:3]))))
    audit.said(TEX, f"Fisher's \\(p\\) against each ASE arm is {next(iter(morse_p[0]))} on the first three and "
                    f"{next(iter(morse_p[3]))} on {label('m55r14')}.")
    section = audit.text(TEX).split("\\label{sec:ase}", 1)[1].split("\\subsection{TIP4P clusters}", 1)[0]
    audit.record("Mann" not in section, TEX, "sec:ase reports Fisher only", "no rank test", "no rank test"
                 if "Mann" not in section else "a rank test")
    discussion = audit.text(TEX).split("\\label{sec:discussion}", 1)[1].split("\\section{Conclusions}", 1)[0]
    audit.record("Mann" not in discussion, TEX, "the discussion reports Fisher only", "no rank test", "no rank test"
                 if "Mann" not in discussion else "a rank test")

    # Abstract, introduction, discussion and conclusions.
    lj_hits = [str(hits(s, "recommended")) for s in lj]
    ase_max = max(hits(s, a) for s in lj for a in ASE_ARMS)
    audit.said(TEX, f"this setting hits {series(label(s) for s in lj)} on {series(lj_hits)} of 12 seeds, and neither "
                    f"ASE arm hits more than {ase_max} of 12 on any of them.")
    gmin_au = gmin[("au55", "anneal")], gmin[("au55", "gmin")]
    rev_gmin = {r["code_revision"] for r in gmin_au[0]}
    rev_ase = {r["code_revision"] for r in groups[("au55", "recommended")]}
    rev_ase_short = sorted(rev_ase)[0][:8]
    overlap = rev_ase.intersection(rev_gmin)
    audit.record(bool(overlap), TEX, "the two Au55 counts share a revision",
                 "overlap", "overlap" if overlap else "disjoint")
    au_gmin = f"{hits_of(gmin_au[0])} of {len(gmin_au[0])}", f"{hits_of(gmin_au[1])} of {len(gmin_au[1])}"
    audit.said(TEX, f"it hits {series(str(n) for n in m)} of 12, and both ASE arms hit {max(morse_ase)} of 12.")
    audit.said(TEX, f"and against the basin hopping and the minima hopping of the Atomic Simulation Environment on "
                    f"{len(lj)} Lennard-Jones clusters, Gupta {label('au55')}, and {len(MORSE_PANEL)} Morse clusters "
                    "\\cite{wales1997,larsenAse2017,goedecker2004}.")
    audit.said(TEX, f"Section~\\ref{{sec:results}} runs this setting against GMIN on {len(gmin_panel.PAPER_BUDGETS)} "
                    "Lennard-Jones, Gupta, and Morse clusters,")
    mh_all = all(r["stop"] == "wall" for s in (*ASE_PANEL, *MORSE_PANEL) for r in groups[(s, "ase_mh")])
    audit.record(mh_all, TEX, "minima hopping reached its cap on every run", "every", "every" if mh_all else "not every")
    def partial(system: str) -> str:
        return f"on {stops(system, 'ase_bh')} of 12 on {label(system)}"

    full = [s for s in ASE_PANEL if stops(s, "ase_bh") == len(groups[(s, "ase_bh")])]
    part = [s for s in ASE_PANEL if 0 < stops(s, "ase_bh") < len(groups[(s, "ase_bh")])]
    audit.said(TEX, "Minima hopping reached its cap on every run, and basin hopping on every run of "
                    f"{series([*(label(s) for s in full), 'the Morse clusters'])} and "
                    f"{' and '.join(partial(s) for s in part)}.")
    missed = [s for s in spent if hits(s, "ase_bh") == 0]
    audit.cell(TEX, "basin hopping spent the budget and missed", " ".join(spent), " ".join(missed))
    audit.said(TEX, f"On {series(label(s) for s in spent)} basin hopping spent the budget before the cap and missed on "
                    "every seed.")
    audit.said(TEX, f"Against the Atomic Simulation Environment this setting hits {morse_sentence} and "
                    f"{label('m55r14')} on {m[3]} of 12, where neither ASE arm hits.")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(gmin[(system, arm)])} of {len(gmin[(system, arm)])}"

    def name(system: str) -> str:
        return gmin_panel.LABELS[system][1].replace(", \\(\\rho=", " at \\(\\rho=")

    morse_gmin = [s for s in gmin_panel.PAPER_BUDGETS if s.startswith("m")]
    behind = [s for s in morse_gmin if hits_of(gmin[(s, "anneal")]) < hits_of(gmin[(s, "gmin")])]
    level = [s for s in morse_gmin if s not in behind]
    def ours(system: str) -> str:
        return f"{name(system)} on {of(system, 'anneal')}"

    audit.said(TEX, f"Against GMIN it hits {series(ours(s) for s in level)}, where "
                    f"GMIN hits {series(str(hits_of(gmin[(s, 'gmin')])) for s in level)}, and it hits "
                    f"{series(ours(s) for s in behind)}, where GMIN hits "
                    f"{series(str(hits_of(gmin[(s, 'gmin')])) for s in behind)}.")
    rev_gmin_text = sorted(rev_gmin)[0][:8]
    audit.cell(TEX, "the GMIN panel revision", "1ce3ab64", rev_gmin_text)
    audit.said(TEX, f"{name('m55r14')} is in both panels: {hits('m55r14', 'recommended')} of 12 on revision "
                    f"\\texttt{{{rev_morse_short}}} and {of('m55r14', 'anneal')} on \\texttt{{{rev_gmin_text}}}.")
    p_au = fisher(hits_of(gmin_au[0]), len(gmin_au[0]), hits_of(gmin_au[1]), len(gmin_au[1]))
    audit.said(TEX, f"In the ASE panel it hits {hits('au55', 'recommended')} of 12, as do both ASE arms, and its median "
                    f"best energy lies {au_gaps[0]}~eV above the reference.")
    audit.said(TEX, f"Those two {label('au55')} counts come from revision \\texttt{{{rev_gmin_text}}} with the gold "
                    f"driver of Section~\\ref{{sec:gmin}} and from revision \\texttt{{{rev_ase_short}}} with the preset "
                    "scaled to the Gupta length and well depth.")
    step = 1e-4

    def curvature(pair) -> float:
        return (pair(1.0 + step) - 2.0 * pair(1.0) + pair(1.0 - step)) / step**2

    morse6 = curvature(lambda r: morse_pair(6.0, r))
    lennard_jones = curvature(lambda r: r**-12 - 2.0 * r**-6)
    audit.record(abs(morse6 - lennard_jones) < 1e-3, TEX, "Morse rho=6 curvature at the pair minimum",
                 f"{lennard_jones:.3f}", f"{morse6:.3f}")
    audit.said(TEX, "the Morse pair at \\(\\rho=6\\), whose well has the curvature of the Lennard-Jones well, and at "
                    "the shorter range \\(\\rho=14\\), and the many-body Gupta potential of gold.")
    ahead_lj = [s for s in lj if hits(s, "recommended") > max(hits(s, a) for a in ASE_ARMS)]
    ahead_morse = [s for s in MORSE_PANEL if hits(s, "recommended") > max(hits(s, a) for a in ASE_ARMS)]
    audit.cell(TEX, "ahead of both ASE arms on every Lennard-Jones and Morse cluster",
               " ".join((*lj, *MORSE_PANEL)), " ".join((*ahead_lj, *ahead_morse)))
    audit.said(TEX, f"this setting hits more seeds on all {len(ahead_lj)} Lennard-Jones clusters and all "
                    f"{len(ahead_morse)} Morse clusters, and no arm hits {label('au55')}.")
    lj75 = gmin[("lj75", "anneal")], gmin[("lj75", "gmin")]
    audit.said("paper/jctc/NUMBERS_TODO.md", f"In tab:gmin, seeds 2000-2011, the recommended preset hits "
                                             f"{hits_of(lj75[0])} of {len(lj75[0])}.")
    audit.said("paper/jctc/NUMBERS_TODO.md", f"And the Wales-group basin-hopping program (GMIN) hits {hits_of(lj75[1])} "
                                             f"of {len(lj75[1])}.")


def panel_references(audit: Audit) -> None:
    """The panel references of Section sec:hit, and every panel hit inside the window of equation eq:abs."""
    gmin = load_script("gmin_panel_refs", "scripts/gmin_panel/run_panel.py")
    panel = load_script("ase_panel_refs", "scripts/ase_panel/summarize.py")
    nonlj = load_script("ase_nonlj_refs", "scripts/ase_nonlj/summarize.py")
    refs: dict[str, set[float]] = defaultdict(set)
    for row in gmin.SYSTEMS:
        refs[row[0]].add(row[3])
    for system, spec in panel.SYSTEMS.items():
        refs[system].add(spec[2])
    for system, spec in nonlj.SPEC.items():
        refs[system].add(spec[3])
    for system, values in sorted(refs.items()):
        audit.cell("panel references", f"{system} one reference", "1", float(len(values)))
    ref = {s: next(iter(v)) for s, v in refs.items()}
    for path, rows in (("bench/gmin-panel/rows.csv", audit.rows("bench/gmin-panel/rows.csv")),
                       ("bench/ase-nonlj/rows.csv", audit.rows("bench/ase-nonlj/rows.csv")),
                       *((f"bench/ase-{s}/rows.csv", audit.rows(f"bench/ase-{s}/rows.csv")) for s in ASE_PANEL),
                       ("bench/ase-au55/rows-for-cluster.csv", audit.rows("bench/ase-au55/rows-for-cluster.csv"))):
        worst = max((abs(float(r["best"]) - ref[r["system"]]) for r in rows if hits_of([r])), default=0.0)
        audit.record(worst < 1e-3, path, "every hit within 1e-3 of its reference", f"{worst:.3g}", "below 0.001")

    def f6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    audit.said(TEX, f"The panels of Sections~\\ref{{sec:gmin}} and~\\ref{{sec:ase}} add {f6('lj102')} at 102 atoms, "
                    f"{f6('lj103')} at 103, and {f6('lj150')} at 150, \\({ref['au55']:.7f}\\)~eV for gold at 55 atoms "
                    f"in the copper--gold Gupta archive, and from the Morse table {f6('m38r6')} and {f6('m38r14')} for "
                    f"M\\(_{{38}}\\) at \\(\\rho=6\\) and \\(14\\), {f6('m55r6')} and {f6('m55r14')} for M\\(_{{55}}\\), "
                    f"and {f6('m80r6')} and {f6('m80r14')} for M\\(_{{80}}\\).")
    for system, text in (("lj38", "\\(-173.928427\\) at 38"), ("lj55", "\\(-279.248470\\) at 55"),
                         ("lj75", "\\(-397.492331\\) at 75"), ("lj98", "\\(-543.665361\\) at"),
                         ("lj104", "\\(-582.086642\\) at 104")):
        audit.cell(TEX, f"{system} reference", re.search(r"\\\((-[\d.]+)\\\)", text).group(1), ref[system])
        audit.said(TEX, text)


def generated_tables(audit: Audit) -> None:
    """Each panel table in the manuscript is the block its summariser writes from the committed rows."""
    text = audit.text(TEX)

    def block(marks: tuple[str, str]) -> str:
        start, end = marks
        if text.count(start) != 1 or text.count(end) != 1:
            return ""
        return text.split(start, 1)[1].split(end, 1)[0].strip("\n")

    gmin = load_script("gmin_panel_tex", "scripts/gmin_panel/run_panel.py")
    rows = gmin.read_rows(ROOT / "bench/gmin-panel/rows.csv")
    want = gmin.tex_table(gmin.summarise(rows, "latest"), "latest")
    panel = load_script("ase_panel_tex", "scripts/ase_panel/summarize.py")
    nonlj = load_script("ase_nonlj_tex", "scripts/ase_nonlj/summarize.py")
    ablation = load_script("ablation_tex", "paper/jctc/scripts/ablation_table.py")
    h2h = load_script("h2h_tables_tex", "paper/jctc/scripts/h2h_tables.py")
    si_text = audit.text(SI)
    for label, marks, written, source in (
            ("tab:gofee", h2h.MARKS["gofee"], h2h.gofee_block(), text),
            ("tab:tip4p-gmin", h2h.MARKS["tip4p"], h2h.tip4p_block(), text),
            ("tab:gofee-si", h2h.MARKS["gofee-si"], h2h.gofee_si_block(), si_text)):
        found = source.split(marks[0], 1)[1].split(marks[1], 1)[0].strip("\n") if marks[0] in source else ""
        ok = found == written
        audit.record(ok, SI if source is si_text else TEX, f"{label} is the generator's block",
                     "as written" if ok else "edited by hand", "as written")
    for label, marks, written in (("tab:ablation", ablation.TEX_MARKS, ablation.tex_table()),
                                  ("tab:gmin", gmin.TEX_MARKS, want),
                                  ("tab:ase", panel.TEX_MARKS, panel.tex_table()),
                                  ("tab:ase-morse", nonlj.TEX_MARKS, nonlj.tex_table(nonlj.load_rows()))):
        ok = block(marks) == written
        audit.record(ok, TEX, f"{label} is the summariser's block", "as written" if ok else "edited by hand", "as written")


def supporting_gmin(audit: Audit) -> None:
    """The SI sentence that sets the 64-seed GMIN rows apart from the panel."""
    seeds = pick(audit.rows("paper/jctc/data/lj55_lj98/seeds.csv"), system="lj98", arm="gmin")
    numbers = sorted(int(r["seed"]) for r in seeds)
    ceilings = sorted({r["ceiling"] for r in seeds})
    audit.said(SI, f"The GMIN arm hits {sum(r['hit'] == 'True' for r in seeds)} of {len(seeds)}.")
    audit.cell(SI, "64-seed GMIN ceilings", "1", float(len(ceilings)))
    audit.said(SI, f"Those GMIN rows use seeds {numbers[0]}--{numbers[-1]}, the starts of that campaign, and a ceiling "
                   f"of {ceilings[0]} calls.")
    audit.cell(SI, "64 consecutive seeds", " ".join(map(str, range(numbers[0], numbers[0] + len(numbers)))),
               " ".join(map(str, numbers)))
    _panel, groups = gmin_panel_groups(audit)
    g = groups[("lj98", "gmin")]
    audit.cell(SI, "panel LJ98 GMIN budget", "4000000", float(g[0]["budget"]))
    audit.said(SI, f"where GMIN hits LJ\\(_{{98}}\\) on {hits_of(g)} of {len(g)} seeds at \\(4\\times 10^{{6}}\\) calls.")


SI_TIP = "ce80337b7507ca12a99913da7ee8606184d4a3e1"
SI_ARCHIVE = "paper/jctc/logs/lj75-source-37535ec1.tar.xz"


def plain(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def numstat(old: bytes, new: bytes) -> tuple[int, int]:
    """Insertions and deletions that git reports between two versions of a file."""
    with tempfile.TemporaryDirectory() as scratch:
        left, right = Path(scratch, "old"), Path(scratch, "new")
        left.write_bytes(old)
        right.write_bytes(new)
        out = subprocess.run(["git", "diff", "--no-index", "--numstat", str(left), str(right)],
                             capture_output=True, text=True, check=False).stdout.split()
    return (int(out[0]), int(out[1])) if out else (0, 0)


def line_of(audit: Audit, path: str, number: int) -> str:
    return audit.text(path).splitlines()[number - 1]


def supporting_numbers(audit: Audit) -> None:
    """Every number of the SI, read from the row, log, archive or source file that holds it."""
    gmin = load_script("gmin_panel_si", "scripts/gmin_panel/run_panel.py")
    ref = {row[0]: row[3] for row in gmin.SYSTEMS}

    def r6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    # Board constants.
    audit.cell(SI, "hit tolerance of the h2h examples", "0.001", float(re.search(
        r'"lj38" => \(38, Kind::Lj, -173\.928427, ([\d.]+)\)', audit.text("examples/h2h_arm3.rs")).group(1)))
    audit.said(SI, "We read a hit as an energy strictly below the reference plus \\(0.001\\),")
    audit.said(SI, f"The references, in reduced units, are {r6('lj38')} for LJ\\(_{{38}}\\), {r6('lj55')} for "
                   f"LJ\\(_{{55}}\\), and {r6('lj75')} for LJ\\(_{{75}}\\). LJ\\(_{{98}}\\) uses {r6('lj98')}, "
                   f"and LJ\\(_{{104}}\\) uses {r6('lj104')}.")
    preset = "src/methods/cluster_hopping/preset.rs"
    audit.cell(SI, "temperature at preset.rs:8", "0.8",
               float(re.fullmatch(r"\s*pub\(super\) const TEMPERATURE: f64 = ([\d.]+);", line_of(audit, preset, 8)).group(1)))
    audit.said(SI, "The temperature is \\(0.8\\), the value Wales and Doye used for Lennard-Jones basin hopping and the "
                   "value stored at \\texttt{preset.rs:8}.")
    audit.said("examples/lj_cluster_search.rs", "random_cluster(n, 0.7, cfg.min_separation, &mut rng)")
    audit.said(SI, "The start density passed to \\texttt{random\\_cluster} is \\(0.7\\).")
    audit.said("src/methods/cluster_hopping.rs", "/// Seeds a non-overlapping configuration at liquid-like density.")
    audit.said(SI, "The function comment calls the draw liquid-like.")

    # Plain walk.
    log = "paper/jctc/logs/lj38-plain-seed0.log"
    macros = audit.text("paper/jctc/macros.tex")
    energy = re.search(r"\\newcommand\{\\PlainEnergy\}\{(-?[\d.]+)\}", macros).group(1)
    calls = re.search(r"\\newcommand\{\\PlainCalls\}\{(\d+)\}", macros).group(1)
    arms = [float(m) for m in re.findall(r"^\s+arm \S+\s+draws\s+\d+\s+accepts\s+\d+\s+best\s+(-?[\d.]+)", audit.text(log), re.M)]
    audit.cell(SI, "plain-walk best energy", energy, min(arms))
    audit.said(log, f"LJ38, budget {calls} charged evaluations, 1 seeds, reference {ref['lj38']:.6f}")
    audit.said(log, f"lj_cluster_search 38 {calls} 1 rec,noclimb,nosym")
    audit.said(log, "SEED_OFFSET=0")
    audit.said(SI, "with the stall exits and the Bayes screen turned off, seed 0, budget 30000.")
    search = "examples/lj_cluster_search.rs"
    audit.said(search, "cfg.escape_on_stall = (cfg.escape_on_stall || climb) && !noclimb;")
    audit.said(search, "cfg.symmetrise_on_stall = (cfg.symmetrise_on_stall || sym) && !nosym;")
    audit.said(search, 'cfg.bayes_screen = opts.contains(&"bayes");')
    audit.cell(SI, "plain-walk budget", "30000", float(calls))
    audit.said(SI, "The best energy in that log is \\PlainEnergy, after \\PlainCalls{} charged evaluations.")

    # Published symmetrised costs.
    lit = audit.rows("paper/jctc/data/literature.tsv", "\t")
    csm = pick(lit, method="GMIN CSM BH", N="38")[0]
    audit.said(SI, f"The continuous-symmetry basin-hopping mean at 38 atoms is {csm['reported_cost']} evaluations over "
                   f"{csm['hits']} successful runs.")
    audit.cell(SI, "successful runs equal runs", csm["runs"], float(csm["hits"]))
    audit.cell(SI, "cost unit", "energy/gradient evaluations", csm["cost_unit"])
    ga = sorted(r["N"] for r in lit if r["method"].endswith(" GA") and r["citation"].startswith("Oakley"))
    audit.said(SI, "quotes Oakley, Johnston and Wales, Table 1.")
    audit.cell(SI, "genetic-algorithm rows of Table 2", "75 98", " ".join(ga))
    audit.said(SI, "The Table 2 rows of that source are genetic-algorithm means at 75 and 98 atoms. "
                   "It has no genetic-algorithm row at 104 atoms.")
    audit.record("104" not in ga, SI, "no Oakley genetic-algorithm row at 104 atoms", "none", " ".join(ga))

    # TIP4P pair constants.
    constants = audit.text("paper/jctc/logs/tip4p_constants.txt")
    a_j, c_j = re.search(r"Jorgensen A = ([\d.]+) kJ A\^12/mol, C = ([\d.]+) kJ A\^6/mol", constants).groups()
    eps, sigma = (float(v) for v in re.search(r"anneal epsilon = ([\d.]+), sigma = ([\d.]+)", constants).groups())
    audit.cell(SI, "Jorgensen A", "2510400", float(a_j))
    audit.cell(SI, "Jorgensen C", "2552.24", float(c_j))
    audit.said(SI, "records Jorgensen's \\(A = 2510400\\) and \\(C = 2552.24\\).")
    audit.said("paper/jctc/logs/tip4p_constants.txt", "GMIN C6 = 2552.24, C12 = 2.5104e+06: equal to Jorgensen C and A: True")
    audit.said(SI, "The Wales-group basin-hopping program (GMIN) matches those two numbers in \\(C_6\\) and \\(C_{12}\\).")
    audit.cell(SI, "anneal epsilon and sigma", "0.6485 3.15365", f"{eps:g} {sigma:g}")
    audit.said(SI, f"anneal uses \\(\\varepsilon = {eps:g}\\)~kJ/mol and \\(\\sigma = {sigma:g}\\)~\\AA.")
    c6, c12 = 4 * eps * sigma**6, 4 * eps * sigma**12
    audit.said(SI, f"Its C6 is \\({c6:.4f}\\) and its C12 is \\({c12:.1f}\\).")
    audit.said(SI, f"give the anneal pair \\(C_6 = {c6:.4f}\\) and \\(C_{{12}} = {c12:.1f}\\).")
    audit.said("paper/jctc/logs/tip4p_constants.txt", f"C6 = {c6:.4f}, C12 = {c12:.1f}")
    audit.said(SI, f"It uses the Jorgensen TIP4P constants C6 \\(= {float(c_j):g}\\) and C12 \\(= {float(a_j):.0f}\\).")
    tip4p = audit.text("src/potentials/tip4p.rs")
    audit.said("src/potentials/tip4p.rs", "the rounded 0.6485 leaves every cluster")
    audit.record(f"{eps:g}" in tip4p, "src/potentials/tip4p.rs", "epsilon is in the source", f"{eps:g}", "present")

    # Earlier campaign.
    scores = audit.rows("paper/jctc/data/campaign-scores.csv")
    counts = defaultdict(set)
    for r in scores:
        counts[r["seeds"]].add(r["board"])
    audit.cell(SI, "campaign seed counts", "16 64", " ".join(sorted(counts, key=int)))
    audit.cell(SI, "the 16-seed board", "tip4p20", " ".join(sorted(counts["16"])))
    audit.said(SI, "is an earlier 64-seed campaign, and on TIP4P\\(_{20}\\) a 16-seed campaign.")
    audit.record(any(r["code"] == "abcluster" for r in scores), SI, "ABCluster is in the campaign file", "present", "present")
    seeds = audit.rows("paper/jctc/data/lj55_lj98/seeds.csv")
    lj98 = {arm: sorted(pick(seeds, system="lj98", arm=arm), key=lambda r: int(r["seed"])) for arm in ("full", "gmin")}

    def par2(row: dict[str, str]) -> float:
        return float(row["hit_call"]) if row["hit"] == "True" else 2.0 * float(row["ceiling"])

    full, gm = ([par2(r) for r in lj98[a]] for a in ("full", "gmin"))
    for arm, values, sentence in (("full", full, "The full arm hits {h} of {n}."), ("gmin", gm, "The GMIN arm hits {h} of {n}.")):
        rows = lj98[arm]
        audit.said(SI, sentence.format(h=hits_of(rows), n=len(rows)))
        audit.said(SI, f"Its call PAR-2 median is {plain(median(values))} and its type-7 interquartile range is "
                       f"{plain(type7(values, 0.75) - type7(values, 0.25))}.")
    rng = random.Random(2)
    ratios = []
    for _ in range(2000):
        pick_ = [rng.randrange(len(full)) for _ in full]
        ratios.append(median([full[i] for i in pick_]) / median([gm[i] for i in pick_]))
    bootstrap = "paper/jctc/logs/paired_bootstrap_lj98.txt"
    audit.said(bootstrap, "resamples=2000")
    audit.said(bootstrap, "rng=random.Random(2)")
    audit.cell(SI, "median ratio", "12.62", median(full) / median(gm))
    audit.cell(SI, "bootstrap lower endpoint", "8.52", type7(ratios, 0.025))
    audit.cell(SI, "bootstrap upper endpoint", "15.22", type7(ratios, 0.975))
    audit.said(SI, "interval of the median ratio for the 24-of-64 hop, 12.62 with endpoints 8.52 and 15.22, from 2000 resamples.")
    lj75 = {name: len(audit.rows(f"paper/jctc/data/lj75_gmin/{name}.csv")) for name in
            ("gmin-lj75-sym-starts-1-200", "lj75-recommended-seeds-2000-2199", "lj75-derived-seeds-2000-2199")}
    audit.cell(SI, "the LJ75 campaign has 200 seeds in each arm", "200", float(min(lj75.values())))
    audit.cell(SI, "campaign-scores LJ75 seeds", "64", float(max(int(r["seeds"]) for r in scores if r["board"] == "lj75")))
    audit.said(SI, "None of those rows is the LJ\\(_{75}\\) campaign of 200 seeds.")

    # Hardware and reproduction.
    receipt = audit.text("paper/jctc/logs/lj75-hardware-receipt.txt")
    tree = re.search(r"^TREE ([0-9a-f]{40})$", receipt, re.M).group(1)
    audit.said(SI, f"\\texttt{{\\seqsplit{{{tree}}}}}, the tree of the unpublished build commit.")
    audit.said(SI, f"We keep the source archive at \\path{{logs/lj75-source-{tree[:8]}.tar.xz}}.")
    with tarfile.open(ROOT / SI_ARCHIVE) as archive:
        def member(name: str) -> bytes:
            return archive.extractfile(name).read()

        audit.record(b"Copyright (c) 2023 anneal core developers" in member("LICENSE.txt"), SI,
                     "the archive notice", "Copyright (c) 2023 anneal core developers", "present")
        audit.said(SI, "The notice reads Copyright (c) 2023 anneal core developers.")
        listing = audit.text("paper/jctc/logs/lj75-source-37535ec1.lst")
        audit.record("\tLICENSE.txt" in listing, SI, "the archive includes LICENSE.txt", "present", "present")
        want = {"src/methods/cluster_hopping.rs": (218, 2), "src/potentials/tip4p.rs": (93, 15),
                "src/rigid_body.rs": (52, 14), "examples/molecular_cluster.rs": (1, 0),
                "Cargo.toml": (4, 1), "Cargo.lock": (0, 55)}
        for path, shown in want.items():
            new = git_bytes("show", f"{SI_TIP}:{path}")
            got = numstat(member(path), new)
            audit.record(got == shown, SI, f"{path} against {SI_TIP[:8]}", f"+{shown[0]} -{shown[1]}", f"+{got[0]} -{got[1]}")
    audit.said(SI, "Against commit \\texttt{ce80337b}, \\path{cluster_hopping.rs} differs by 218 insertions and 2 "
                   "deletions, \\texttt{tip4p.rs} by 93 and 15, \\texttt{rigid\\_body.rs} by 52 and 14, "
                   "\\path{examples/molecular_cluster.rs} by 1 insertion, \\texttt{Cargo.toml} by 4 insertions and "
                   "1 deletion, and \\texttt{Cargo.lock} by 55 deletions.")
    with tarfile.open(ROOT / "paper/jctc/logs/logs-water.tar.xz") as water:
        versions = water.extractfile("water/VERSIONS").read().decode()
    digest = re.search(r"^engine_sha256=([0-9a-f]{64})$", versions, re.M).group(1)
    audit.said(SI, f"\\texttt{{\\seqsplit{{{digest}}}}}.")

    # Landscape and fold captions.
    directory = "paper/jctc/data/landscapes_7695255"
    best: dict[str, float] = {}
    for name in ("lj38_bh_s0", "lj38_rss_s0", "lj38_recommended_s0", "lj75_bh_s0", "lj75_recommended_s0",
                 "lj98_bh_s0", "lj98_recommended_s0"):
        energies = [float(line.split()[0]) for line in (ROOT / directory / f"{name}.min").read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
        best[name] = min(energies) - ref[name.split("_")[0]]
    gap = {k: f"{v:.3f}" for k, v in best.items()}
    audit.record(abs(best["lj38_recommended_s0"]) < 1e-3, SI, "LJ38 recommended reaches the fcc minimum",
                 "within 0.001", f"{best['lj38_recommended_s0']:.2e}")
    audit.said(SI, f"Basin hopping sits \\({gap['lj38_bh_s0']}\\) above it. Random structure search sits "
                   f"\\({gap['lj38_rss_s0']}\\) above it.")
    audit.said(SI, f"Basin hopping and the recommended hop both sit \\({gap['lj75_bh_s0']}\\) above it.")
    audit.cell(SI, "LJ75 arms agree", gap["lj75_bh_s0"], best["lj75_recommended_s0"])
    audit.said(SI, f"The recommended hop sits \\({gap['lj98_recommended_s0']}\\) above it. Basin hopping sits "
                   f"\\({gap['lj98_bh_s0']}\\) above it.")
    audit.said(SI, f"basin hopping stops \\({gap['lj38_bh_s0']}\\) above it.")
    audit.said(SI, f"both arms stop \\({gap['lj75_bh_s0']}\\) above the Marks decahedron.")
    audit.said(SI, f"the recommended hop stops \\({gap['lj98_recommended_s0']}\\) above the Leary minimum and basin "
                   f"hopping \\({gap['lj98_bh_s0']}\\) above it.")
    report = "paper/jctc/REPORT.md"
    audit.said(report, f"Basin hopping stops {gap['lj38_bh_s0']} above it.")

    # Copper slab.
    script = "scripts/elja_hq_sci_molslab_bank.sh"
    audit.cell(SI, "TARGET_ENERGY line", "119", float(next(
        i for i, text in enumerate(audit.text(script).splitlines(), 1) if text == "export TARGET_ENERGY=-415.971529")))
    audit.cell(SI, "TARGET_TOL line", "120", float(next(
        i for i, text in enumerate(audit.text(script).splitlines(), 1) if i > 119 and text == "export TARGET_TOL=1e-3")))
    audit.said(SI, "Its line 119 sets \\texttt{TARGET\\_ENERGY} to \\(-415.971529\\) and its line 120 sets "
                   "\\texttt{TARGET\\_TOL} to \\(10^{-3}\\).")
    tally = "paper/jctc/scripts/hq_sci_tally.py"
    audit.said(tally, "scripts/elja_hq_sci_molslab_bank.sh line 119 sets it too.")
    audit.cell(tally, "the jcc script's target line", "80", float(next(
        i for i, text in enumerate(audit.text("scripts/elja_jcc_molslab_ensemble.sh").splitlines(), 1)
        if text.strip() == "TARGET=-415.971529")))
    audit.said(tally, "scripts/elja_jcc_molslab_ensemble.sh line 80 sets that target.")
    bests = []
    for index in range(8):
        found = re.search(r"seed \d+: best (\S+) eV", audit.text(f"paper/jctc/logs/cuh2_fcc_rec_{index}.out"))
        bests.append(float(found.group(1)))
    gaps = [b + 415.971529 for b in bests]
    audit.cell(SI, "smallest copper gap", "8e-6", min(gaps))
    audit.cell(SI, "largest copper gap", "3.68e-4", max(gaps))
    audit.cell(SI, "copper seeds inside the tolerance", "8", float(sum(g < 1e-3 for g in gaps)))
    audit.cell(SI, "copper seeds inside 1e-6 eV", "0", float(sum(g < 1e-6 for g in gaps)))
    audit.said(SI, "The per-seed gaps run from \\(8\\times 10^{-6}\\)~eV to \\(3.68\\times 10^{-4}\\)~eV above that reference.")
    audit.said(SI, "No seed reached the reference within \\(10^{-6}\\)~eV.")
    audit.said(SI, "Eight seeds lie inside the 0.001~eV tolerance.")
    audit.said(SI, "The hit tolerance is 0.001~eV.")


def manuscript_tip4p(audit: Audit) -> None:
    rows = audit.rows("bench/tip4p-water-rates/rates-a24ba760.csv")
    ten = sorted(pick(rows, n="10", budget="500000"), key=lambda r: int(r["seed"]))
    audit.said(TEX, f"\\((\\mathrm{{H_2O}})_{{10}}\\) hits on {frac(ten).replace('/', ' of ')} seeds.")
    calls = [r["calls_to_first_hit"] for r in ten]
    audit.said(TEX, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    for n in ("15", "20"):
        group = pick(rows, n=n, budget="500000")
        audit.cell(TEX, f"tip4p ({n}) hits at 5e5", "0/6", frac(group))
        gaps = [float(r["best_kj"]) - float(r["reference_kj"]) for r in group]
        word = {"15": "fifteen", "20": "twenty"}[n]
        audit.said(TEX, f"The best {word}-molecule energies lie \\({min(gaps):.10f}\\) to \\({max(gaps):.10f}\\)~kJ/mol above")


def manuscript_hexamer(audit: Audit) -> None:
    numbers = "paper/jctc/logs/water/NUMBERS.tsv"
    lines = [line.rstrip("\n").split("\t") for line in (ROOT / numbers).read_text().splitlines()]
    header = next(cells for cells in lines if cells[0] == "# seed")
    header[0] = "seed"
    seeds = [dict(zip(header, cells)) for cells in lines if cells[0].isdigit()]
    audit.read[numbers] = seeds
    by = {(r["seed"], r["arm"]): r for r in seeds}
    for row in tex_table(audit, TEX, "tab:crest-counters"):
        seed = row[0]
        log = ROOT / f"paper/jctc/logs/water/crest-hexamer-seed{seed}.log"
        engrad = re.search(r"Total number of energy\+grad calls:\s*(\d+)", log.read_text()).group(1)
        audit.cell(f"{TEX} tab:crest-counters seed {seed}", "old counter", row[1], float(by[(seed, "crest")]["pes_calls"]))
        audit.cell(f"{TEX} tab:crest-counters seed {seed}", "energy+gradient", row[2], float(engrad))
        audit.cell(f"{TEX} tab:crest-counters seed {seed}", "fair cap = old counter", row[1], float(by[(seed, "fair")]["pes_calls"]))
    for row in tex_table(audit, TEX, "tab:hexamer-gaps"):
        seed = row[0]
        where = f"{TEX} tab:hexamer-gaps seed {seed}"
        audit.cell(where, "gap 4000", row[1], float(by[(seed, "paper4000")]["signed_dE_polished_Eh"]))
        audit.cell(where, "gap old-counter cap", row[2], float(by[(seed, "fair")]["signed_dE_polished_Eh"]))
        audit.cell(where, "deviation 4000", row[3], float(by[(seed, "paper4000")]["rmsd_label_free_A"]))
        audit.cell(where, "deviation old-counter cap", row[4], float(by[(seed, "fair")]["rmsd_label_free_A"]))
    old = [float(by[(str(s), "crest")]["pes_calls"]) for s in range(6)]
    audit.said(TEX, f"from {min(old):.0f} to {max(old):.0f} evaluations.")
    engrads = [float(re.search(r"energy\+grad calls:\s*(\d+)", (ROOT / f"paper/jctc/logs/water/crest-hexamer-seed{s}.log").read_text()).group(1)) for s in range(6)]
    audit.said(TEX, f"CREST's energy and gradient totals run from {min(engrads):.0f} to {max(engrads):.0f}.")
    audit.cell(TEX, "the old counter is smaller on every seed", "6", float(sum(o < e for o, e in zip(old, engrads))))
    audit.cell(TEX, "the 4000-call arm spends 4000", "4000", max(float(by[(str(s), "paper4000")]["pes_calls"]) for s in range(6)))
    within = [s for s in range(6) if all(abs(float(by[(str(s), a)]["signed_dE_polished_Eh"])) <= 1e-7 for a in ("paper4000", "fair"))]
    audit.cell(TEX, "seeds within 1e-7 Eh on both ledgers", "5", float(len(within)))
    audit.cell(TEX, "the seed outside 1e-7 Eh", "1", float(next(s for s in range(6) if s not in within)))
    dev = max(float(by[(str(s), a)]["rmsd_label_free_A"]) for s in range(6) for a in ("paper4000", "fair"))
    audit.said(TEX, f"most \\({dev:.5f}\\)~\\AA.")
    for arm, text in (("crest", "\\(544.0\\)~s for CREST"), ("paper4000", "\\(18.8\\)~s for the 4000-call arm"), ("fair", "\\(195.5\\)~s for the arm")):
        audit.cell(TEX, f"{arm} wall median", re.search(r"\\\((\d+\.\d)\\\)", text).group(1), median([float(by[(str(s), arm)]["wall_s"]) for s in range(6)]))
        audit.said(TEX, text)

    repolish = audit.rows("paper/jctc/data/hexamer-xtb680-repolish.csv")
    shifts = sorted((abs(float(r["new_energy_Eh"]) - float(r["old_energy_Eh"])), r["structure"], r["seed"]) for r in repolish)
    audit.cell(TEX, "xtb 6.8.0 finals", "18", float(len(shifts)))
    audit.cell(TEX, "largest xtb 6.8.0 shift", "1.08e-7", shifts[-1][0])
    audit.cell(TEX, "largest shift is the CREST final of seed 1", "crest 1", f"{shifts[-1][1]} {shifts[-1][2]}")
    audit.cell(TEX, "smallest of the other seventeen shifts", "0.9e-8", shifts[0][0])
    audit.cell(TEX, "largest of the other seventeen shifts", "2.0e-8", shifts[-2][0])
    gap = {(r["structure"], r["seed"]): float(r["gap_Eh"]) for r in repolish}
    audit.cell(TEX, "xtb 6.8.0 seed 1 gap, 4000 calls", "-4.35e-8", gap[("paper4000", "1")])
    audit.cell(TEX, "xtb 6.8.0 seed 1 gap, old counter", "-4.34e-8", gap[("fair", "1")])
    for arm in ("paper4000", "fair"):
        inside = sum(abs(gap[(arm, str(s))]) <= 1e-7 for s in range(6))
        audit.cell(TEX, f"xtb 6.8.0 {arm} seeds within 1e-7 Eh", "6", float(inside))
    for text in ("\\(-4.35\\times 10^{-8}\\)~Eh on the 4000-call ledger", "\\(-4.34\\times 10^{-8}\\)~Eh on the old-counter ledger",
                 "\\(1.08\\times 10^{-7}\\)~Eh.", "\\(0.9\\times 10^{-8}\\) to \\(2.0\\times 10^{-8}\\)~Eh"):
        audit.said(TEX, text)

    campaign = audit.rows("paper/jctc/data/water_v45b/water-v45b-perseed.csv")
    audit.said(TEX, f"hits {hits_of(campaign, 'anneal_hit')} of {len(campaign)} seeds for the default hop and {hits_of(campaign, 'crest_hit')} of {len(campaign)} for CREST")
    audit.said(TEX, f"The motif match remains {hits_of(campaign, 'anneal_hit')} of {len(campaign)} for the default hop and {hits_of(campaign, 'crest_hit')} of {len(campaign)}")


def manuscript_cross_field(audit: Audit) -> None:
    rows = audit.rows("results/cross-field-hits.csv")
    stalls = audit.rows("results/lj-funnel-stalls.csv")
    rates = audit.rows("bench/tip4p-water-rates/rates.csv")
    lj98 = sorted(pick(rows, system="lj", n="98", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    calls = [r["calls_to_first_hit"] for r in lj98]
    audit.said(TEX, f"LJ\\(_{{98}}\\) hit on {frac(lj98).replace('/', ' of ')}.")
    audit.said(TEX, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    audit.said(TEX, f"The median is {median([float(c) for c in calls]):.0f}.")
    retry98 = pick(stalls, phase="retry", n="98")
    lj104 = sorted(pick(rows, system="lj", n="104", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    hit104 = [r for r in lj104 if r["hit"] == "1"]
    retry104 = sorted((r["seed"] for r in pick(stalls, phase="retry", n="104") if r["hit"] == "1"), key=int)
    retry_median = median([float(r["calls_to_hit"]) for r in retry98])
    arm_median = median([float(c) for c in calls])
    audit.record(retry_median < arm_median, TEX, "the retry median on LJ98 is lower than the arm median",
                 "lower", "lower" if retry_median < arm_median else "not lower")
    audit.said(TEX, f"By comparison, the pentagonal-retry baseline also hits {frac(retry98).replace('/', ' of ')} on "
                    f"LJ\\(_{{98}}\\) with a lower median of {retry_median:.0f} calls, and on LJ\\(_{{104}}\\) reaches "
                    f"{len(retry104)} of 6 seeds ({retry104[0]}, {retry104[1]}, and {retry104[2]}) versus "
                    f"{len(hit104)} of {len(lj104)} ({hit104[0]['seed']} and {hit104[1]['seed']}, at "
                    f"{hit104[0]['calls_to_first_hit']} and {hit104[1]['calls_to_first_hit']} calls) for the BIPOP variant.")
    only_arm = sorted({r["seed"] for r in hit104} - set(retry104))
    only_retry = sorted(set(retry104) - {r["seed"] for r in hit104})
    audit.cell(TEX, "the hit the retry rows miss", "6401", " ".join(only_arm))
    audit.cell(TEX, "hits the retry rows record and this arm misses", "6402 6404", " ".join(only_retry))
    audit.said(TEX, f"and the rescaled arm hit {len(hit104)} of 6 and was not kept.")
    for n, budget in (("15", "500000"), ("20", "2000000")):
        trust = pick(rows, system="tip4p", n=n, arm="cage-trust")
        audit.cell(TEX, f"cage-trust ({n}) hits", "0/6", frac(trust))
        audit.cell(TEX, f"cage-trust ({n}) budget", budget, float(trust[0]["budget"]))
        rate = pick(rates, n=n, budget=budget)
        audit.cell(TEX, f"rate file ({n}) at {budget}", "1/6", frac(rate))
        audit.cell(TEX, f"rate file ({n}) hit seed", {"15": "3", "20": "2"}[n], " ".join(r["seed"] for r in rate if r["hit"] == "1"))
    best = min(pick(rows, system="tip4p", n="20", arm="cage-trust"), key=lambda r: float(r["best"]))
    audit.said(TEX, f"\\({float(best['best']):.6f}\\)~kJ/mol, which is \\({float(best['best']) - float(best['reference']):.3f}\\)~kJ/mol above")
    ten = pick(audit.rows("bench/tip4p-water-rates/rates-a24ba760.csv"), n="10")
    audit.said(TEX, f"On TIP4P, 10 molecules hit on {frac(ten).replace('/', ' of ')} seeds.")


def supporting_information(audit: Audit) -> None:
    scores = audit.rows("paper/jctc/data/campaign-scores.csv")
    ogolem = pick(scores, board="lj104", code="ogolem")[0]
    audit.said(SI, f"OGOLEM, cited as Dieterich and Hartke, is in the campaign file at {ogolem['hits']} of {ogolem['seeds']}")
    eps, sigma = 0.6485, 3.15365
    audit.said(SI, f"\\(C_6 = {4 * eps * sigma**6:.4f}\\) and \\(C_{{12}} = {4 * eps * sigma**12:.1f}\\).")
    refs = audit.rows("paper/jctc/data/system_visuals/references.csv")
    hexamer = next(r for r in refs if r.get("label") == "TIP4P-6" or "TIP4P-6" in ",".join(r.values()))
    energy = next(v for v in hexamer.values() if v and re.fullmatch(r"-\d+\.\d+", v))
    audit.said(SI, f"The (H\\(_2\\)O)\\(_6\\) plate prints the energy \\({float(energy):.4f}\\)~kJ/mol.")


# --------------------------------------------------------------------------- bench READMEs


def paper_0120(audit: Audit) -> None:
    spec = importlib.util.spec_from_file_location("run_panel", ROOT / "scripts/paper_0120/run_panel.py")
    panel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(panel)
    path = "bench/paper-0120/README.md"
    rows = audit.rows("bench/paper-0120/rows.csv")
    groups = panel.release_groups(rows, "07f89cadbd9ccd8386e92408a8ff0e8153b180e3")
    names = {"LJ75, `h2h_seed`": ("lj75", "h2h_seed"), "LJ103": ("lj103", "h2h_gmin_proof+lj103"),
             "Au55": ("au55", "h2h_metal"), "M80 ρ=6": ("m80r6", "h2h_morse"), "M80 ρ=14": ("m80r14", "h2h_morse")}
    for line, row in md_table(audit, path, "system | budget | 0.12.0 hits"):
        system, driver = names.get(row["system"], (row["system"].lower(), "h2h_gmin_proof"))
        group = groups[(system, driver)]
        dev = panel.DEVELOP[(system, driver)]
        found = pick(audit.rows(dev["path"]), **dev["where"])
        if dev["path"] == "bench/morse-panel/seeds.csv":
            found.extend(pick(audit.rows("bench/morse-panel/template-free-rows.csv"), **dev["where"]))
        found = [r for r in found if r["system"] == system]
        where = f"{path}:{line} {row['system']}"
        calls = [float(r["evals_to_hit"]) for r in group]
        mine = hits_of(group)
        audit.cell(where, "budget", row["budget"], float(group[0]["budget"]))
        audit.cell(where, "0.12.0 hits", row["0.12.0 hits"], f"{mine}/{len(group)}")
        audit.cell(where, "median calls 0.12.0", row["median calls 0.12.0"], median(calls))
        for shown, key in zip(row["0.12.0 wall / inside / outside (s)"].split(" / "), ("wall_s", "pot_s", "outside_s")):
            audit.cell(where, key, shown, median([float(r[key]) for r in group]))
        if "summary" in dev:
            base = found[0]
            dev_hits, dev_n = int(base["hits"]), int(base["n"])
            audit.cell(where, "median calls develop", row["median calls develop"], float(base["evals_median"]))
            audit.cell(where, "MW p", row["MW p"], None)
            audit.cell(where, "same seeds", row["same seeds"], None)
        else:
            by_seed = {int(r["seed"]): r for r in found}
            base = [by_seed[s] for s in panel.SEEDS]
            dev_hits, dev_n = sum(panel.is_hit(r["hit"]) for r in base), len(base)
            base_calls = [float(r[dev["calls"]]) for r in base]
            audit.cell(where, "median calls develop", row["median calls develop"], median(base_calls))
            audit.cell(where, "MW p", row["MW p"], mann_whitney(calls, base_calls, continuity=True))
            same = sum(panel.same_seed(r, by_seed[int(r["seed"])], dev["calls"]) for r in group)
            audit.cell(where, "same seeds", row["same seeds"], float(same))
        audit.cell(where, "develop hits", row["develop hits"], f"{dev_hits}/{dev_n}")
        audit.cell(where, "Fisher p", row["Fisher p"], fisher(mine, len(group), dev_hits, dev_n))


def gad_published(audit: Audit) -> None:
    path = "bench/gad-published/README.md"
    rows = audit.rows("bench/gad-published/rows.csv")
    columns = {"Median calls to hit": "evals_median", "IQR": "evals_iqr", "Best (median)": "best_median",
               "Wall (s)": "wall_median_s", "Pot (s)": "pot_median_s", "Outside (s)": "outside_median_s",
               "Inside µs/call": "inside_us", "Outside µs/call": "outside_us"}
    for line, row in md_table(audit, path, "Size | Arm | Hits"):
        system = row["Size"].lower()
        line_rows = pick(rows, system=system, arm=row["Arm"])
        base = line_rows[0]
        where = f"{path}:{line} {system} {row['Arm']}"
        audit.cell(where, "hits", row["Hits"], f"{base['hits']}/{base['n']}")
        for label, key in columns.items():
            audit.cell(where, label, row[label], float(base[key]))
        if row["Arm"] == "anneal":
            audit.cell(where, "Fisher p", row["Fisher p"], None)
            continue
        anneal = pick(rows, system=system, arm="anneal")[0]
        p = fisher(int(base["hits"]), int(base["n"]), int(anneal["hits"]), int(anneal["n"]))
        audit.cell(where, "Fisher p (recomputed)", row["Fisher p"], p)
        audit.cell(where, "Fisher p (rows.csv)", row["Fisher p"], float(base["fisher_p"]))
        audit.cell(where, "Mann-Whitney p (rows.csv)", row["Mann–Whitney p"], float(base["mannwhitney_p"]))
    for row in pick(rows, arm="gad_deposit"):
        gen, lo, hi = row["hit_generation"], row["evals_lower"], row["evals_upper"]
        audit.cell(path, f"{row['system']} lower bound, generations 0..{gen} x 64", lo, float((int(gen) + 1) * 64))
        audit.cell(path, f"{row['system']} upper bound, x 1000 FIRE steps", hi, float((int(gen) + 1) * 64 * 1000))
        size = row["system"].removeprefix("lj")
        audit.said(path, f"generation {gen} for LJ{size} (bounds {lo} and {hi} calls)")


OPENCONF_ARM = {"branch": "anneal", "anneal": "anneal", "develop": "develop", "openconf": "openconf"}


def openconf(audit: Audit) -> None:
    base = "bench/openconf-h2h"
    tables = [
        ("paired/README.md", 0, "paired/seeds.csv"),
        ("conf-fix/README.md", 0, "conf-fix/seeds.csv"),
        ("conf-fix3/README.md", 0, "conf-fix3/seeds.csv"),
        ("conf-fix4/README.md", 0, "conf-fix4/seeds.csv"),
        ("conf-fix4/README.md", 85, "conf-fix4/n24/seeds.csv"),
    ]
    for readme, after, source in tables:
        path = f"{base}/{readme}"
        by = defaultdict(list)
        for r in audit.rows(f"{base}/{source}"):
            by[(r["molecule"], r["arm"])].append(r)
        heads = ("molecule | hits", "molecule | calls to hit", "molecule | wall")
        for head in heads:
            try:
                table = md_table(audit, path, head, after)
            except SystemExit:
                continue
            if after == 0 and readme == "conf-fix4/README.md" and table[0][0] > 85:
                continue
            for line, row in table:
                mol = row["molecule"]
                for label, shown in row.items():
                    if label == "molecule":
                        continue
                    audit.cell(f"{path}:{line} {mol}", label, shown, openconf_value(label, by, mol))


def openconf_value(label: str, by: dict, mol: str) -> object:
    def group(arm: str) -> list[dict[str, str]]:
        return by[(mol, OPENCONF_ARM[arm])]

    def to_hit(g: list[dict[str, str]]) -> list[float]:
        return [float(r["calls_to_hit"]) for r in g if r["calls_to_hit"]]

    def col(g: list[dict[str, str]], key: str) -> float | None:
        values = [float(r[key]) for r in g if r[key]]
        return median(values) if values else None

    if m := re.fullmatch(r"hits (\w+)", label):
        return frac(group(m.group(1)))
    if m := re.fullmatch(r"Fisher p(?: vs (\w+))?", label):
        a, b = group("anneal"), group(m.group(1) or "openconf")
        return fisher(hits_of(a), len(a), hits_of(b), len(b))
    if m := re.fullmatch(r"Mann-Whitney p(?: vs (\w+))?", label):
        a, b = to_hit(group("anneal")), to_hit(group(m.group(1) or "openconf"))
        return mann_whitney(a, b, continuity=True, degenerate=None) if a and b else None
    if m := re.fullmatch(r"n_3kcal (\w+)", label):
        return col(group(m.group(1)), "n_3kcal")
    if m := re.fullmatch(r"ΔE_min (\w+)", label):
        return col(group(m.group(1)), "delta_min_kcal")
    if m := re.fullmatch(r"calls to hit (\w+)", label):
        c = to_hit(group(m.group(1)))
        return median(c) if c else None
    if m := re.fullmatch(r"wall (\w+) \(s\)", label):
        return col(group(m.group(1)), "wall_s")
    if m := re.fullmatch(r"(\w+) inside \(s\)", label):
        return col(group(m.group(1)), "pot_s")
    if m := re.fullmatch(r"(\w+) outside \(s\)", label):
        return col(group(m.group(1)), "outside_s")
    raise SystemExit(f"openconf column {label!r} has no rule")


def and_list(items: list[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


SC_FAMILIES = ("126", "108", "96")


def number(text: str) -> float | None:
    return float(text) if text else None


def sc_order(system: str) -> tuple[int, int]:
    family, n = system.removeprefix("sc").split("-")
    return int(n), SC_FAMILIES.index(family)


def sc_ahead(anneal: dict[str, str], gmin: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    """Which code leads on hits, calls and wall; n.s. is p >= 0.05.

    On calls and wall the lead is the code lower in more seed pairs, the
    direction of the Mann-Whitney test.
    """
    a_rows = pick(rows, system=anneal["system"], arm="anneal", harness_revision=anneal["harness_revision"])
    g_rows = pick(rows, system=gmin["system"], arm="gmin", harness_revision=gmin["harness_revision"])

    def lead(column: str, p: str) -> str:
        a = [float(r[column]) for r in a_rows]
        g = [float(r[column]) for r in g_rows]
        balance = sum((y < x) - (y > x) for x in a for y in g)
        if balance == 0:
            return "tie"
        who = "GMIN" if balance > 0 else "anneal"
        return who if float(p) < 0.05 else f"{who}, n.s."

    a_hits, g_hits = int(anneal["hits"]), int(gmin["hits"])
    hits = "tie" if a_hits == g_hits else ("GMIN" if g_hits > a_hits else "anneal")
    if hits != "tie" and float(gmin["fisher_p_vs_gmin"]) >= 0.05:
        hits += ", n.s."
    return {"Hits": hits, "Calls": lead("evals_to_hit", gmin["mannwhitney_p_evals"]),
            "Wall": lead("wall_s", gmin["mannwhitney_p_wall"])}


def sc_reading(anneal: dict[str, dict[str, str]], reruns: dict[str, dict[str, str]],
               rows: list[dict[str, str]]) -> list[str]:
    """The sentences of the sc-n12 README that read the rerun tests."""
    reruns = {s: r for s, r in reruns.items()
              if all(r[c] for c in ("fisher_p_vs_gmin", "mannwhitney_p_evals", "mannwhitney_p_wall"))}
    ahead = {s: sc_ahead(anneal[s], r, rows) for s, r in reruns.items()}
    level = [s for s, r in reruns.items() if int(r["hits"]) >= int(anneal[s]["hits"])]

    def where(column: str, value: str, among: list[str]) -> list[str]:
        return [s for s in among if ahead[s][column] == value]

    out = []
    if ties := where("Hits", "tie", list(reruns)):
        out.append(f"GMIN ties anneal on hits on {and_list(ties)}.")
    for who, other in (("GMIN", "anneal"), ("anneal", "GMIN")):
        sig, ns = where("Hits", who, list(reruns)), where("Hits", f"{who}, n.s.", list(reruns))
        if sig or ns:
            parts = ([f"on {and_list(sig)} at Fisher p < 0.05"] if sig else []) + \
                    ([f"on {and_list(ns)} not significantly" if sig else f"on {and_list(ns)}, not significantly"] if ns else [])
            out.append(f"{who.capitalize() if who == 'anneal' else who} hits more seeds than {other} {', and '.join(parts)}.")
    for column, noun in (("Wall", "wall time"), ("Calls", "calls")):
        if wins := where(column, "GMIN", level):
            out.append(f"GMIN beats anneal on {noun} on {and_list(wins)}, at Mann-Whitney p < 0.05.")
        if leads := where(column, "GMIN, n.s.", level):
            out.append(f"GMIN's lead on {noun} on {and_list(leads)} is not significant.")
        if losses := where(column, "anneal", level):
            out.append(f"Anneal beats GMIN on {noun} on {and_list(losses)}, at Mann-Whitney p < 0.05.")
        if trails := where(column, "anneal, n.s.", level):
            out.append(f"Anneal's lead on {noun} on {and_list(trails)} is not significant.")
    cheap = [s for s in reruns if s not in level
             and float(reruns[s]["wall_median_s"]) < float(anneal[s]["wall_median_s"])
             and float(reruns[s]["evals_median"]) == float(reruns[s]["budget"])]
    if cheap:
        out.append(f"On {and_list(cheap)} GMIN's median wall is lower while it hits fewer seeds, "
                   "and its median calls are the full budget.")
    for s, r in reruns.items():
        if r["hits"] != "0":
            continue
        bests = sorted(float(x["best"]) for x in pick(rows, system=s, arm="gmin", harness_revision=r["harness_revision"]))
        h = r["harness_revision"][:8]
        if bests and f"{bests[0]:.4f}" == f"{bests[-1]:.4f}":
            out.append(f"On {s} every {h} GMIN seed ends at {bests[0]:.4f}.")
        elif bests:
            out.append(f"On {s} the {h} GMIN seeds end between {bests[0]:.4f} and {bests[-1]:.4f}, "
                       f"against {float(r['reference']):.4f}.")
    return out


def summary_against_rows(audit: Audit, summary: str, per_seed: str, arm_key: str, calls_key: str,
                         continuity: bool, mw_column: str = "mannwhitney_p_evals") -> None:
    """A summary file derived from per-seed rows must match them."""
    rows = audit.rows(per_seed)
    for line in audit.rows(summary):
        harness = {"harness_revision": line["harness_revision"]} if "harness_revision" in line else {}
        group = pick(rows, system=line["system"], **{arm_key: line["arm"]}, **harness)
        where = f"{summary} {line['system']} {line['arm']}" + (f" {line['harness_revision'][:8]}" if harness else "")
        if not group:
            audit.record(False, where, "per-seed rows", line["n"] if "n" in line else "?", 0)
            continue
        calls = [float(r[calls_key]) for r in group]
        audit.cell(where, "hits", line["hits"], float(hits_of(group)))
        audit.cell(where, "evals_median", line["evals_median"], median(calls))
        if "evals_iqr" in line:
            audit.cell(where, "evals_iqr", line["evals_iqr"], iqr(calls))
        audit.cell(where, "wall_median_s", line["wall_median_s"], median([float(r["wall_s"]) for r in group]))
        for r in group:
            if r["hit"] in ("1", "true") and float(r[calls_key]) > float(r["charged"]):
                audit.record(False, per_seed, f"{r['system']} {r['seed']} hit after its charged stop", r[calls_key], r["charged"])
        # A GMIN rerun under a later harness is tested against the anneal rows;
        # the anneal row carries the tests against the GMIN rows of its own harness.
        other = pick(rows, system=line["system"], **{arm_key: "gmin" if line["arm"] == "anneal" else "anneal"},
                     **(harness if line["arm"] == "anneal" else {}))
        if line.get(mw_column):
            b = [float(r[calls_key]) for r in other]
            audit.cell(where, mw_column, line[mw_column], mann_whitney(calls, b, continuity))
        if line.get("fisher_p_vs_gmin"):
            audit.cell(where, "fisher_p_vs_gmin", line["fisher_p_vs_gmin"],
                       fisher(hits_of(group), len(group), hits_of(other), len(other)))


def head_to_head(audit: Audit) -> None:
    summary_against_rows(audit, "bench/morse-h2h/results.csv", "bench/morse-h2h/seeds.csv", "code", "evals_to_hit", True)
    summary_against_rows(audit, "bench/metal-cluster-h2h/sc-n12/results.csv", "bench/metal-cluster-h2h/sc-n12/seeds.csv",
                         "arm", "evals_to_hit", True)
    path = "bench/metal-cluster-h2h/sc-n12/README.md"
    sc_rows = audit.rows("bench/metal-cluster-h2h/sc-n12/seeds.csv")
    sc_summary = audit.rows("bench/metal-cluster-h2h/sc-n12/results.csv")
    shown = set()
    for line, row in md_table(audit, path, "System | Arm | Harness"):
        group = [r for r in pick(sc_rows, system=row["System"], arm=row["Arm"])
                 if r["harness_revision"].startswith(row["Harness"])]
        where = f"{path}:{line} {row['System']} {row['Arm']} {row['Harness']}"
        shown.add((row["System"], row["Arm"], row["Harness"]))
        audit.cell(where, "seeds", "12", float(len(group)))
        audit.cell(where, "hits", row["Hits"], frac(group))
        audit.cell(where, "calls median", row["Calls median"], median([float(r["evals_to_hit"]) for r in group]))
        audit.cell(where, "wall median", row["Wall median (s)"], median([float(r["wall_s"]) for r in group]))
    # A GMIN row whose harness is not that of the anneal row of its system is
    # a rerun. Every rerun, the rows it replaces and its tests must be shown,
    # and no system may be left with GMIN rows from the old file only.
    anneal = {r["system"]: r for r in sc_summary if r["arm"] == "anneal"}
    anneal_harness = {s: r["harness_revision"] for s, r in anneal.items()}
    reruns: dict[str, dict[str, str]] = {}
    for r in sc_summary:
        if r["arm"] == "gmin" and r["harness_revision"] != anneal_harness[r["system"]]:
            reruns[r["system"]] = r
    reruns = dict(sorted(reruns.items(), key=lambda item: sc_order(item[0])))
    tested = {}
    for line, row in md_table(audit, path, "System | GMIN harness"):
        where = f"{path}:{line} {row['System']} {row['GMIN harness']}"
        r = reruns.get(row["System"])
        tested[row["System"]] = row["GMIN harness"]
        audit.cell(where, "latest GMIN harness", row["GMIN harness"], r["harness_revision"][:8] if r else "not rerun")
        if r:
            audit.cell(where, "Fisher p", row["Fisher p"], number(r["fisher_p_vs_gmin"]))
            audit.cell(where, "Mann-Whitney p calls", row["Mann-Whitney p calls"], number(r["mannwhitney_p_evals"]))
            audit.cell(where, "Mann-Whitney p wall", row["Mann-Whitney p wall"], number(r["mannwhitney_p_wall"]))
            audit.cell(where, "wall ratio", row["Wall ratio"], number(r["wall_ratio"]))
            if not all(r[c] for c in ("fisher_p_vs_gmin", "mannwhitney_p_evals", "mannwhitney_p_wall")):
                continue
            ahead = sc_ahead(anneal[row["System"]], r, sc_rows)
            for column in ("Hits", "Calls", "Wall"):
                audit.cell(where, f"{column} ahead", row[f"{column} ahead"], ahead[column])
    by_harness: dict[str, list[str]] = {}
    for system, r in reruns.items():
        by_harness.setdefault(r["harness_revision"][:8], []).append(system)
        for arm, harness in (("anneal", anneal_harness[system]), ("gmin", anneal_harness[system]),
                             ("gmin", r["harness_revision"])):
            audit.cell(path, f"table row {system} {arm} {harness[:8]}",
                       "shown" if (system, arm, harness[:8]) in shown else "missing", "shown")
        audit.cell(path, f"test row {system}", tested.get(system, "missing"), r["harness_revision"][:8])
    for system in tested:
        audit.cell(path, f"test row {system} is a rerun", "rerun" if system in reruns else "not rerun", "rerun")
    for system in anneal_harness:
        audit.cell(path, f"{system} GMIN row from the corrected file",
                   "rerun" if system in reruns else f"old file only ({anneal_harness[system][:8]})", "rerun")
    parts = [f"{and_list(systems)} under harness {h}" for h, systems in by_harness.items()]
    audit.said(path, f"The GMIN rows rerun with the corrected file are {'; '.join(parts)}.")
    audit.said("bench/metal-cluster-h2h/sc-8e076b7f/README.md",
               f"`../sc-n12/README.md` has the corrected file and GMIN reruns of {and_list(reruns)}.")
    for sentence in sc_reading(anneal, reruns, sc_rows):
        audit.said(path, sentence)

    path = "bench/morse-h2h/README.md"
    results = audit.rows("bench/morse-h2h/results.csv")
    m38 = pick(results, system="m38r6", arm="anneal")[0]
    audit.said(path, f"{float(m38['evals_median']):.0f}")

    water = audit.rows("bench/water-gmin/rows.csv")
    path = "bench/water-gmin/README.md"
    sizes = {"(H₂O)₈": "tip4p8", "(H₂O)₁₂": "tip4p12", "(H₂O)₁₆": "tip4p16", "(H₂O)₂₀": "tip4p20", "(H₂O)₂₁": "tip4p21"}
    for line, row in md_table(audit, path, "Size | anneal | GMIN"):
        system = sizes[row["Size"]]
        a, g = pick(water, system=system, code="anneal"), pick(water, system=system, code="gmin")
        where = f"{path}:{line} {system}"
        audit.cell(where, "anneal hits", row["anneal"], frac(a))
        audit.cell(where, "GMIN hits", row["GMIN"], frac(g))
        for arm, rows_ in (("anneal", a), ("GMIN", g)):
            shown_all, shown_hit = row[f"{arm} calls (all / hits)"].split(" / ")
            hit_calls = [float(r["evals_to_hit"]) for r in rows_ if r["hit"] == "1"]
            audit.cell(where, f"{arm} calls all", shown_all, median([float(r["evals_to_hit"]) for r in rows_]))
            audit.cell(where, f"{arm} calls hits", shown_hit, median(hit_calls) if hit_calls else None)
        wall_a, wall_g = row["wall a / g"].split(" / ")
        audit.cell(where, "anneal wall", wall_a, median([float(r["wall_s"]) for r in a]))
        audit.cell(where, "GMIN wall", wall_g, median([float(r["wall_s"]) for r in g]))
        pot, out = row["anneal pot / out"].split(" / ")
        audit.cell(where, "anneal pot", pot, median([float(r["pot_s"]) for r in a]))
        audit.cell(where, "anneal out", out, median([float(r["outside_s"]) for r in a]))
        audit.cell(where, "GMIN pot", row["GMIN pot"], median([float(r["pot_s"]) for r in g]))
        audit.cell(where, "p(F)", row["p(F)"], fisher(hits_of(a), len(a), hits_of(g), len(g)))
        audit.cell(where, "p(U), no continuity correction", row["p(U)"],
                   mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in g], continuity=False))
    twelve = pick(water, system="tip4p12", code="anneal")
    audit.said(path, "where the median run stops within 0.25 s.")
    audit.cell(path, "median (H2O)12 anneal wall under 0.25 s", "0.229", median([float(r["wall_s"]) for r in twelve]))
    for system in ("tip4p16", "tip4p21"):
        for code in ("anneal", "gmin"):
            audit.cell(path, f"{system} {code} all-seed calls median is the budget", "2000000",
                       median([float(r["evals_to_hit"]) for r in pick(water, system=system, code=code)]))

    rho6 = audit.rows("bench/morse-h2h/seeds-cdb82168.csv")
    hits = sorted((r["system"], r["seed"]) for r in rho6 if r["hit"] == "1" and int(r["charged"]) < int(r["budget"]))
    audit.cell("bench/morse-h2h/README-cdb82168.md", "hits stopping inside the budget",
               "m55r14 2000 2001 2003 2004 2005; m80r14 2000 2001 2002 2003",
               "; ".join(f"{s} " + " ".join(seed for sys_, seed in hits if sys_ == s) for s in ("m55r14", "m80r14")))


SC96_78_ROWS = "data/sc96_78_hits.csv"
SC96_78_HOWTO = "docs/orgmode/howto/sutton-chen-head-to-head.org"
SC96_78_BUDGET = 2_000_000


def signflip(diffs: list[float], draws: int = 20000, seed: int = 0) -> float:
    """Two-sided paired sign-flip p of docs/scripts/plot_sc_hits.py."""
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        total = sum(v if rng.randrange(2) == 0 else -v for v in diffs)
        if abs(total / len(diffs)) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (draws + 1)


def sc96_78_howto(audit: Audit) -> None:
    """The 9-6 N=78 table scores the GMIN rows of the last harness revision."""
    rows = audit.rows(SC96_78_ROWS)
    gmin = [r for r in rows if r["arm"] == "gmin" and r["harness_revision"]]
    harness = gmin[-1]["harness_revision"]
    arms = {"gmin": pick(gmin, harness_revision=harness), "iso": pick(rows, arm="iso"), "iso-hop": pick(rows, arm="iso-hop")}
    par2 = {arm: {r["seed"]: float(r["hit_calls"]) if r["hit"] == "1" else 2.0 * SC96_78_BUDGET for r in group}
            for arm, group in arms.items()}
    text = audit.text(SC96_78_HOWTO)
    start = text.index("#+begin_example\nlabel hits")
    block = text[start:text.index("#+end_example", start)].splitlines()[2:]
    audit.cell(SC96_78_HOWTO, "table arms", " ".join(line.split()[0] for line in block), " ".join(arms))
    for line in block:
        label, hits, med, mean, p = line.split()
        group = arms[label]
        where = f"{SC96_78_HOWTO} {label}"
        calls = [float(r["hit_calls"]) for r in group if r["hit"] == "1"]
        audit.cell(where, "hits", hits, frac(group))
        audit.cell(where, "median calls to a hit", med, median(calls) if calls else None)
        audit.cell(where, "mean PAR-2", mean, sum(par2[label].values()) / len(group))
        common = sorted(set(par2["gmin"]) & set(par2[label]))
        audit.cell(where, "paired sign-flip p against gmin", p,
                   None if label == "gmin" else signflip([par2["gmin"][s] - par2[label][s] for s in common]))
        audit.said(SC96_78_HOWTO, f"={label}= hits {hits_of(group)} of {len(group)} seeds")
    audit.said(SC96_78_HOWTO, f"Its rows are those of harness {harness[:8]}")
    earlier = [r for r in rows if r["arm"] == "gmin" and r["harness_revision"] != harness]
    audit.said(SC96_78_HOWTO, f"hit {hits_of(earlier)} of {len(earlier)} seeds. They are kept in the file.")


# Numbers of the GMIN rows from the misread data file, as they were cited.
STALE_GMIN = re.compile(r"\b0/32\b|\b0 of 32\b|\bp\s*[<≤]\s*\$?(?:1e-4|10\^\{?-4\}?|0\.0001)|(?<![\d.])0\.0000\b")
RETIRED = ("Only 9-6 at N=38 has been rerun", "a rerun of 9-6 at 38 atoms", "from the old file, and are not rerun",
           "This article reports no force-call count for that optimiser",
           "hits more seeds on 9 of 13 and GMIN hits more on 2")
# Sentences that cite the old rows as old rows; each is derived from the rows elsewhere.
HISTORICAL = {
    (SC96_78_HOWTO, "The earlier GMIN rows, with temperature 0.8, step 0.40 and a container radius of 30, hit 0 of 32 seeds."),
}


def documents() -> list[str]:
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.md", "*.org", "*.tex", "*.rst"],
                            capture_output=True, text=True, check=True).stdout
    return listed.split()


def stale_claims(audit: Audit, paths: list[str] | None = None) -> None:
    """No sentence about GMIN still cites a number of its old data file."""
    seen = set()
    for path in sorted(set(documents() if paths is None else paths)):
        text = audit.text(path)
        for phrase in RETIRED:
            if phrase in collapse(text):
                audit.record(False, path, "retired claim", phrase, "absent")
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            if re.search(r"gmin", sentence, re.I) and STALE_GMIN.search(sentence):
                key = (path, collapse(sentence))
                seen.add(key)
                audit.record(key in HISTORICAL, path, "old GMIN number", collapse(sentence), "derived or absent")
    scanned = set(documents() if paths is None else paths)
    for path, sentence in sorted(h for h in HISTORICAL - seen if h[0] in scanned):
        audit.record(False, path, "historical sentence", sentence, "present")


def gfn2_ranked(rows: list[dict[str, str]]) -> list[float]:
    """Calls to the hit with a miss ranked after every hit, as gfn2_crest.py --summary ranks them."""
    return [float(r["evals_to_hit"]) if hits_of([r]) else math.inf for r in rows]


def gfn2_table(audit: Audit, path: str, rows: list[tuple[int, dict[str, str]]],
               groups: dict[tuple[str, str], list[dict[str, str]]], per_call: str) -> None:
    """``per_call`` "pooled" is summed time over summed calls; "median" is the median over runs that made a call."""
    for line, row in rows:
        group = groups[(row["system"], row["code"])]
        where = f"{path}:{line} {row['system']} {row['code']}"
        calls = [float(r["evals_to_hit"]) for r in group]
        charged = sum(float(r["charged"]) for r in group)

        def time_per_call(column: str) -> float:
            if per_call == "pooled":
                return sum(float(r[column]) for r in group) / charged * 1e3
            return median([float(r[column]) / float(r["charged"]) * 1e3 for r in group if float(r["charged"])])

        audit.cell(where, "hits", row["hits"], frac(group))
        audit.cell(where, "median calls", row["median calls"], median(calls))
        audit.cell(where, "IQR", row["IQR"], iqr(calls))
        audit.cell(where, "best", row["best / eV"], min(float(r["best"]) for r in group))
        audit.cell(where, "median wall", row["median wall / s"], median([float(r["wall_s"]) for r in group]))
        audit.cell(where, "inside / call ms", row["inside / call"].split()[0], time_per_call("pot_s"))
        audit.cell(where, "outside / call ms", row["outside / call"].split()[0], time_per_call("outside_s"))


def covalent_gfn2(audit: Audit) -> None:
    """The GFN2 tables of the covalent README.

    The first table takes anneal's rows from seeds.csv and CREST's from seeds.csv
    and rows.csv. The paper's Si10 table takes anneal's rows of rows.csv at the
    revision its code cell names, with each row false-hits.csv names counted as
    a miss at its budget and given the energy of its geometry.
    """
    path = "bench/covalent-h2h/README.md"
    header = "system | code | hits | median calls | IQR | best / eV"
    seeds = audit.rows("bench/covalent-h2h/gfn2/seeds.csv")
    rows = audit.rows("bench/covalent-h2h/gfn2/rows.csv")
    named = {(r["system"], r["code"], r["code_revision"], r["seed"]): r
             for r in audit.rows("bench/covalent-h2h/gfn2/false-hits.csv")}
    systems = {"Si10": "si10", "C10": "c10"}

    def crest(system: str) -> list[dict[str, str]]:
        return pick(seeds, system=system, code="crest") + pick(rows, system=system, code="crest")

    first = md_table(audit, path, header)
    groups = {}
    for _, row in first:
        system = systems[row["system"]]
        code = {"anneal G": "anneal-general", "CREST": "crest"}[row["code"]]
        groups[(row["system"], row["code"])] = crest(system) if code == "crest" else pick(seeds, system=system, code=code)
    gfn2_table(audit, path, first, groups, "pooled")
    a, c = groups[("Si10", "anneal G")], groups[("Si10", "CREST")]
    audit.cell(path, "GFN2 Si10 Fisher p", "1", fisher(hits_of(a), len(a), hits_of(c), len(c)))
    audit.cell(path, "GFN2 Si10 Mann-Whitney p", "3.7e-5", mann_whitney(gfn2_ranked(a), gfn2_ranked(c), continuity=True))
    audit.said(path, "CREST stops and CREST makes no call. Mann-Whitney p = 3.7e-5 over the twelve seeds of each code, "
                     "a miss ranked after every hit")
    a, c = groups[("C10", "anneal G")], groups[("C10", "CREST")]
    p = mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in c], continuity=True)
    audit.cell(path, "GFN2 C10 Mann-Whitney p", "0.0005", p)
    audit.cell(path, "GFN2 C10 Fisher p", "0.0094", fisher(hits_of(a), len(a), hits_of(c), len(c)))
    audit.said(path, "CREST on 4 (Fisher p = 0.0094, Mann-Whitney p = 0.0005)")

    paper = md_table(audit, path, header, after=first[-1][0] + 1)
    revision = re.search(r"`([0-9a-f]{7,40})`", paper[0][1]["code"]).group(1)
    recorded = [r for r in pick(rows, system="si10", code="anneal-general") if r["code_revision"].startswith(revision)]
    counted = []
    for r in recorded:
        false_hit = named.get((r["system"], r["code"], r["code_revision"], r["seed"]))
        counted.append(r if false_hit is None else dict(r, hit="false", evals_to_hit=r["budget"], best=false_hit["xtb_eV"]))
    groups = {("Si10", paper[0][1]["code"]): counted, ("Si10", "CREST"): crest("si10")}
    gfn2_table(audit, path, paper, groups, "median")
    c = crest("si10")
    audit.cell(path, "GFN2 paper Si10 Fisher p", "1", fisher(hits_of(counted), len(counted), hits_of(c), len(c)))
    audit.cell(path, "GFN2 paper Si10 Mann-Whitney p", "5.3e-4",
               mann_whitney(gfn2_ranked(counted), gfn2_ranked(c), continuity=True))
    audit.said(path, "Mann-Whitney p = 5.3e-4 on calls to the hit, both misses ranked after every hit.")
    calls = [float(r["evals_to_hit"]) for r in recorded]
    audit.cell(path, "GFN2 Si10 recorded hits", "12/12", frac(recorded))
    audit.cell(path, "GFN2 Si10 recorded median calls", "720.5", median(calls))
    audit.cell(path, "GFN2 Si10 recorded IQR", "925", iqr(calls))
    audit.cell(path, "GFN2 Si10 recorded Mann-Whitney p", "3.7e-5",
               mann_whitney(gfn2_ranked(recorded), gfn2_ranked(c), continuity=True))
    audit.said(path, "anneal has 12/12, median 720.5 calls and IQR 925, and CREST has Fisher p = 1 and Mann-Whitney p = 3.7e-5")
    shown = [f"{int(r['evals_to_hit']):,}" for r in sorted(recorded, key=lambda r: int(r["seed"]))]
    audit.said(path, f"Anneal takes {', '.join(shown[:9])}, {shown[9]} (the false hit), {shown[10]} and {shown[11]} "
                     "calls on seeds 2000-2011")
    gfn2_revision_tables(audit, path, header, paper[-1][0] + 1, rows, crest)


GFN2_SYSTEMS = {"Si10": "si10", "C10": "c10", "Si13": "si13", "C20": "c20"}
# (system, anneal revision, Fisher p, Mann-Whitney p, its sample, sentence that prints them). The sample is
# "calls", calls to the hit with a miss ranked after every hit, or "best", the best energies. The sentence
# is filled from the rows by gfn2_sentence_fields.
GFN2_REVISION_TESTS = (
    ("Si10", "ce1e7a46", "1", "3.7e-5", "calls",
     "anneal reaches the minimum on {anneal_hits} of {runs} seeds and CREST on {crest_hits} "
     "(Fisher p = {fisher_p}, Mann-Whitney p = {mw_p}, a miss ranked after every hit)"),
    ("C10", "ce1e7a46", "0.0013", "2.6e-5", "calls",
     "anneal reaches the ring on {anneal_hits} of {runs} seeds and CREST on {crest_hits} "
     "(Fisher p = {fisher_p}, Mann-Whitney p = {mw_p})"),
    ("C20", "5351f272", "1", "0.0004", "best",
     "Anneal's best is lower than CREST's on {below} of the {pairs} pairs of seeds "
     "(Mann-Whitney p = {mw_p} on the best energies)"),
    ("Si13", "c1c02d16", "0.45", "0.005", "calls",
     "anneal reaches the reference on {anneal_share} seeds, in {anneal_calls} calls, and CREST on "
     "{crest_hits_word}, in {crest_calls} (Fisher p = {fisher_p}, Mann-Whitney p = {mw_p}, a miss ranked "
     "after every hit)"),
)
NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


def gfn2_sentence_fields(a: list[dict[str, str]], c: list[dict[str, str]]) -> dict[str, object]:
    """The counts a GFN2_REVISION_TESTS sentence prints, from anneal's rows ``a`` and CREST's ``c``."""
    def span(rows: list[dict[str, str]]) -> str:
        calls = sorted(int(r["evals_to_hit"]) for r in rows if hits_of([r]))
        return f"{calls[0]:,} to {calls[-1]:,}" if calls else "no hit"

    def word(count: int) -> str:
        return NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)

    hits = hits_of(a)
    return {
        "anneal_hits": hits, "runs": len(a), "crest_hits": hits_of(c), "crest_hits_word": word(hits_of(c)),
        "anneal_share": f"all {word(len(a))}" if hits == len(a) else f"{word(hits)} of {word(len(a))}",
        "anneal_calls": span(a), "crest_calls": span(c), "pairs": len(a) * len(c),
        "below": sum(float(x["best"]) < float(y["best"]) for x in a for y in c),
    }


def gfn2_revision_tables(audit: Audit, path: str, header: str, after: int, rows: list[dict[str, str]], crest) -> None:
    """The GFN2 tables after the paper's Si10 one.

    Anneal's rows are those of rows.csv at the revision its code cell names,
    CREST's every CREST row of the system, as in the first table.
    """
    def anneal(system: str, revision: str) -> list[dict[str, str]]:
        return [r for r in pick(rows, system=system, code="anneal-general") if r["code_revision"].startswith(revision)]

    lines = audit.text(path).splitlines()
    tabled = set()
    while any(line.startswith("| " + header) for line in lines[after:]):
        table = md_table(audit, path, header, after=after)
        groups = {}
        for _, row in table:
            system = GFN2_SYSTEMS[row["system"]]
            if row["code"] == "CREST":
                groups[(row["system"], row["code"])] = crest(system)
            else:
                revision = re.fullmatch(r"anneal G, `([0-9a-f]{7,40})`", row["code"]).group(1)
                groups[(row["system"], row["code"])] = anneal(system, revision)
                tabled.add((row["system"], revision))
        gfn2_table(audit, path, table, groups, "pooled")
        after = table[-1][0] + 1
    for system, revision, fisher_p, mw_p, sample, sentence in GFN2_REVISION_TESTS:
        a, c = anneal(GFN2_SYSTEMS[system], revision), crest(GFN2_SYSTEMS[system])
        where = f"{path} {system} {revision}"
        # A table whose header no longer matches is not read at all, so its rows are not checked.
        audit.record((system, revision) in tabled, where, "GFN2 table row", f"anneal G, `{revision}`", "checked")
        audit.cell(where, "GFN2 Fisher p", fisher_p, fisher(hits_of(a), len(a), hits_of(c), len(c)))
        if sample == "best":
            p = mann_whitney([float(r["best"]) for r in a], [float(r["best"]) for r in c], continuity=True)
        else:
            p = mann_whitney(gfn2_ranked(a), gfn2_ranked(c), continuity=True)
        audit.cell(where, "GFN2 Mann-Whitney p", mw_p, p)
        audit.said(path, sentence.format(fisher_p=fisher_p, mw_p=mw_p, **gfn2_sentence_fields(a, c)))
    below = sum(float(x["best"]) < float(y["best"]) for x in anneal("c20", "5351f272") for y in crest("c20"))
    audit.cell(f"{path} C20 5351f272", "anneal below CREST seed pairs", "134", below)


COVALENT_SYSTEMS = {"Si10": "sw10", "Si13": "sw13", "Si20": "sw20", "C10": "tc10", "C20": "tc20"}
COVALENT_CODES = {
    "anneal R": "anneal", "anneal P": "anneal-nopriors", "anneal G": "anneal-general",
    "anneal P v1": "anneal-nopriors@v1", "anneal G v1": "anneal-general@v1",
    "anneal P v2": "anneal-nopriors@v2", "anneal G v2": "anneal-general@v2",
    "anneal R 5e-3": "anneal-sloppy", "anneal P 5e-3": "anneal-nopriors-sloppy", "anneal G 5e-3": "anneal-general-sloppy",
    "GMIN": "gmin", "GMIN 5e-3": "gmin", "GMIN 1e-6": "gmin-tight", "ASE": "ase", "ASE BH": "ase",
}


def covalent_panels(audit: Audit) -> None:
    """The panel tables of bench/covalent-h2h; GMIN and ASE of the After table ran in the before session."""
    path = "bench/covalent-h2h/README.md"
    panels = (("## Before", "before", None), ("## After", "after", "before"),
              ("## Second panel", "v2", None), ("## Third panel", "v3", None))
    lines = audit.text(path).splitlines()
    for heading, folder, fallback in panels:
        start = lines.index(heading) + 1
        seeds = audit.rows(f"bench/covalent-h2h/{folder}/seeds.csv")
        spare = audit.rows(f"bench/covalent-h2h/{fallback}/seeds.csv") if fallback else []

        def group(system: str, label: str) -> list[dict[str, str]]:
            code = COVALENT_CODES[label]
            found = pick(seeds, system=COVALENT_SYSTEMS[system], code=code)
            return found or pick(spare, system=COVALENT_SYSTEMS[system], code=code)

        for line, row in md_table(audit, path, "system | code | hits | median calls", start):
            g = group(row["system"], row["code"])
            where = f"{path}:{line} {folder} {row['system']} {row['code']}"
            calls = [float(r["evals_to_hit"]) for r in g]
            audit.cell(where, "hits", row["hits"], frac(g))
            audit.cell(where, "median calls", row["median calls"], median(calls))
            if "IQR" in row:
                audit.cell(where, "IQR", row["IQR"], iqr(calls))
            audit.cell(where, "best", row["best"], min(float(r["best"]) for r in g))
            audit.cell(where, "pot_s", row["pot_s"], median([float(r["pot_s"]) for r in g]))
            audit.cell(where, "outside_s", row["outside_s"], median([float(r["outside_s"]) for r in g]))
        if folder not in ("v2", "v3"):
            continue
        for line, row in md_table(audit, path, "system | comparison", start):
            left, right = row["comparison"].split(" vs ")
            a, b = group(row["system"], left), group(row["system"], right)
            where = f"{path}:{line} {folder} {row['system']} {row['comparison']}"
            shown_a, shown_b = row["median calls"].split(" vs ")
            audit.cell(where, "median calls left", shown_a, median([float(r["evals_to_hit"]) for r in a]))
            audit.cell(where, "median calls right", shown_b, median([float(r["evals_to_hit"]) for r in b]))
            audit.cell(where, "hits", row["hits"], f"{frac(a)} vs {frac(b)}")
            p = mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in b],
                             continuity=True, degenerate=None)
            audit.cell(where, "Mann-Whitney p", row["Mann-Whitney p"].replace("tied", ""), p)
            audit.cell(where, "Fisher p", row["Fisher p"], fisher(hits_of(a), len(a), hits_of(b), len(b)))


def release(audit: Audit) -> None:
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    audit.said(TEX, f"The newest released tag is v{version}, which is the crate version.")
    tagged = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}"], cwd=ROOT,
                            capture_output=True, text=True).returncode == 0
    audit.record(tagged, TEX, "release tag exists", f"v{version}", "tagged" if tagged else "no tag")


def lj_large(audit: Audit) -> None:
    audit.rows("bench/ase-gofee-h2h/lj-large.csv")


def ase_gofee(audit: Audit) -> None:
    rows = audit.rows("bench/ase-gofee-h2h/results.csv")
    row = pick(rows, system="lj75", arm="agox_gofee", seed="2000")[0]
    audit.cell("bench/ase-gofee-h2h/README.md", "GOFEE LJ75 seed 2000 stop", "budget", row["stop"])
    audit.said("bench/ase-gofee-h2h/README.md",
               f"it spent the full {row['charged']} calls in {float(row['wall_s']):.0f} s, best energy {float(row['best']):.6f}")


def tip4p_panel(audit: Audit) -> None:
    """The TIP4P panel; its per-seed rows were not committed, so p(U) is checked against the summary only."""
    path = "bench/tip4p-h2h/README-h2h.md"
    summary = {r["system"]: r for r in audit.rows("bench/tip4p-h2h/h2h-summary.csv")}
    sizes = {"(H₂O)₈": "tip4p8", "(H₂O)₁₅": "tip4p15", "(H₂O)₂₀": "tip4p20", "(H₂O)₂₁": "tip4p21", "(H₂O)₂₅": "tip4p25"}
    pairs = (("anneal best / median", "anneal_best_kj", "anneal_best_median_kj"),
             ("GMIN best / median", "gmin_best_kj", "gmin_best_median_kj"),
             ("wall (a/g s)", "anneal_wall_median_s", "gmin_wall_median_s"),
             ("pot (a/g s)", "anneal_pot_median_s", "gmin_pot_median_s"))
    for line, row in md_table(audit, path, "Size | anneal | GMIN | anneal best"):
        s = summary[sizes[row["Size"]]]
        where = f"{path}:{line} {s['system']}"
        audit.cell(where, "anneal hits", row["anneal"], s["anneal_hits"])
        audit.cell(where, "GMIN hits", row["GMIN"], s["gmin_hits"])
        for label, left, right in pairs:
            shown_l, shown_r = row[label].split(" / ")
            audit.cell(where, f"{label} left", shown_l, float(s[left]))
            audit.cell(where, f"{label} right", shown_r, float(s[right]))
        ha, na = map(int, s["anneal_hits"].split("/"))
        hg, ng = map(int, s["gmin_hits"].split("/"))
        audit.cell(where, "p(F)", row["p(F)"], fisher(ha, na, hg, ng))
        audit.cell(where, "p(U) (summary)", row["p(U)"], float(s["mwu_p"]))


def bench_prose(audit: Audit) -> None:
    fp = audit.rows("bench/fp-fingerprint/rows.csv")
    requench = {r["system"]: r["requenches"] for r in fp}
    audit.said("bench/fp-fingerprint/README.md",
               "twelve on Au55 and LJ38, ten on LJ75 and two on TIP4P15")
    audit.cell("bench/fp-fingerprint/README.md", "re-quenches au55 lj38 lj75 tip4p15", "12 12 10 2",
               " ".join(requench[s] for s in ("au55", "lj38", "lj75", "tip4p15")))

    growpal = audit.rows("bench/growpal-h2h/seeds.csv")
    spreads = []
    for system in ("lj38", "lj75", "lj98", "lj104"):
        walls = [float(r["wall_s"]) for r in pick(growpal, system=system, arm="growpal")]
        spreads.append(f"{max(walls) - min(walls):.2f}")
    audit.said("bench/growpal-h2h/README.md",
               f"spreads by {spreads[0]} s at LJ38, {spreads[1]} s at LJ75, {spreads[2]} s at LJ98 and {spreads[3]} s at LJ104.")

    gad = audit.rows("bench/gad-published/summary.csv")
    rerun = pick(gad, system="lj38", arm="anneal", provenance="rerun")[0]
    audit.said("bench/pimc-lj38/README.md",
               f"| anneal | {rerun['hits']}/{rerun['n']} | {float(rerun['evals_median']):g} | {rerun['wall_median_s']} | "
               f"{rerun['inside_us']} | {rerun['outside_us']} | {rerun['anneal_head'][:8]} |")

    nh3 = audit.rows("bench/nh3-octamer/ladder.csv")
    stops = {r["seed"]: r["calls"] for r in nh3 if r["budget"] == "16000" and r["calls"] != "16000"}
    audit.said("bench/nh3-octamer/README.md", f"seeds 17000 and 17002 stopped at {stops['17000']} and {stops['17002']} calls")

    icosa = audit.rows("bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv")
    pt = [float(r["best"]) for r in pick(icosa, system="pt6cu17")]
    audit.said("bench/metal-cluster-h2h/ptcu-b3d03061/README.md",
               f"reach −{-min(pt):.8f} for Pt6Cu17 on all six seeds")
    audit.cell("bench/metal-cluster-h2h/ptcu-b3d03061/README.md", "Pt6Cu17 seeds at that energy", "6",
               float(sum(abs(x - min(pt)) < 1e-8 for x in pt)))

    tip = "bench/tip4p-h2h/README.md"
    audit.said(tip, "18.4 µs per GMIN call. The adaptive arm takes 18.3 µs")
    audit.said(tip, "Seed 2004 moves from −623.336 to −623.877 and still misses.")
    audit.said(tip, "Seeds 2002 and 2004, the two one-contact misses, reach −628.371.")
    audit.cell(tip, "cage wall ratio at 15 molecules", "5.237", 40.21225 / 7.6787855)


def provenance(audit: Audit) -> dict[str, str]:
    gaps: dict[str, str] = {}
    revisions: dict[str, set[str]] = defaultdict(set)
    for path, rows in audit.read.items():
        columns = set(rows[0]) if rows else set()
        revs = [c for c in REVISION_COLUMNS if c in columns]
        hosts = [c for c in HOST_COLUMNS if c in columns]
        problems = []
        if not revs and not hosts:
            problems.append("revision and host unrecorded (no revision or host column)")
        elif not revs:
            problems.append("revision unrecorded (no revision column)")
        elif not hosts:
            problems.append("revision recorded; no host column")
        for column in revs + hosts:
            if any(r[column].strip() in ("", "unknown") for r in rows):
                what = "unrecorded" if column in revs else "empty or unknown"
                problems.append(f"{column} {what} on some rows")
        for column in revs:
            for r in rows:
                value = r[column].strip()
                if re.fullmatch(r"[0-9a-f]{7,40}", value) and value not in EXTERNAL_REVISIONS:
                    revisions[value].add(path)
        if problems:
            gaps[path] = "; ".join(problems)
    names = sorted(revisions)
    table = alias_table()
    index = alias_index(names, table)
    aliases = verified_aliases(audit, table, index)
    unpushed = set()
    for name in (n for n in names if not index.exists(n)):
        if any(local.startswith(name) for local in aliases):
            continue
        unpushed.add(name)
        for path in revisions[name]:
            note = f"revision {name[:8]} is not in this repository"
            gaps[path] = f"{gaps[path]}; {note}" if path in gaps else note
    for name in sorted(unpushed | (set(UNPUSHED) & set(names))):
        audit.record(name in unpushed and name in UNPUSHED, ", ".join(sorted(revisions[name])), f"unpushed revision {name[:8]}",
                     UNPUSHED.get(name, "not listed"), "no pushed equivalent" if name in unpushed else "resolved")
    return gaps


def git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, text=True, check=False)


def git_bytes(*args: str, stdin: bytes = b"", env: dict[str, str] | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, input=stdin, capture_output=True, check=False, env=env).stdout


class RevisionIndex:
    """The revision lookups of one audit, from a fixed number of git calls.

    ``cites`` holds, for each text, the commits that ``git log -S<text>`` over
    the tips would list: those with a file pair, as ``git log --raw`` gives
    them, whose two sides hold the text a different number of times. One
    ``git grep -F`` over every blob of those pairs finds the blobs that hold
    any text at all; only those are read and counted. When a configured diff
    textconv driver applies to a path in that history, ``cites`` comes from
    ``git log -S`` itself, once per text, and ``textconv`` names the paths.
    """

    def __init__(self, names: tuple[str, ...], texts: tuple[str, ...], tips: tuple[str, ...], patched: tuple[str, ...]) -> None:
        asked = sorted({n for n in names + tips + patched if n})
        out = git("cat-file", "--batch-check", stdin="".join(f"{n}^{{commit}}\n" for n in asked)).stdout.splitlines()
        self.commit = {n: line.split()[0] for n, line in zip(asked, out) if line.split()[1:2] == ["commit"]}
        self.parents: dict[str, list[str]] = {}
        for line in git("rev-list", "--parents", "--all", *sorted(set(self.commit.values()))).stdout.splitlines():
            commit, *parents = line.split()
            self.parents[commit] = parents
        remote = git("for-each-ref", "--format=%(objectname)", "refs/remotes").stdout.split()
        self.remote = self.closure(r for r in remote if r in self.parents)
        self.patch_ids: dict[str, str] = {}
        shown = sorted({self.commit[p] for p in patched if p in self.commit})
        if shown:
            for line in git("patch-id", "--stable", stdin=git("show", *shown).stdout).stdout.splitlines():
                pid, commit = line.split()
                self.patch_ids[commit] = pid
        self.cites: dict[str, set[str]] = {t: set() for t in texts}
        self.textconv: dict[str, str] = {}
        heads = sorted({self.commit[t] for t in tips if t in self.commit})
        if self.cites and heads:
            self.pickaxe(heads)

    def closure(self, starts) -> set[str]:
        seen: set[str] = set()
        stack = list(starts)
        while stack:
            commit = stack.pop()
            if commit not in seen:
                seen.add(commit)
                stack.extend(self.parents.get(commit, ()))
        return seen

    def textconv_paths(self, heads: list[str]) -> dict[str, str]:
        """Paths in the history of ``heads`` whose diff attribute names a driver with a textconv command.

        git log -S counts a text in a path's textconv output, not in its blob,
        so the blob counts of ``pickaxe`` do not hold for these paths.
        """
        config = git("config", "--null", "--get-regexp", r"^diff\..*\.textconv$").stdout
        drivers = {entry.split("\n", 1)[0][len("diff."):-len(".textconv")] for entry in config.split("\0") if entry}
        if not drivers:
            return {}
        listed = git("log", "--format=", "--name-only", "--no-renames", "-z", *heads).stdout
        paths = sorted({p for p in listed.split("\0") if p})
        out = git("check-attr", "-z", "--stdin", "diff", stdin="".join(f"{p}\0" for p in paths)).stdout.split("\0")
        return {path: value for path, value in zip(out[0::3], out[2::3]) if value in drivers}

    def pickaxe(self, heads: list[str]) -> None:
        self.textconv = self.textconv_paths(heads)
        if self.textconv:
            drivers = ", ".join(sorted(set(self.textconv.values())))
            print(f"paper_audit: diff textconv driver {drivers} applies to {len(self.textconv)} paths; "
                  "running git log -S once per revision", file=sys.stderr)
            for text in self.cites:
                self.cites[text] = set(git("log", "--format=%H", f"-S{text}", *heads).stdout.split())
            return
        pairs: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        commit = ""
        for line in git("log", "--raw", "--no-abbrev", "--format=commit %H", *heads).stdout.splitlines():
            if line.startswith("commit "):
                commit = line[len("commit "):]
            elif line.startswith(":"):
                old_mode, new_mode, old, new = line[1:].split("\t", 1)[0].split()[:4]
                pairs[commit].append((old_mode, old, new_mode, new))
        absent = "0" * 40
        blobs = sorted({b for ps in pairs.values() for m1, b1, m2, b2 in ps for m, b in ((m1, b1), (m2, b2))
                        if b != absent and m != "160000"})
        objects = git("rev-parse", "--path-format=absolute", "--git-path", "objects").stdout.strip()
        with tempfile.TemporaryDirectory() as scratch:
            env = dict(os.environ, GIT_OBJECT_DIRECTORY=scratch, GIT_ALTERNATE_OBJECT_DIRECTORIES=objects)
            listing = "".join(f"100644 blob {b}\t{b}\n" for b in blobs).encode()
            tree = git_bytes("mktree", stdin=listing, env=env).decode().strip()
            patterns = [arg for t in self.cites for arg in ("-e", t)]
            found = git_bytes("grep", "-a", "-l", "-F", *patterns, tree, env=env).decode().split()
        held = sorted(line.split(":", 1)[1] for line in found)
        counts: dict[str, dict[str, int]] = {}
        stream = git_bytes("cat-file", "--batch", stdin="".join(f"{b}\n" for b in held).encode())
        at = 0
        for blob in held:
            end = stream.index(b"\n", at)
            size = int(stream[at:end].split()[2])
            content = stream[end + 1:end + 1 + size]
            at = end + 2 + size
            counts[blob] = {t: content.count(t.encode()) for t in self.cites}
        none = dict.fromkeys(self.cites, 0)

        def held_in(mode: str, blob: str) -> dict[str, int]:
            if blob == absent:
                return none
            if mode == "160000":
                content = f"Subproject commit {blob}\n".encode()
                return {t: content.count(t.encode()) for t in self.cites}
            return counts.get(blob, none)

        for commit, ps in pairs.items():
            for old_mode, old, new_mode, new in ps:
                before, after = held_in(old_mode, old), held_in(new_mode, new)
                for text in self.cites:
                    if before[text] != after[text]:
                        self.cites[text].add(commit)

    def resolve(self, revision: str) -> str | None:
        return self.commit.get(revision) or (revision if revision in self.parents else None)

    def exists(self, revision: str) -> bool:
        return revision in self.commit

    def on_remote(self, revision: str) -> bool:
        """The commit is reachable from a ref fetched from the remote."""
        return self.resolve(revision) in self.remote

    def patch_id(self, revision: str) -> str | None:
        return self.patch_ids.get(self.resolve(revision) or "")

    def parents_of(self, revision: str) -> list[str]:
        return self.parents.get(self.resolve(revision) or "", [])

    def is_ancestor(self, ancestor: str, revision: str) -> bool:
        commit, of = self.resolve(ancestor), self.resolve(revision)
        return bool(commit and of) and commit in self.closure([of])

    def first_cites(self, text: str, revision: str) -> bool:
        """``revision`` is the oldest commit in its history whose diff changes how often ``text`` occurs."""
        citing = self.cites.get(text, set())
        return revision in citing and not citing & self.closure(self.parents.get(revision, ()))


@functools.cache
def revision_index(names: tuple[str, ...], texts: tuple[str, ...], tips: tuple[str, ...], patched: tuple[str, ...]) -> RevisionIndex:
    return RevisionIndex(names, texts, tips, patched)


def alias_table() -> list[dict[str, str]]:
    with (ROOT / ALIASES).open(newline="") as handle:
        return list(csv.DictReader(line for line in handle if not line.startswith("#")))


def alias_index(names: list[str], table: list[dict[str, str]]) -> RevisionIndex:
    """The index for the cited revisions ``names`` and every revision ``table`` names."""
    revisions = ("local_revision", "pushed_revision", "branch_revision")
    return revision_index(tuple(names) + tuple(r[c] for r in table for c in revisions),
                          texts=tuple(r["local_revision"] for r in table), tips=tuple(r["rows_commit"] for r in table),
                          patched=tuple(r[c] for r in table for c in ("pushed_revision", "branch_revision")))


def verified_aliases(audit: Audit, table: list[dict[str, str]], index: RevisionIndex) -> dict[str, str]:
    """Local revisions of ALIASES whose pushed equivalent checks out, mapped to it.

    Each alias is checked: the local revision is not in the repository; the
    pushed revision and the branch revision, when there is one, are, are
    reachable from a remote ref, and have the recorded patch-id; the rows
    commit is the first in its history to cite the local revision and has
    one parent. That parent is the pushed revision, or, when the alias names
    the paths in ``differs_only_in``, descends from it and differs from it
    in those paths alone.
    """
    verified = {}
    for row in table:
        local, pushed = row["local_revision"], row["pushed_revision"]
        where = f"{ALIASES} {local[:8]}"
        checks = [("local revision is not in this repository", not index.exists(local))]
        for column in ("pushed_revision", "branch_revision"):
            revision = row[column]
            if not revision and column == "branch_revision":
                continue
            checks.append((f"{column} {revision[:8]} is in this repository", index.exists(revision)))
            checks.append((f"{column} {revision[:8]} is on the remote", index.on_remote(revision)))
            checks.append((f"{column} {revision[:8]} has patch-id {row['patch_id'][:8]}", index.patch_id(revision) == row["patch_id"]))
        rows_commit = row["rows_commit"]
        parents = index.parents_of(rows_commit)
        allowed = row["differs_only_in"].split()
        if allowed:
            parent = parents[0] if len(parents) == 1 else ""
            descends = bool(parent) and index.is_ancestor(pushed, parent)
            changed = git("diff", "--name-only", pushed, parent, "--").stdout.split() if descends else None
            checks.append((f"rows commit {rows_commit[:8]} sits on a descendant of {pushed[:8]}", descends))
            checks.append((f"{pushed[:8]} differs from the rows commit's parent only in {' '.join(allowed)}",
                           changed is not None and set(changed) <= set(allowed)))
        else:
            checks.append((f"rows commit {rows_commit[:8]} sits on {pushed[:8]}", parents == [pushed]))
        checks.append((f"rows commit {rows_commit[:8]} first cites {local[:8]}", index.first_cites(local, rows_commit)))
        for label, ok in checks:
            audit.record(ok, where, label, "holds", "holds" if ok else "fails")
        if all(ok for _, ok in checks):
            verified[local] = pushed
    return verified


def unauditable() -> list[str]:
    """Why this clone cannot answer the revision checks, if it cannot.

    A shallow clone, or one without the remote branches, would read every
    revision as missing or unpushed; the audit stops instead of reporting that.
    """
    problems = []
    if git("rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        problems.append("the clone is shallow; fetch the full history (actions/checkout with fetch-depth: 0, "
                        "or git fetch --unshallow origin)")
    if not git("for-each-ref", "--count=1", "refs/remotes").stdout.strip():
        problems.append("there are no remote-tracking refs; fetch the branches of origin "
                        "(git fetch origin '+refs/heads/*:refs/remotes/origin/*')")
    wanted = [(row[c], row["branch"] if c == "branch_revision" else "develop")
              for row in alias_table() for c in ("pushed_revision", "branch_revision") if row[c]]
    out = git("cat-file", "--batch-check", stdin="".join(f"{rev}^{{commit}}\n" for rev, _ in wanted)).stdout.splitlines()
    for (rev, branch), line in zip(wanted, out):
        if line.split()[1:2] != ["commit"]:
            problems.append(f"{ALIASES} names {rev[:8]}, which this clone does not have; fetch {branch} from origin")
    return problems


RERUN_FILES = (TEX, SI, "paper/jctc/cover-letter.tex")
COVER = "paper/jctc/cover-letter.tex"
COVER_TOC = "paper/jctc/toc-graphic.tex"
# Sentences of the cover letter that carry a digit and are not abstract sentences.
COVER_OTHER = ("This submission replaces the preprint arXiv:2302.02811, deposited in 2023 under the title "
               "``Unified Software Design Patterns for Simulated Annealing.",
               "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")


def collapsed(text: str) -> str:
    """Whitespace collapsed, with the rerun tags dropped so a sentence compares by its words."""
    return " ".join(re.sub(r"~?\\rerun\{\d\}", "", text).split())


def cover_letter(audit: Audit) -> None:
    """A cover-letter sentence that carries a number is an abstract sentence, word for word."""
    letter = collapsed(audit.text(COVER))
    manuscript = collapsed(audit.text(TEX))
    body = letter.split("Dear Editor,")[1].split("\\vspace{1em} \\noindent Sincerely")[0]
    sentences = re.split(r"(?<=\.)(?:''|\\\\)?\s+(?=[A-Z`\\])", body)
    checked = 0
    for sentence in sentences:
        bare = re.sub(r"GFN2|TIP4P|CREST|GMIN|\\url\{[^}]*\}|Email", "", sentence)
        if not re.search(r"\d", bare) or "Telephone" in sentence:
            continue
        checked += 1
        if sentence in COVER_OTHER:
            audit.record(True, COVER, "the cover-letter sentence is the preprint line", sentence[:60], sentence[:60])
            continue
        found = sentence in manuscript
        audit.record(found, COVER, "a cover-letter sentence with a number is an abstract sentence",
                     sentence[:70], sentence[:70] if found else "not in the manuscript")
    audit.record(checked >= 8, COVER, "the cover letter carries the abstract numbers", ">= 8", str(checked))
    toc = audit.text(COVER_TOC)
    audit.record("paperwidth=3.25in,paperheight=1.75in" in toc, COVER_TOC, "the graphic slot is 3.25 in by 1.75 in",
                 "3.25in 1.75in", "3.25in 1.75in")
    audit.said(COVER, "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")
    for word in ("TODO", "\\todo"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter has no {word}", "none", "none")


def manuscript_new(audit: Audit) -> None:
    """The abstract split, the added panels, the definitions, the priors, and the host paragraph."""
    panel, groups = gmin_panel_groups(audit)
    ase = ase_groups(audit)

    def name(system: str) -> str:
        return panel.LABELS[system][1].replace(", \\(\\rho=", " at \\(\\rho=")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(groups[(system, arm)])} of {len(groups[(system, arm)])}"

    def p_of(system: str) -> float:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return fisher(hits_of(a), len(a), hits_of(g), len(g))

    def wall(system: str, arm: str) -> float:
        return median([float(r["wall_s"]) for r in groups[(system, arm)]])

    systems = list(panel.PAPER_BUDGETS)
    ahead = [s for s in systems if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")])]
    ours = [s for s in ahead if p_of(s) < 0.05]
    rest = [s for s in ahead if s not in ours]
    audit.cell(TEX, "the significant wins of this setting", "lj38 lj75 lj98 m55r14 m80r14", " ".join(ours))
    audit.cell(TEX, "the wins that are not significant", "lj102 lj103 lj104 au55", " ".join(rest))
    lower = [s for s in systems if wall(s, "anneal") < wall(s, "gmin")]
    slower = [s for s in systems if s not in lower]
    audit.cell(TEX, "the two clusters where this setting is slower", "m55r6 m80r6", " ".join(slower))
    audit.said(TEX, f"The median wall of a run is lower for this setting on {len(lower)} of the {len(systems)} "
                    f"clusters and higher on {name('m55r6')} and {name('m80r6')}.")
    audit.said(TEX, f"Its median wall is lower on {len(lower)} of the {len(systems)} clusters of "
                    "Table~\\ref{tab:gmin}, and it is higher where GMIN hits 12 of 12.")
    audit.cell(TEX, "GMIN hits 12 of 12 where it is faster", "12", float(hits_of(groups[("m80r6", "gmin")])))
    s_ = panel.show_s
    audit.said(TEX, f"The median wall of a run is {s_(wall('lj38', 'anneal'))}~s for this setting against "
                    f"{s_(wall('lj38', 'gmin'))}~s for GMIN on LJ\\(_{{38}}\\), and {s_(wall('lj75', 'anneal'))}~s "
                    f"against {s_(wall('lj75', 'gmin'))}~s on LJ\\(_{{75}}\\).")
    audit.said(TEX, f"On {name('m80r6')} it is {s_(wall('m80r6', 'anneal'))}~s against {s_(wall('m80r6', 'gmin'))}~s.")

    lj38 = ase[("lj38", "recommended")]
    cap = ase_cap(audit, ase, "lj38")
    capped = all(r["stop"] == "wall" for a in ASE_ARMS for r in ase[("lj38", a)])
    audit.record(capped, TEX, "every ASE run on LJ38 stops at the cap", "every", "every" if capped else "not every")
    audit.said(TEX, f"On LJ\\(_{{38}}\\) the median wall of this setting is "
                    f"{s_(median([float(r['wall_s']) for r in lj38]))}~s, and every ASE run stops at the cap of "
                    f"{s_(cap)}~s.")

    gofee_dir = "bench/ase-gofee-h2h"
    summary = {(r["system"], r["arm"]): r for r in audit.rows(f"{gofee_dir}/summary.csv")}
    lj = ("lj38", "lj55", "lj75")
    audit.cell(TEX, "this setting hits the three clusters on 6 of 6", "6/6 6/6 6/6",
               " ".join(f"{summary[(s, 'paper')]['hits']}/{summary[(s, 'paper')]['n']}" for s in lj))
    audit.cell(TEX, "ASE basin hopping and GOFEE hit none of the three", "0 0 0 0 0 0",
               " ".join(summary[(s, a)]["hits"] for a in ("ase_bh", "agox_gofee") for s in lj))
    mh = [summary[(s, "ase_mh")]["hits"] for s in lj]
    audit.said(TEX, "ASE basin hopping hits none of them, and GOFEE hits none.")
    audit.cell(TEX, "ASE minima hopping hits none of LJ75", "0", float(mh[2]))
    audit.said(TEX, f"ASE minima hopping hits LJ\\(_{{38}}\\) on {mh[0]} of 6, LJ\\(_{{55}}\\) on {mh[1]} of 6, and "
                    "LJ\\(_{75}\\) on none.")
    audit.said(TEX, "This setting hits all three clusters on 6 of 6 seeds.")
    g = [float(summary[(s, "agox_gofee")]["wall_median_s"]) for s in lj]
    a = [float(summary[(s, "paper")]["wall_median_s"]) for s in lj]
    audit.said(TEX, f"The median wall of a GOFEE run is {s_(g[0])}~s on LJ\\(_{{38}}\\), {s_(g[1])}~s on "
                    f"LJ\\(_{{55}}\\), and {s_(g[2])}~s on LJ\\(_{{75}}\\), against {s_(a[0])}~s, {s_(a[1])}~s, "
                    f"and {s_(a[2])}~s for this setting.")
    audit.said(TEX, f"the median wall of a GOFEE run on LJ\\(_{{38}}\\) is {s_(g[0])}~s against {s_(a[0])}~s.")
    audit.record(all(x > y for x, y in zip(g, a)), TEX, "the GOFEE median wall is higher on all three",
                 "higher", "higher" if all(x > y for x, y in zip(g, a)) else "not")
    audit.said(TEX, "its median wall is higher than that of this setting on all three")
    rows = audit.rows(f"{gofee_dir}/results.csv")
    heads = {r["head"][:8] for r in rows if r["head"] != "unknown"}
    unknown = sum(r["head"] == "unknown" for r in rows)
    audit.said(TEX, f"were collected at {len(heads)} repository heads, which")
    audit.said(TEX, f"with the head unrecorded on {unknown} rows.")
    versions = dict(line.split("=", 1) for line in audit.text(f"{gofee_dir}/versions.txt").split() if "=" in line)
    audit.said(TEX, f"The versions are ASE {versions['ase']} and AGOX {versions['agox']}.")
    audit.cell(TEX, "the GOFEE and minima-hopping wall cap on LJ75 is 2 hours", "2", float(versions["lj75_mh_gofee_wall_cap_s"]) / 3600)
    audit.said(TEX, "ASE minima hopping and GOFEE on LJ\\(_{75}\\) also stop at a wall cap of 2 hours")
    other = [r for r in audit.rows(f"{gofee_dir}/asg2-summary.csv") if r["source"] == "measured"]
    audit.cell(SI, "the other panel systems of the GOFEE table", "au55 cu38 cu12au26 sw10 tc10 tip4p8",
               " ".join(dict.fromkeys(r["system"] for r in other)))

    tip = {r["system"]: r for r in audit.rows("bench/tip4p-h2h/h2h-summary.csv")}
    sizes = ("tip4p8", "tip4p15", "tip4p20")
    audit.said(TEX, "hits \\((\\mathrm{H_2O})_{8}\\), \\((\\mathrm{H_2O})_{15}\\), and\n\\((\\mathrm{H_2O})_{20}\\) on "
                    + series(tip[s]["anneal_hits"].split("/")[0] for s in sizes) + " of 12 seeds and GMIN on "
                    + series(tip[s]["gmin_hits"].split("/")[0] for s in sizes) + "\nof 12.")
    audit.cell(TEX, "TIP4P against GMIN: seeds and budget", "12 2000000",
               " ".join(sorted({tip[s]["n"] for s in tip} | {tip[s]["budget"] for s in tip}, key=len)))
    tp = {s: panel.show_p(float(tip[s]["fisher_p"]), tex=True) for s in tip}
    audit.said(TEX, f"This setting hits \\((\\mathrm{{H_2O}})_{{8}}\\) on 12 of 12 and GMIN on 8 of 12, with\n\\(p={tp['tip4p8']}\\).")
    audit.cell(TEX, "TIP4P15 and TIP4P20 p", f"{tp['tip4p15']} {tp['tip4p20']}",
               "\\(3.4\\times 10^{-4}\\) 0.037")
    audit.said(TEX, "It hits \\((\\mathrm{H_2O})_{15}\\) on 9 of 12 and \\((\\mathrm{H_2O})_{20}\\) on 5 of\n12, and GMIN hits neither, with \\(p\\) of \\(3.4\\times 10^{-4}\\) and 0.037.")
    audit.cell(TEX, "GMIN hits neither of TIP4P15 and TIP4P20", "0/12 0/12",
               f"{tip['tip4p15']['gmin_hits']} {tip['tip4p20']['gmin_hits']}")
    audit.cell(TEX, "TIP4P21 and TIP4P25 hits", "0/12 0/12 1/12 0/12",
               f"{tip['tip4p21']['anneal_hits']} {tip['tip4p21']['gmin_hits']} {tip['tip4p25']['anneal_hits']} {tip['tip4p25']['gmin_hits']}")
    quicker = [s for s in tip if float(tip[s]["anneal_wall_median_s"]) < float(tip[s]["gmin_wall_median_s"])]
    audit.cell(TEX, "the median wall of this setting is lower on all 5 TIP4P sizes", "5", float(len(quicker)))
    more = [s for s in tip if int(tip[s]["anneal_hits"].split("/")[0]) > int(tip[s]["gmin_hits"].split("/")[0])]
    audit.said(TEX, f"Against GMIN on TIP4P, this setting hits more seeds on {len(more)} of the {len(tip)} sizes")
    audit.said(TEX, "against GMIN on TIP4P this setting hits \\((\\mathrm{H_2O})_{15}\\) on 9 of 12\nand GMIN on 0 of 12.")
    audit.said("bench/tip4p-h2h/README-h2h.md",
               "(H₂O)₂₅ is not in the database, so its hit is scored against the lowest energy either arm reached")

    # Constants of the definitions and of the priors, read from the source.
    preset = "src/methods/cluster_hopping/preset.rs"
    for fragment in ("TEMPERATURE: f64 = 0.8;", "BIAS_HEIGHT: f64 = 0.25;", "MERGE_RADIUS: f64 = 0.7;",
                     "NEIGHBOUR_CUTOFF: f64 = 1.6;", "SYMMETRISE_CUTOFF: f64 = 2.5;", "SOAP_RMSD: f64 = 0.35;",
                     "SOAP_CUTOFF: f64 = 3.5;", "SCREEN_MARGIN: f64 = 2.0;"):
        audit.said(preset, fragment)
    config = "src/methods/cluster_hopping/config.rs"
    for fragment in ("bias_gamma: 5.0,", "tabu_capacity: 8,",
                     "ContinuousSymmetry::Measure { interval: 10 }",
                     "prior_soap: true,", "calibrated_screen: true,"):
        audit.said(config, fragment)
    audit.said("src/methods/cluster_hopping.rs", "const WINDOW: usize = 256;")
    audit.said("src/methods/cluster_hopping.rs", "const WARMUP: usize = 8;")
    audit.said("src/methods/cluster_hopping.rs", "const QUANTILE: f64 = 0.9;")
    hop = audit.text("src/methods/cluster_hopping.rs")
    audit.record("if frac_421 < 0.20 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_421 < 0.20 {" not in hop else "present")
    audit.record("if frac_422 < 0.15 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_422 < 0.15 {" not in hop else "present")
    audit.record("if frac_555 < 0.20 {" not in hop, "src/methods/cluster_hopping.rs", "motif-seeded growth gate is removed",
                 "absent", "absent" if "if frac_555 < 0.20 {" not in hop else "present")
    audit.record("let motif_gap = ledger.budget() / 2;" not in hop, "src/methods/cluster_hopping.rs",
                 "motif reseed gap is removed", "absent",
                 "absent" if "let motif_gap = ledger.budget() / 2;" not in hop else "present")
    audit.record("if cfg.long_ranged && !ledger.target_reached() && ledger.remaining() > 0 {" not in hop,
                 "src/methods/cluster_hopping.rs", "long-ranged Mackay-cap branch is removed", "absent",
                 "absent" if "if cfg.long_ranged && !ledger.target_reached() && ledger.remaining() > 0 {" not in hop else "present")
    morse_driver = audit.text("examples/h2h_morse.rs")
    audit.record("cfg.long_ranged = kind.second_neighbour_energy() < -0.05;" not in morse_driver,
                 "examples/h2h_morse.rs", "Morse long-ranged Mackay-cap toggle is removed", "absent",
                 "absent" if "cfg.long_ranged = kind.second_neighbour_energy() < -0.05;" not in morse_driver else "present")
    audit.said("src/bias.rs", "let w = scale * (self.w0 + entropy) * libm::exp(-self.v[i] / denom);")
    audit.said("src/bias.rs", "let denom = (self.gamma - 1.0) * temp;")
    audit.said("src/csm.rs", "Lowest-measure average from symmetry operations detected on the incumbent.")
    audit.said("src/methods/lattice_search.rs", "hollow_cutoff: 1.47 * r,")
    audit.said("src/methods/lattice_search.rs", "clearance: 0.85 * r,")
    audit.said("src/methods/lattice_search.rs", "interior_coordination: 12,")
    for fragment in ("fcc_cut(n_atoms, centre, nearest)", "binary_geometry::mackay_cut(n_atoms, nearest)",
                     "binary_geometry::polyicosahedral_cut(n_atoms, join, nearest)"):
        audit.record(fragment not in hop, "src/methods/cluster_hopping.rs", "template cut proposal is removed", "absent",
                     "absent" if fragment not in hop else "present")
    audit.record("&& cfg.prior_core" not in hop, "src/methods/cluster_hopping.rs", "prior_core branch is removed",
                 "absent", "absent" if "&& cfg.prior_core" not in hop else "present")
    audit.record("&& species.len() >= 2" not in hop, "src/methods/cluster_hopping.rs", "binary motif-gated branch is removed",
                 "absent", "absent" if "&& species.len() >= 2" not in hop else "present")
    geom = audit.text("src/methods/cluster_hopping/binary_geometry.rs")
    audit.record("pub(crate) fn mackay_cap(n: usize, nearest: f64)" not in geom,
                 "src/methods/cluster_hopping/binary_geometry.rs", "mackay_cap helper is removed", "absent",
                 "absent" if "pub(crate) fn mackay_cap(n: usize, nearest: f64)" not in geom else "present")

    # Host and commands.
    receipt = "paper/jctc/logs/cuh2-hardware-receipt.txt"
    audit.said(receipt, "Model name:                              Intel(R) Xeon(R) Processor")
    audit.said(receipt, "CPU family:                              6")
    audit.said(receipt, "Model:                                   207")
    audit.said(receipt, "CPU(s):                                  8")
    audit.said(receipt, "Hypervisor vendor:                       KVM")
    audit.said(receipt, "Ubuntu 13.3.0-6ubuntu2~24.04.1")
    audit.said(receipt, "rustc 1.88.0")
    audit.said(receipt, "2400.000")
    audit.said("bench/gmin-panel/gmin-source.txt", "gfortran=13.3.0")
    audit.said("bench/gmin-panel/gmin-source.txt", "/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.0")
    audit.said("bench/gmin-panel/README.md", "(`taskset -c 0`, one thread, every other process moved\nto cores 1-7)")
    audit.said("paper/jctc/logs/water/HOST-TOPOLOGY", "cpu=Intel Xeon Processor, family 6, model 207")
    text = audit.text(TEX)
    for needle in ("host name\n\\texttt{bench}", "8 cores of an Intel Xeon Processor, family 6, model 207",
                   "2400~MHz under KVM", "Ubuntu 24.04", "Rust 1.88.0 and gfortran 13.3.0",
                   "reference BLAS and LAPACK 3.12.0", "no log records the\nmemory size or the kernel"):
        audit.record(needle in text, TEX, "the host paragraph says", needle.replace("\n", " "), "present")
    commands = {
        "bash scripts/gmin_panel/build_gmin.sh": "scripts/gmin_panel/build_gmin.sh",
        "snakemake -s workflow/Snakefile gmin_panel": "workflow/Snakefile",
        "bash scripts/gmin_panel/enter_bench.sh --summarize --tex": "scripts/gmin_panel/enter_bench.sh",
        "bash scripts/ase_panel/enter_bench.sh --system lj38\n": "scripts/ase_panel/enter_bench.sh",
        "bash scripts/ase_nonlj/enter_bench.sh\n": "scripts/ase_nonlj/enter_bench.sh",
        "python3 paper/jctc/scripts/water_h2h.py": "paper/jctc/scripts/water_h2h.py",
    }
    for command, path in commands.items():
        audit.record(command in text and (ROOT / path).exists(), TEX, "the command names a script that exists",
                     command, path)
    audit.said("workflow/Snakefile", "rule gmin_panel:")
    audit.said("scripts/gmin_panel/enter_bench.sh", "--summarize")
    audit.said("scripts/ase_panel/enter_bench.sh", "lj38|lj55|lj98|lj104|lj150|au55")
    audit.said("scripts/ase_nonlj/enter_bench.sh", "m38r6")
    audit.said("scripts/ase_panel/enter_bench.sh", ': "${ASE_PANEL_REVISION:?')
    audit.said("scripts/ase_panel/enter_bench.sh", "ROWS=${ASE_PANEL_ROWS:-$DEFAULT_ROWS}")
    audit.said("scripts/ase_nonlj/enter_bench.sh", 'echo "keep existing $SYSTEM $ARM seed $SEED"')
    audit.said(TEX, "The ASE rows cannot be searched again from this repository.")
    audit.said(TEX, "\\path{bench/unreachable-revisions.csv} lists each revision with the rows that cite it.")
    audit.said("paper/jctc/scripts/water_h2h.py", 'command = f"gfn2_water_h2h {MOLECULES} {budget} {seed} {outdir.name}"')
    audit.said("paper/jctc/scripts/water_h2h.py", 'paper_dir = LOGS / f"anneal-paper4000-seed{seed}"')
    audit.said("paper/jctc/logs/water/crest-hexamer-seed0.log", "$ path start.xyz -T 1")
    audit.said("paper/jctc/logs/water/VERSIONS", "polish=xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500")
    audit.said(TEX, "crest start.xyz -T 1\nxtb --gfn 2 --opt vtight --acc 0.01 --iterations 500\ngfn2_water_h2h 6 4000 0 anneal-paper4000-seed0")
    audit.said(TEX, "\\texttt{paper/jctc/scripts/ablation\\_table.py} writes Table~\\ref{tab:ablation}")
    audit.record((ROOT / "paper/jctc/scripts/ablation_table.py").exists(), TEX, "ablation_table.py exists", "exists", "exists")

    # Title, affiliation, spelling, and the pending DOI.
    cover = "paper/jctc/cover-letter.tex"
    title = "One basin-hopping setting for Lennard-Jones,\nMorse, gold, and water clusters"
    audit.said(TEX, "\\title{" + title + "}")
    audit.said(SI, "One basin-hopping setting for Lennard-Jones,\nMorse, gold, and water clusters")
    for path in (TEX, SI):
        audit.said(path, "\\affiliation{SURF B.V., Amsterdam}")
        audit.record("SURF BV" not in audit.text(path), path, "the affiliation is spelled one way", "SURF B.V.", "SURF B.V.")
    audit.record("\\todo{" not in audit.text(TEX), TEX, "no \\todo remains in the manuscript", "none", "none")
    audit.said(TEX, "The Zenodo record for this submission is pending, and it has no DOI yet.")
    for path in (TEX, SI, cover):
        found = sorted({m.group(0) for m in BRITISH.finditer(audit.text(path))})
        audit.record(not found, path, "American spelling", "none", " ".join(found) or "none")
    lf = load_script("landfold_panels_tex", "paper/jctc/scripts/landfold_panels.py")
    audit.said(TEX, lf.caption_tex())
    raster = lf.raster_letters()
    audit.said(TEX, f"Panels (a) to ({chr(ord(raster[0]) - 1)}) are drawn as vectors from the landscape tables.")
    audit.said(TEX, f"Panels ({raster[0]}) to ({raster[-1]}) show the sealed raster maps of the campaigns, because "
                    "their landscape tables are not in the repository.")
    audit.record(raster == "".join(chr(c) for c in range(ord(raster[0]), ord(raster[-1]) + 1)), TEX,
                 "the raster panels of the landfold figure are one run of letters", "contiguous", raster)


RERUN_ITEMS = {"5": "GMIN and ASE caps and tuning", "7": "the GFN2 sign and rerun"}
RERUN_TAG = re.compile(r"\\rerun\{([^}]*)\}")



def rerun_tags(text: str) -> list[tuple[int, str]]:
    """The line number and item of each [RERUN: item N] tag in a source text."""
    return [(number, found.group(1)) for number, line in enumerate(text.splitlines(), 1)
            for found in RERUN_TAG.finditer(line) if not line.lstrip().startswith("%")]


# Results that a rerun replaces. Each sentence that holds one of these fragments must end in its tag,
# so a replaced claim cannot sit in the text without the fence that --pre-submission reads.
RERUN_CLAIMS = (
    (TEX, "5", ("Fisher's test on hits is below 0.05 on 6 of the 9 wins", "It is not below 0.05 on the other 3",
                "GMIN hits LJ\\(_{150}\\) on 9 of 12 against 5 of 12, with \\(p=0.21\\)",
                "The median wall of a run is 0.075~s for this setting", "On M\\(_{80}\\) at \\(\\rho=6\\) it is 168.9~s against 10.691~s",
                "neither ASE arm hits more than 1 of 12 on any of them",
                "Fisher's test on hits is below 0.05 on 6 clusters", "The 5 are LJ\\(_{38}\\)",
                "The other 4 wins of this setting are", "is not significant either",
                "9 of 12 against 5 of 12 and 12 of 12 against 2 of 12",
                "hits more seeds on all 5 Lennard-Jones clusters and all 4 Morse clusters, and no arm hits",
                "6 are significant by Fisher's test", "This setting hits more seeds than GMIN on 9 of the 13 clusters",
                "minima hopping hits none", "Fisher's test against each ASE arm is below 0.05 on all 5 Lennard-Jones clusters",
                "Both ASE arms hit 0 of 12 on all 4", "ASE minima hopping hits LJ\\(_{38}\\) on 1 of 6, LJ\\(_{55}\\) on 6 of 6",
                "Wall is the median wall in seconds of all six runs", "Minima hopping reached its cap on every run")),
    (SI, "5", ("The GMIN arm hits 64 of 64", "where GMIN hits LJ\\(_{98}\\) on 0 of 12 seeds",
               "hit LJ\\(_{75}\\) on 0 of 12 seeds")),
    ("paper/jctc/cover-letter.tex", "5", ("this setting hits more seeds on 9 of 13", "Fisher's test on hits is below 0.05 on 6 of the 9 wins",
                                         "GMIN hits LJ\\(_{150}\\) on 9 of 12", "The median wall of a run is 0.075~s",
                                         "On M\\(_{80}\\) at \\(\\rho=6\\) it is 168.9~s")),
)
SENTENCE_END = re.compile(r"[.?!](?![\d])")


def untagged_claims(text: str, fragment: str, item: str) -> tuple[int, int]:
    """The occurrences of a fragment, and how many of them end their sentence in the tag of that item."""
    flat, found, tagged, start = collapse(text), 0, 0, 0
    while (at := flat.find(collapse(fragment), start)) >= 0:
        found += 1
        end = SENTENCE_END.search(flat, at + len(collapse(fragment)))
        tagged += bool(end and flat.startswith(f"~\\rerun{{{item}}}", end.end()))
        start = at + 1
    return found, tagged


def rerun_claims(audit: Audit) -> None:
    """Every replaced result is present and fenced with the tag of its item."""
    for path, item, fragments in RERUN_CLAIMS:
        for fragment in fragments:
            found, tagged = untagged_claims(audit.text(path), fragment, item)
            audit.record(found > 0 and tagged == found, path, f"a result of item {item} ends its sentence in its tag",
                         fragment, f"{found} found, {tagged} tagged")


def rerun_check(audit: Audit, pre_submission: bool = False) -> None:
    """Every tag names an item that waits on a rerun; under --pre-submission a tag is a failure."""
    macro = audit.text("paper/jctc/macros.tex")
    audit.record("\\newcommand{\\rerun}[1]{\\textbf{[RERUN: item #1]}}" in macro, "paper/jctc/macros.tex",
                 "the rerun tag macro is defined", "defined", "defined")
    for path in RERUN_FILES:
        for number, item in rerun_tags(audit.text(path)):
            audit.record(item in RERUN_ITEMS, f"{path}:{number}", "the tag names an item that waits on a rerun",
                         " ".join(RERUN_ITEMS), item)
            if pre_submission:
                audit.record(False, f"{path}:{number}", f"[RERUN: item {item}] remains ({RERUN_ITEMS.get(item, 'unknown')})",
                             "no tag", "tag present")


REVISION_MAP = "bench/unreachable-revisions.csv"
REVISION_MAP_COLUMNS = ("revision", "why", "row_file", "system", "arm", "revision_column", "rows", "seed_min", "seed_max")


def revision_map_rows(audit: Audit) -> list[dict[str, str]]:
    """Each unreachable revision with the row groups of the files read that cite it."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path, rows in audit.read.items():
        for row in rows:
            for column in REVISION_COLUMNS:
                value = row.get(column, "").strip()
                if value in UNPUSHED:
                    key = (value, path, row.get("system", row.get("board", "")), row.get("arm", row.get("code", "")), column)
                    groups[key].append(row)
    out = []
    for (revision, path, system, arm, column), rows in sorted(groups.items()):
        seeds = [int(r["seed"]) for r in rows if r.get("seed", "").strip().lstrip("-").isdigit()]
        out.append({"revision": revision, "why": UNPUSHED[revision], "row_file": path, "system": system, "arm": arm,
                    "revision_column": column, "rows": str(len(rows)),
                    "seed_min": str(min(seeds)) if seeds else "", "seed_max": str(max(seeds)) if seeds else ""})
    return out


def revision_map_text(rows: list[dict[str, str]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, REVISION_MAP_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def au55_superseded(audit: Audit) -> None:
    """Each row of rows.csv that rows-for-cluster.csv replaces is listed, and no other row is."""
    old = audit.rows("bench/ase-au55/rows.csv")
    new = {
        (r["arm"], r["seed"]): r
        for r in audit.rows("bench/ase-au55/rows-for-cluster.csv")
        if r["code_revision"].startswith("fc1af066")
    }
    listed = audit.rows("bench/ase-au55/superseded.csv")
    want = {(r["system"], r["arm"], r["seed"], r["code_revision"], "bench/ase-au55/rows-for-cluster.csv",
             new[(r["arm"], r["seed"])]["code_revision"]) for r in old
            if r["arm"] != "recommended" and not r["code_revision"].startswith("1ce3ab64")}
    got = {(r["system"], r["arm"], r["seed"], r["code_revision"], r["superseded_by_file"], r["superseded_by_revision"])
           for r in listed if r["host"] == "bench"}
    audit.record(got == want, "bench/ase-au55/superseded.csv", "the superseded rows are the replaced ones",
                 f"{len(want)} rows", f"{len(got)} rows")
    audit.said("bench/ase-au55/README.md", "`superseded.csv` lists each of them.")


def revision_map(audit: Audit) -> None:
    """The committed map from each unreachable revision to its panel rows equals the rows."""
    rows = revision_map_rows(audit)
    audit.record((ROOT / REVISION_MAP).read_text() == revision_map_text(rows), REVISION_MAP, "map equals the cited rows",
                 "committed file", f"{len(rows)} row groups")
    audit.record({r["revision"] for r in rows} == set(UNPUSHED), REVISION_MAP, "every unreachable revision is mapped",
                 f"{len(UNPUSHED)} revisions", f"{len({r['revision'] for r in rows})} mapped")


BRITISH = re.compile(r"\b\w*(?:optimis|neighbour|initialis|symmetris|normalis|minimis|colour|centred|licence|"
         r"favour|behaviour|analyse|organis|recognis|characteris|parametris|summaris|catalogue|"
         r"labell|travell|modell|cancell|fuell)\w*\b",
         re.IGNORECASE)


def manuscript_followups(audit: Audit) -> None:
    """The cluster-preset hits, the wall-cap reconciliation, the GMIN settings, and the M55 disclosures."""
    ase = ase_groups(audit)
    _panel, gmin = gmin_panel_groups(audit)
    names = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj98": "LJ\\(_{98}\\)", "lj104": "LJ\\(_{104}\\)",
             "lj150": "LJ\\(_{150}\\)", "au55": "Au\\(_{55}\\)"}
    hits = [hits_of(ase[(system, "anneal")]) for system in ASE_PANEL]
    audit.said(TEX, f"Its hits are {', '.join(map(str, hits[:-1]))}, and {hits[-1]} of 12 on "
                    f"{', '.join(names[s] for s in ASE_PANEL[:-1])}, and {names['au55']}.")
    audit.cell(TEX, "cluster-preset arm rows per system", "12", float(len(ase[("lj38", "anneal")])))

    summary = audit.rows("bench/ase-recommended/summary.csv")
    caps = {r["system"]: float(r["wall_cap_s"]) for r in summary if r["arm"] == "recommended"}
    gofee = {(r["system"], r["arm"]): r for r in audit.rows("bench/ase-gofee-h2h/summary.csv")}
    mh38, mh55 = gofee[("lj38", "ase_mh")], gofee[("lj55", "ase_mh")]
    audit.said(TEX, f"ASE minima hopping on 0 of 12 seeds at LJ\\(_{{38}}\\) and LJ\\(_{{55}}\\), because its runs stop at wall caps "
                    f"of {caps['lj38']:.3f} and {caps['lj55']:.3f}~s.")
    audit.said(TEX, f"They hit LJ\\(_{{38}}\\) on {mh38['hits']} of {mh38['n']} seeds in a median wall of "
                    f"{float(mh38['wall_median_s']):.1f}~s and LJ\\(_{{55}}\\) on {mh55['hits']} of {mh55['n']} in "
                    f"{float(mh55['wall_median_s']):.1f}~s, both far beyond those caps.")
    audit.cell(TEX, "gofee ASE MH LJ38 and LJ55 are uncapped", "0 0", f"{mh38['capped']} {mh55['capped']}")
    audit.cell(TEX, "gofee walls exceed the panel caps", "1", float(float(mh38["wall_median_s"]) > caps["lj38"]
                                                                  and float(mh55["wall_median_s"]) > caps["lj55"]))
    for system in ("lj38", "lj55"):
        audit.cell(TEX, f"tab:ase ASE MH {system} hits", "0", float(hits_of(ase[(system, "ase_mh")])))

    g98, g75 = gmin[("lj98", "gmin")], gmin[("lj75", "gmin")]
    near98 = sum(abs(float(r["best"]) + 543.6429) < 1e-3 for r in g98)
    other98 = [float(r["best"]) for r in g98 if abs(float(r["best"]) + 543.6429) >= 1e-3]
    near75 = sum(abs(float(r["best"]) + 396.2822) < 1e-3 for r in g75)
    audit.cell(TEX, "panel GMIN LJ98 hits and LJ75 hits", "0 0", f"{hits_of(g98)} {hits_of(g75)}")
    audit.cell(TEX, "panel GMIN LJ98 stuck runs", "11", float(near98))
    audit.cell(TEX, "panel GMIN LJ98 other run", "-540.7447", f"{other98[0]:.4f}")
    audit.cell(TEX, "panel GMIN LJ75 stuck runs", "12", float(near75))
    audit.said(TEX, "& GMIN & 0/12~\\rerun{5} & 4000000 & 0 & 119.9")
    audit.said(TEX, "& GMIN & 0/12~\\rerun{5} & 4000000 & 0 & 182.5")
    audit.said(TEX, "Eleven of its 12 LJ\\(_{98}\\) runs end at \\(-543.6429\\) and one at \\(-540.7447\\), and all 12 of its "
                    "LJ\\(_{75}\\) runs end at \\(-396.2822\\).")
    sym = audit.rows("paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv")
    calls = sorted(int(float(r["force_to_hit"])) for r in sym if r["hit"].strip().lower() in ("1", "true", "yes"))
    audit.said(TEX, f"GMIN hit LJ\\(_{{75}}\\) on {len(calls)} of {len(sym)} seeds in a median of "
                    f"{int(median(calls))} force calls.")
    audit.said(TEX, "The committed \\path{paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv} campaign")
    audit.said(TEX, "GMIN hit LJ\\(_{98}\\) on 64 of 64, as the Supporting Information records.")
    audit.said(SI, "and hit on 200 of 200 seeds.")
    audit.cell(SI, "the sym-starts rows hit", "200", float(len(calls)))
    audit.said(TEX, "The panel uses three GMIN inputs.")
    audit.said(TEX, "The Lennard-Jones rows run the Wales-group LJ\\(_{38}\\) introduction file at every Lennard-Jones size,")
    audit.said(TEX, "The Morse rows add GMIN's native \\texttt{MORSE} keyword and \\(\\rho\\) to the same numerical settings.")
    audit.said(TEX, "The Au\\(_{55}\\) row instead uses the Darby gold parameters, the \\(11.829521\\)~\\AA\\ "
                    "nearest-neighbor container, \\(T=0.1\\)~eV, and the LJ step and taboo distances scaled by "
                    "\\(r_0/2^{1/6}\\).")
    audit.said(TEX, "# GMIN: LJ/Morse use TEMPERATURE 1.0 and NEWRESTART 40;")
    audit.said(TEX, "# Au55 Gupta uses TEMPERATURE 0.1, NEWRESTART 40 and scaled RADIUS/STEP/AVOID")
    audit.said(TEX, "with \\(T=1.0\\), \\texttt{NEWRESTART 40}, and an energy difference of 0.003")
    data = "examples/h2h_gmin_lj.data"
    for fragment in ("TEMPERATURE 1.0", "NEWRESTART 40", "EDIFF 0.003"):
        audit.said(data, fragment)
    audit.record("SYMMETRISE" not in audit.text(data), data, "the panel input has no symmetrization keyword", "absent",
                 "absent" if "SYMMETRISE" not in audit.text(data) else "present")
    audit.said("paper/jctc/data/lj75_gmin/provenance.txt", "SYMMETRISE")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_ALIGN=OFF")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_MPI=OFF")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_OPENMP=OFF")
    audit.said(TEX, "GMIN is built without ALIGN, MPI, or OpenMP and linked against the reference BLAS and LAPACK 3.12.0.")

    cap = [int(r["evals_to_hit"]) for r in gmin[("m55r6", "anneal")] if r["hit"] == "1"]
    gm = [int(r["evals_to_hit"]) for r in gmin[("m55r6", "gmin")] if r["hit"] == "1"]
    ar = [int(r["evals_to_hit"]) for r in ase[("m55r6", "recommended")] if r["hit"] == "1"]
    audit.said(TEX, f"At M\\(_{{55}}\\) and \\(\\rho=6\\), this setting hits 12 of 12 and the 12 hits come at "
                    f"{min(cap)} to {max(cap)} evaluations.")
    audit.said(TEX, f"GMIN hits the same cluster in {min(gm)} to {max(gm)} calls.")
    audit.said(TEX, f"its {len(ar)} hits at the same cluster come at {min(ar)} to {max(ar)} calls.")
    audit.said(TEX, "M\\(_{55}\\), \\(\\rho=6\\) & \\(1\\times 10^{6}\\) & this work & 12/12")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap disclosure prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")
    audit.record(not (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists(),
                 "src/methods/cluster_hopping/mackay_cap_tests.rs", "Mackay cap test module is removed",
                 "absent", "present" if (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists() else "absent")
    audit.said("examples/h2h_morse.rs", "let cfg = Config::recommended(spec.n);")
    audit.said("bench/ase-nonlj/README.md", "Anneal `recommended` uses `Config::recommended_with_scales`")
    audit.cell(TEX, "the M55 rho=14 revisions", "5 7", " ".join(str(hits_of(g)) for g in (
        ase[("m55r14", "recommended")], gmin[("m55r14", "anneal")])))
    audit.said(TEX, "The two panels differ in that scaling, and no run separates the two, so we do not assign the difference "
                    "to it.")
    audit.said(TEX, "Recommended is the preset of this article.")
    audit.record("two-shell icosahedron" not in audit.text(TEX), TEX, "cap-disclosure prose is removed",
                 "absent", "present" if "two-shell icosahedron" in audit.text(TEX) else "absent")
    audit.record("largest cap" not in audit.text(TEX), TEX, "cap-size prose is removed",
                 "absent", "present" if "largest cap" in audit.text(TEX) else "absent")
    audit.said(TEX, "Derived adds the budget window and the Bayes screen of")


def run(report: bool = False, write_map: bool = False, pre_submission: bool = False) -> int:
    problems = unauditable()
    if problems:
        for problem in problems:
            print(f"paper_audit: cannot check revisions: {problem}", file=sys.stderr)
        return 2
    audit = Audit()
    for check in (manuscript_ablation, selection_tables, manuscript_retry, manuscript_gmin, manuscript_tip4p, manuscript_hexamer,
                  manuscript_cross_field, supporting_information, release, paper_0120, gad_published, openconf,
                  head_to_head, sc96_78_howto, stale_claims, tip4p_panel, covalent_panels, covalent_gfn2, ase_gofee, lj_large, bench_prose,
                  manuscript_ase, panel_references, generated_tables, supporting_gmin, supporting_numbers, au55_superseded, manuscript_new, manuscript_followups, cover_letter):
        check(audit)
    if write_map:
        (ROOT / REVISION_MAP).write_text(revision_map_text(revision_map_rows(audit)))
    revision_map(audit)
    rerun_claims(audit)
    rerun_check(audit, pre_submission)
    for path, fragment, _why in UNTRACED:
        audit.said(path, fragment)
    gaps = provenance(audit)
    cited = {r[c].strip() for rows in audit.read.values() for r in rows for c in REVISION_COLUMNS if c in r}
    for name in sorted(set(UNPUSHED) - cited):
        audit.record(False, "UNPUSHED", f"unpushed revision {name[:8]}", UNPUSHED[name], "no row cites it")
    for path, why in sorted(KNOWN_GAPS.items()):
        audit.record(any(word in why for word in ("unrecorded", "revision recorded", "is not in this repository")), path,
                     "a known gap names the revision or says it is unrecorded", "revision or unrecorded", why)
    for path in sorted(set(gaps) | set(KNOWN_GAPS)):
        listed = path in KNOWN_GAPS
        audit.record(listed and path in gaps and KNOWN_GAPS[path] == gaps[path], path, "provenance",
                     KNOWN_GAPS.get(path, "not listed"), gaps.get(path, "no gap"))
    if report:
        for status, where, label, shown, value in audit.results:
            print(f"{status:5}  {where}  {label}  shown={shown}  computed={value}")
        for path, fragment, why in UNTRACED:
            print(f"UNTRACED  {path}  {fragment!r}  {why}")
    drift = audit.drifts()
    for status, where, label, shown, value in drift:
        print(f"{status}  {where}  {label}  shown={shown}  computed={value}", file=sys.stderr)
    print(f"{len(audit.results)} checks, {len(drift)} drifted, {len(audit.read)} row files")
    return 1 if drift else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="print every check, not only the drifted ones")
    parser.add_argument("--write-revision-map", action="store_true", help=f"rewrite {REVISION_MAP} from the rows")
    parser.add_argument("--pre-submission", action="store_true",
                        help="fail while any [RERUN: item N] tag remains in the manuscript, the SI or the cover letter")
    args = parser.parse_args(argv)
    return run(args.report, args.write_revision_map, args.pre_submission)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
