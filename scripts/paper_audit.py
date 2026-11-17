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
import functools
import importlib.util
import math
import os
import random
import re
import subprocess
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
}

KNOWN_GAPS = {
    "data/sc96_78_hits.csv": "code_revision empty or unknown on some rows; harness_revision empty or unknown on some "
                             "rows; host empty or unknown on some rows",
    "bench/depth-posterior-ablation/results.csv": "no revision column",
    "results/size-tiered-selection.csv": "no revision or host column",
    "results/lj-funnel-stalls.csv": "no revision or host column",
    "results/cross-field-hits.csv": "no revision or host column",
    "bench/gmin-wall-v4/results.csv": "no revision or host column",
    "bench/tip4p-water-rates/rates-a24ba760.csv": "no revision column",
    "bench/tip4p-water-rates/rates.csv": "no revision column",
    "paper/jctc/logs/water/NUMBERS.tsv": "no revision or host column",
    "paper/jctc/data/hexamer-xtb680-repolish.csv": "no revision column",
    "paper/jctc/data/water_v45b/water-v45b-perseed.csv": "no revision or host column",
    "paper/jctc/data/campaign-scores.csv": "no revision or host column",
    "paper/jctc/data/system_visuals/references.csv": "no revision or host column",
    "bench/gad-published/summary.csv": "no host column",
    "bench/morse-panel/seeds.csv": "no revision or host column",
    "bench/morse-h2h/seeds.csv": "no revision or host column",
    "bench/fp-fingerprint/rows.csv": "no revision or host column",
    "bench/tip4p-h2h/h2h-summary.csv": "no revision or host column",
    "bench/covalent-h2h/before/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/after/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/v2/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/v3/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/gfn2/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/gfn2/false-hits.csv": "no host column",
    "bench/covalent-h2h/gfn2/rows.csv": "revision 5351f272 is not in this repository; revision ce1e7a46 is not in "
                                        "this repository",
    "bench/ase-gofee-h2h/results.csv": "no host column; head empty or unknown on some rows; revision 0f988569 is not in "
                                       "this repository; revision 349523d3 is not in this repository; revision befbc3e8 "
                                       "is not in this repository; revision bf09f7a2 is not in this repository",
    "bench/growpal-h2h/seeds.csv": "no revision or host column",
    "bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv": "no revision or host column",
    "bench/nh3-octamer/ladder.csv": "no revision or host column",
}

UNTRACED = [
    (SI, "-197.7750", "only REPORT.md quotes it; no committed row holds it"),
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
        return shown in ("", "-", "–", "—")
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


def manuscript_gmin(audit: Audit) -> None:
    summary = audit.rows("bench/gmin-h2h/results-05e9ef4d.csv")
    per_seed = audit.rows("bench/gmin-wall-v4/results.csv")
    arm_name = {"this work": "anneal", "GMIN": "gmin"}
    for row in tex_table(audit, TEX, "tab:gmin"):
        size = CLUSTER[row[0]]
        arm = arm_name[row[1]]
        line = pick(summary, system=f"lj{size}", arm=arm)[0]
        where = f"{TEX} tab:gmin lj{size} {arm}"
        audit.cell(where, "hits", row[2], line["hits_over_n"])
        audit.cell(where, "median", row[3], float(line["evals_median"]))
        audit.cell(where, "IQR", row[4], float(line["evals_iqr"]))
        if arm == "gmin":
            seeds = pick(per_seed, system=f"lj{size}", arm="gmin")
            calls = [float(r["evals_to_hit"]) for r in seeds]
            audit.cell(where + " from gmin-wall-v4", "hits", row[2], frac(seeds))
            audit.cell(where + " from gmin-wall-v4", "median", row[3], median(calls))
            audit.cell(where + " from gmin-wall-v4", "IQR", row[4], iqr(calls))
    h98a = pick(summary, system="lj98", arm="anneal")[0]["hits"]
    h98g = pick(summary, system="lj98", arm="gmin")[0]["hits"]
    audit.said(TEX, f"On LJ\\(_{{98}}\\) this setting hits {h98a} of 6 and GMIN hits {h98g} of 6.")
    audit.said(TEX, f"the comparison with GMIN is {h98a} of 6 against {h98g} of 6 on LJ\\(_{{98}}\\)")


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
    audit.said(TEX, f"The retry rows also hit on {frac(retry98).replace('/', ' of ')}, and their median is {median([float(r['calls_to_hit']) for r in retry98]):.0f}.")
    lj104 = sorted(pick(rows, system="lj", n="104", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    hit104 = [r for r in lj104 if r["hit"] == "1"]
    audit.said(TEX, f"LJ\\(_{{104}}\\) hit on {len(hit104)} of {len(lj104)}, seeds {hit104[0]['seed']} and {hit104[1]['seed']}, at {hit104[0]['calls_to_first_hit']} and {hit104[1]['calls_to_first_hit']} calls.")
    retry104 = sorted((r["seed"] for r in pick(stalls, phase="retry", n="104") if r["hit"] == "1"), key=int)
    audit.said(TEX, f"The retry rows hit on {len(retry104)} of 6, seeds {retry104[0]}, {retry104[1]}, and {retry104[2]}.")
    only_arm = sorted({r["seed"] for r in hit104} - set(retry104))
    only_retry = sorted(set(retry104) - {r["seed"] for r in hit104})
    audit.cell(TEX, "the hit the retry rows miss", "6401", " ".join(only_arm))
    audit.cell(TEX, "hits the retry rows record and this arm misses", "6402 6404", " ".join(only_retry))
    audit.said(TEX, f"It hit {len(hit104)} of 6 and was not kept.")
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
    audit.said(SI, f"The (H\\(_2\\)O)\\(_6\\) plate prints \\({float(energy):.4f}\\)~kJ/mol.")


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
RETIRED = ("Only 9-6 at N=38 has been rerun", "a rerun of 9-6 at 38 atoms", "from the old file, and are not rerun")
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
# "calls", calls to the hit with a miss ranked after every hit, or "best", the best energies.
GFN2_REVISION_TESTS = (
    ("Si10", "ce1e7a46", "1", "3.7e-5", "calls",
     "CREST on 11 (Fisher p = 1, Mann-Whitney p = 3.7e-5, a miss ranked after every hit)"),
    ("C10", "ce1e7a46", "0.0013", "2.6e-5", "calls", "CREST on 4 (Fisher p = 0.0013, Mann-Whitney p = 2.6e-5)"),
    ("C20", "5351f272", "1", "0.0004", "best",
     "CREST's on 134 of the 144 pairs of seeds (Mann-Whitney p = 0.0004 on the best energies)"),
    ("Si13", "c1c02d16", "0.45", "0.005", "calls",
     "CREST on four, in 95,213 to 143,630 (Fisher p = 0.45, Mann-Whitney p = 0.005, a miss ranked after every hit)"),
)


def gfn2_revision_tables(audit: Audit, path: str, header: str, after: int, rows: list[dict[str, str]], crest) -> None:
    """The GFN2 tables after the paper's Si10 one.

    Anneal's rows are those of rows.csv at the revision its code cell names,
    CREST's every CREST row of the system, as in the first table.
    """
    def anneal(system: str, revision: str) -> list[dict[str, str]]:
        return [r for r in pick(rows, system=system, code="anneal-general") if r["code_revision"].startswith(revision)]

    lines = audit.text(path).splitlines()
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
        gfn2_table(audit, path, table, groups, "pooled")
        after = table[-1][0] + 1
    for system, revision, fisher_p, mw_p, sample, sentence in GFN2_REVISION_TESTS:
        a, c = anneal(GFN2_SYSTEMS[system], revision), crest(GFN2_SYSTEMS[system])
        where = f"{path} {system} {revision}"
        audit.cell(where, "GFN2 Fisher p", fisher_p, fisher(hits_of(a), len(a), hits_of(c), len(c)))
        if sample == "best":
            p = mann_whitney([float(r["best"]) for r in a], [float(r["best"]) for r in c], continuity=True)
        else:
            p = mann_whitney(gfn2_ranked(a), gfn2_ranked(c), continuity=True)
        audit.cell(where, "GFN2 Mann-Whitney p", mw_p, p)
        audit.said(path, sentence)
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
            problems.append("no revision or host column")
        elif not revs:
            problems.append("no revision column")
        elif not hosts:
            problems.append("no host column")
        for column in revs + hosts:
            if any(r[column].strip() in ("", "unknown") for r in rows):
                problems.append(f"{column} empty or unknown on some rows")
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


def run(report: bool = False) -> int:
    problems = unauditable()
    if problems:
        for problem in problems:
            print(f"paper_audit: cannot check revisions: {problem}", file=sys.stderr)
        return 2
    audit = Audit()
    for check in (manuscript_ablation, selection_tables, manuscript_retry, manuscript_gmin, manuscript_tip4p, manuscript_hexamer,
                  manuscript_cross_field, supporting_information, release, paper_0120, gad_published, openconf,
                  head_to_head, sc96_78_howto, stale_claims, tip4p_panel, covalent_panels, covalent_gfn2, ase_gofee, lj_large, bench_prose):
        check(audit)
    for path, fragment, _why in UNTRACED:
        audit.said(path, fragment)
    gaps = provenance(audit)
    cited = {r[c].strip() for rows in audit.read.values() for r in rows for c in REVISION_COLUMNS if c in r}
    for name in sorted(set(UNPUSHED) - cited):
        audit.record(False, "UNPUSHED", f"unpushed revision {name[:8]}", UNPUSHED[name], "no row cites it")
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
    return run(parser.parse_args(argv).report)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
