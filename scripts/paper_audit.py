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
import statistics
import subprocess
import tarfile
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

TEX = "paper/jctc/manuscript.tex"
SI = "paper/jctc/si.tex"

REVISION_COLUMNS = ("code_revision", "anneal_head", "release_revision", "head", "harness_revision", "revision",
                    "anneal_sha")
HOST_COLUMNS = ("host", "machine_id", "machine_id_sha256")
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = {"aac1548a8aec13b53ad12a8d6904625081685d2e": "GMIN source/SVNREV",
                      "9cbec1a446352bc9ef064bd2b1a5767194655919": "GMIN source/SVNREV of the eeec8e2c rerun"}
# Local revisions that rows cite, each with the pushed commit it was rebuilt as.
ALIASES = "bench/revision-aliases.csv"
# Remote branches the alias table cites, plus develop. A missing one means
# the clone cannot answer the revision checks, so the audit stops.
AUDIT_REMOTE_BRANCHES = (
    "develop",
    "bench/crest-gaps",
    "bench/gmin-sc3",
    "fix/gmin-sc",
    "fix/gmin-sc2",
)
# Tips that still carry row revisions and are not named by the alias table.
# They join the remote closure when the ref is present. A missing one does
# not make the clone unauditable: the branch may have been deleted after merge.
AUDIT_OPTIONAL_REMOTES = (
    "record/repro-pin",
    "bench/cov-floor2",
    "fix/sc-shelf-waypoint",
    # Pushed record tips that are not in develop. f009acc9 is on the first,
    # 0cc3814c on the second. A missing ref does not stop the audit.
    "record/rerun-tuned-gmin-r2",
    "record/g3-funnel",
)
# Cited revisions with no pushed equivalent. de22f437 records the ASE and GOFEE
# panel from five campaign heads and sits on the last, e09e0487, alone. The
# pushed history before it has one commit, 57735de1, that adds the two-hour
# cap, yet three heads ran capped, so the heads are not its commits one to one.
UNPUSHED = {
    "d12f8825f83bfceb507376555f1c7550da027aa6": "the plateau-restart arm of record/lj150d-restart, run against develop "
                                                "7192e9f8, with no pushed commit of its own",
    "3e2b19b70dced1e50330b13cc8ae96c72021f4e3": "an earlier plateau-restart arm of record/lj150d-restart; no table "
                                                "prints its rows",
    "423d01a571bcf6069cd5247324c197e60b4dcb4f": "an earlier plateau-restart arm of record/lj150d-restart; no table "
                                                "prints its rows",
    "e588305571ae6ee41895fd68af9328379888a1ff": "the pair-scaled arm of record/gmin-losses, run against 628773d6 with "
                                                "the templates patched out, with no pushed commit of its own",
    "0f988569b49afe162e2f5ffd71b2ab45d0b3afec": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "349523d3f5f42b748f3af1573166fd17155ac461": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "befbc3e8": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own; the row gives eight digits",
    "bf09f7a2b7a6bd6359898db675eeafefcd20548b": "an ASE/GOFEE campaign head before e09e0487, with no pushed commit of its own",
    "ce1e7a46826cf51784ed26587c14b2747f5f48e1": "the GFN2 call-floor fix on f2a628c0, a local build head of the GFN2 "
                                                "Si10 and C10 floor rows; the fix was pushed on another base",
    "5351f272370dc8c0d18b06149753effc99aeabd5": "eae26977 with the call floor and the C20 example, a local build head of "
                                                "the GFN2 C20 rows; b4ddad0d on bench/covalent-cov6b has its tree",
    "1ce3ab64493196afd25e8690648e21157d91b882": "the template-free landing head for the GMIN panel and the Au55 ASE "
                                                "reruns, first cited by 93e17b69; no pushed branch holds it",
}

KNOWN_GAPS = {
    "paper/jctc/data/knockouts/restart/rows.csv": "revision 3e2b19b7 is not in this repository; revision 423d01a5 is not "
                                                  "in this repository; revision d12f8825 is not in this repository",
    "paper/jctc/data/knockouts/pair-scale/rows.csv": "revision e5883055 is not in this repository",
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
    "bench/growpal-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv": "revision and host unrecorded (no revision or host column)",
    "bench/nh3-octamer/ladder.csv": "revision and host unrecorded (no revision or host column)",
    "bench/gmin-panel/rows.csv": "revision 1ce3ab64 is not in this repository",
    "bench/gmin-panel/summary.csv": "revision 1ce3ab64 is not in this repository",
    "bench/morse-panel/template-free-rows.csv": "revision 1ce3ab64 is not in this repository",
    "paper/jctc/data/lj55_lj98/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/literature.tsv": "revision and host unrecorded (no revision or host column)",
    "paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv": "revision and host unrecorded (no revision or host "
                                                                "column)",
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


def wilson_interval(hits: int, n: int) -> tuple[float, float]:
    z = statistics.NormalDist().inv_cdf(0.975)
    p = hits / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    centre = p + z * z / (2 * n)
    scale = 1 + z * z / n
    return (centre - half) / scale, (centre + half) / scale


def newcombe_interval(h1: int, n1: int, h2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid score interval for p1 - p2 of independent samples.

    Method 10 of Newcombe, Stat. Med. 17, 873 (1998); his worked example,
    56/70 against 48/80, gives 0.0524 to 0.3339."""
    p1, p2 = h1 / n1, h2 / n2
    l1, u1 = wilson_interval(h1, n1)
    l2, u2 = wilson_interval(h2, n2)
    d = p1 - p2
    return d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def mcnemar(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p: twice the smaller binomial(m, 1/2) tail of the discordant seeds, capped at 1."""
    m = only_a + only_b
    if m == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(m, k) for k in range(min(only_a, only_b) + 1)) / 2.0**m)


def newcombe_paired_interval(both: int, only_a: int, only_b: int, neither: int) -> tuple[float, float]:
    """Method 10 of Newcombe, Stat. Med. 17, 2635 (1998), for the paired difference p_a - p_b.

    His worked example, 36, 12, 2, 0, gives 0.0569 to 0.3404."""
    n = both + only_a + only_b + neither
    pa, pb = (both + only_a) / n, (both + only_b) / n
    la, ua = wilson_interval(both + only_a, n)
    lb, ub = wilson_interval(both + only_b, n)
    product = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    a_ = both * neither - only_a * only_b
    if a_ > 0:
        a_ = max(a_ - n / 2.0, 0.0)
    phi = a_ / math.sqrt(product) if product else 0.0
    d = pa - pb
    lower = d - math.sqrt(max((pa - la) ** 2 - 2 * phi * (pa - la) * (ub - pb) + (ub - pb) ** 2, 0.0))
    upper = d + math.sqrt(max((ua - pa) ** 2 - 2 * phi * (ua - pa) * (pb - lb) + (pb - lb) ** 2, 0.0))
    return lower, upper


def wilcoxon_signed_rank(differences: list[float]) -> tuple[int, float, float]:
    """Nonzero pairs, T+, and the two-sided exact p over all sign assignments, by counting sums of doubled ranks."""
    nonzero = sorted((abs(d), d > 0) for d in differences if d != 0)
    m = len(nonzero)
    if m == 0:
        return 0, 0.0, 1.0
    doubled: list[int] = []
    i = 0
    while i < m:
        j = i
        while j + 1 < m and nonzero[j + 1][0] == nonzero[i][0]:
            j += 1
        doubled += [i + j + 2] * (j - i + 1)
        i = j + 1
    t_plus = sum(r for r, (_, positive) in zip(doubled, nonzero) if positive) / 2.0
    counts = {0: 1}
    for r in doubled:
        nxt = dict(counts)
        for total, ways in counts.items():
            nxt[total + r] = nxt.get(total + r, 0) + ways
        counts = nxt
    center = m * (m + 1) / 4.0
    seen = abs(t_plus - center)
    far = sum(ways for total, ways in counts.items() if abs(total / 2.0 - center) >= seen - 1e-9)
    return m, t_plus, min(1.0, far / 2.0**m)


def paired_table(first: dict[str, bool], second: dict[str, bool]) -> tuple[int, int, int, int]:
    """Both, first only, second only, neither, over the seeds both arms ran; the seed sets must match."""
    if set(first) != set(second):
        raise SystemExit("paired arms do not share their seeds")
    keys = sorted(first)
    both = sum(first[k] and second[k] for k in keys)
    only_a = sum(first[k] and not second[k] for k in keys)
    only_b = sum(second[k] and not first[k] for k in keys)
    return both, only_a, only_b, len(keys) - both - only_a - only_b


def pp(p: float) -> str:
    """A p in prose, inside math: exact when four decimals hold it, else two significant figures."""
    if p >= 1.0:
        return "1"
    four = f"{p:.4f}".rstrip("0")
    if p >= 0.01 and abs(float(four) - p) < 1e-12:
        return four
    if p >= 0.01:
        return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


def interval_text(lo: float, hi: float) -> str:
    def one(x: float) -> str:
        x = round(x, 2) + 0.0
        return f"\\(-\\){-x:.2f}" if x < 0 else f"{x:.2f}"
    return f"{one(lo)} to {one(hi)}"


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
    """tab:ablation pairs the posterior and the uniform draw by seed, as both arms start from the seed's geometry."""
    first = audit.rows("bench/depth-posterior-ablation/results.csv")
    second = audit.rows("results/size-tiered-selection.csv")
    budgets = {38: 400000, 55: 1000000, 75: 4000000, 98: 4000000, 104: 4000000}
    tables = {}
    table = tex_table(audit, TEX, "tab:ablation")
    pooled_row = [c.strip() for c in table[-1]]
    pooled_row[0] = pooled_row[0].replace("\\midrule", "").strip()
    audit.cell(f"{TEX} tab:ablation", "last row is the pooled row", pooled_row[0], "Pooled")
    for row in table[:-1]:
        size = CLUSTER[row[0]]
        arms = {}
        for arm in ("posterior", "uniform"):
            group = {f"a{r['seed']}": r for r in pick(first, system=f"lj{size}", arm=arm)}
            if size >= 75:
                group.update({f"b{r['seed']}": r for r in pick(second, n=str(size), arm=arm)})
            arms[arm] = group
            for r in group.values():
                audit.cell(TEX, f"ablation lj{size} {arm} seed {r['seed']} budget", str(budgets[size]), float(r.get("budget") or budgets[size]))
        hit = {arm: {k: r["hit"] == "1" for k, r in arms[arm].items()} for arm in arms}
        t = paired_table(hit["posterior"], hit["uniform"])
        tables[size] = t
        where = f"{TEX} tab:ablation lj{size}"
        n = sum(t)
        audit.cell(where, "budget", row[1], budgets[size])
        audit.cell(where, "posterior", row[2], f"{t[0] + t[1]}/{n}")
        audit.cell(where, "uniform", row[3], f"{t[0] + t[2]}/{n}")
        for label, cell, value in zip(("both", "posterior only", "uniform only", "neither"), row[4:8], t):
            audit.cell(where, label, cell, float(value))
        audit.cell(where, "exact McNemar p", row[8], mcnemar(t[1], t[2]))
        audit.cell(where, "paired Newcombe 95% interval", row[9], interval_text(*newcombe_paired_interval(*t)))
    audit.cell(f"{TEX} tab:ablation", "sizes in the table", str(len(tables)), 5)
    pooled = tuple(sum(t[i] for t in tables.values()) for i in range(4))
    lo, hi = newcombe_paired_interval(*pooled)
    p_pooled = mcnemar(pooled[1], pooled[2])
    n = sum(pooled)
    where = f"{TEX} tab:ablation pooled"
    audit.cell(where, "posterior", pooled_row[2], f"{pooled[0] + pooled[1]}/{n}")
    audit.cell(where, "uniform", pooled_row[3], f"{pooled[0] + pooled[2]}/{n}")
    for label, cell, value in zip(("both", "posterior only", "uniform only", "neither"), pooled_row[4:8], pooled):
        audit.cell(where, label, cell, float(value))
    audit.cell(where, "exact McNemar p", pooled_row[8], p_pooled)
    audit.cell(where, "paired Newcombe 95% interval", pooled_row[9], interval_text(lo, hi))
    audit.cell(TEX, "ablation: the reviewed pooled outcome table", "33 4 6 5", " ".join(map(str, pooled)))
    widths = [newcombe_paired_interval(*t) for t in tables.values()]
    narrowest = min(round(b - a, 2) for a, b in widths)
    audit.cell(TEX, "ablation: narrowest per-size interval width", "0.52", narrowest)
    audit.said(TEX, f"spans {narrowest:.2f} or more at every size.")
    audit.said(TEX, f"Pooled over the five sizes the posterior hits {pooled[0] + pooled[1]} of {n} and the uniform draw "
                    f"{pooled[0] + pooled[2]} of {n}, the posterior alone hits {pooled[1]} seeds and the uniform draw "
                    f"alone {pooled[2]}, and the exact McNemar \\(p\\) is {pp(p_pooled)}, with an interval from "
                    f"\\({lo:.2f}\\) to {hi:.2f}.")
    audit.cell(TEX, "ablation: the pooled interval", "-0.18 0.10", f"{lo:.2f} {hi:.2f}")
    audit.said(TEX, "No equivalence margin was set before the runs, so the table supports no claim that the two "
                    "draws are equivalent.")
    audit.said(TEX, "Both arms of a seed start from the same geometry.")
    audit.said(TEX, "Table~\\ref{tab:ablation} pairs the two arms by seed.")
    audit.said(TEX, "a Thompson choice among the moves change no cluster at \\(p<0.05\\).")
    for fragment in ("let start = start_of(&cfg, spec.n, seed);", "let mut rng = StdRng::seed_from_u64(seed);",
                     "cfg.arm_selection = ArmSelection::Uniform;"):
        audit.said("examples/depth_ablation.rs", fragment)
    t75, t98, t104 = tables[75], tables[98], tables[104]
    audit.said(TEX, f"the posterior hits LJ\\(_{{75}}\\) on {t75[0] + t75[1]} of {sum(t75)} and the uniform draw hits "
                    f"{t75[0] + t75[2]} of {sum(t75)}.")
    audit.cell(TEX, "ablation: each arm alone hits LJ98 on the same number of seeds", "1", float(t98[1] == t98[2]))
    audit.said(TEX, f"Each arm hits LJ\\(_{{98}}\\) on {t98[0] + t98[1]} of {sum(t98)}: both hit {t98[0]} seeds, "
                    f"each arm alone hits {t98[1]}, and neither hits {t98[3]}.")
    audit.said(TEX, f"The posterior hits LJ\\(_{{104}}\\) on {t104[0] + t104[1]} of {sum(t104)} and the uniform draw "
                    f"hits {t104[0] + t104[2]} of {sum(t104)}.")
    for size in (38, 55):
        t = tables[size]
        audit.cell(TEX, f"ablation: both arms hit lj{size} on 6 of 6", "6 0 0 0", " ".join(map(str, t)))
    audit.said(TEX, "On the first list both arms hit LJ\\(_{38}\\) and LJ\\(_{55}\\) on 6 of 6 seeds.")
    ps = {mcnemar(t[1], t[2]) for t in tables.values()}
    audit.cell(TEX, "ablation: the McNemar probability is 1 at every size", "1", ps.pop() if len(ps) == 1 else ps)
    audit.said(TEX, "The two-sided exact McNemar probability is 1 at each size in")
    tex = audit.text(TEX)
    ablation = tex.split("\\label{sec:ablation}", 1)[1].split("\\subsection{", 1)[0]
    for word in ("Fisher", "independent"):
        audit.record(word not in ablation, TEX, f"the move-selection section does not use {word} inference",
                     "absent", "absent" if word not in ablation else "present")


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
    audit.said(SI, f"LJ\\(_{{98}}\\) hits on {frac(retry98).replace('/', ' of ')} inside \\(4\\times 10^{{6}}\\) calls.")
    audit.said(SI, f"the eighth-leave retry hits {frac(after98).replace('/', ' of ')}.")
    miss = [r for r in after98 if r["hit"] == "0"]
    audit.cell(SI, "retry: the hold-only miss is seed 6404", "6404", miss[0]["seed"] if len(miss) == 1 else None)
    audit.said(SI, f"Seed 6404 stays at a gap of \\({miss[0]['gap']}\\) above \\(-543.665361\\).")
    for size in ("38", "55", "75"):
        audit.cell(SI, f"retry off below 98: lj{size} on 6400-6405", "6/6", frac(pick(rows, phase="after", n=size)))
    retry104 = pick(rows, phase="retry", n="104")
    audit.said(SI, f"The retry rows stored for that leave hit {hits_of(retry104)} of {len(retry104)} on LJ\\(_{{104}}\\)")
    audit.said(SI, f"{frac(retry98).replace('/', ' of ')} on LJ\\(_{{98}}\\).")


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
            hit == energy_hits(best, ref, panel.HIT_TOL)
            and (not hit or 0 < evals <= charged <= budget)
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


def gmin_paired(groups: dict, system: str) -> dict[str, object]:
    """Both arms of a GMIN panel cluster paired by seed: outcome table, McNemar, Newcombe, Wilcoxon on calls and wall."""
    a = {r["seed"]: r for r in groups[(system, "anneal")]}
    g = {r["seed"]: r for r in groups[(system, "gmin")]}
    table = paired_table({k: r["hit"] == "1" for k, r in a.items()}, {k: r["hit"] == "1" for k, r in g.items()})
    seeds = sorted(a)
    return {
        "table": table,
        "mcnemar": mcnemar(table[1], table[2]),
        "interval": newcombe_paired_interval(*table),
        "calls": wilcoxon_signed_rank([float(a[k]["evals_to_hit"]) - float(g[k]["evals_to_hit"]) for k in seeds]),
        "wall": wilcoxon_signed_rank([float(a[k]["wall_s"]) - float(g[k]["wall_s"]) for k in seeds]),
    }


def gmin_tex_line(audit: Audit, where: str, cells: list[str], groups: dict, system: str, arm: str,
                  budget: int) -> None:
    """One tab:gmin line: Budget, Arm, Hits, Only, Both, Neither, Calls, IQR, Wall, In, Out, McNemar p, paired
    Newcombe interval, Wilcoxon p on calls, Wilcoxon p on wall."""
    group = groups[(system, arm)]
    stats = gmin_paired(groups, system)
    both, only_a, only_g, neither = stats["table"]

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in group]

    audit.cell(where, "hits", cells[2], frac(group))
    audit.cell(where, "seeds this arm alone hits", cells[3], float(only_a if arm == "anneal" else only_g))
    audit.cell(where, "median calls", cells[6], median(col("evals_to_hit")))
    audit.cell(where, "IQR calls", cells[7], iqr(col("evals_to_hit")))
    audit.cell(where, "wall median", cells[8], median(col("wall_s")))
    audit.cell(where, "inside median", cells[9], median(col("pot_s")))
    audit.cell(where, "outside median", cells[10], median(col("outside_s")))
    if arm == "anneal":
        audit.cell(where, "budget", cells[0], float(budget))
        audit.cell(where, "both hit", cells[4], float(both))
        audit.cell(where, "neither hits", cells[5], float(neither))
        audit.cell(where, "exact McNemar p", cells[11], stats["mcnemar"])
        audit.cell(where, "paired Newcombe 95% interval", cells[12], interval_text(*stats["interval"]))
        audit.cell(where, "Wilcoxon p calls", cells[13], stats["calls"][2])
        audit.cell(where, "Wilcoxon p wall", cells[14], stats["wall"][2])
    else:
        for label, cell in zip(("budget", "both", "neither", "McNemar p", "interval", "Wilcoxon p calls",
                                "Wilcoxon p wall"), (cells[0], cells[4], cells[5], *cells[11:])):
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


def bisected_z975() -> float:
    low, high = 0.0, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        if 0.5 * math.erfc(-mid / math.sqrt(2)) < 0.975:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def paired_stats_module():
    spec = importlib.util.spec_from_file_location("paired_stats", ROOT / "scripts/paired_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAPER_CHAIN = (
    "paper/jctc/manuscript.tex", "paper/jctc/si.tex", "paper/jctc/cover-letter.tex",
    "scripts/paired_stats.py", "scripts/gmin_panel/run_panel.py", "paper/jctc/scripts/ase_table.py",
    "scripts/h2o16_crest/summarize.py",
    "paper/jctc/scripts/ablation_table.py", "paper/jctc/scripts/water_rerun.py",
    "paper/jctc/scripts/fig_water_crest.py", "paper/jctc/scripts/water_results_table.py",
    "paper/jctc/scripts/water_significance.py", "paper/jctc/scripts/lj75_gmin_table.py",
)


def paired_statistics(audit: Audit) -> None:
    """The paired tests of the audit against known answers, and against the generators' module on the panel."""
    where = "scripts/paper_audit.py paired statistics"
    audit.cell(where, "McNemar, five discordant seeds one way", "0.0625", mcnemar(5, 0))
    audit.cell(where, "McNemar, 1 against 4", "0.375", mcnemar(1, 4))
    audit.cell(where, "McNemar, 12 against 0", "0.00048828125", mcnemar(12, 0))
    audit.cell(where, "Newcombe's paired worked example, 36 12 2 0", "0.0569 0.3404",
               " ".join(f"{x:.4f}" for x in newcombe_paired_interval(36, 12, 2, 0)))
    audit.cell(where, "Wilcoxon on 1 to 5: pairs, T+, p", "5 15 0.0625",
               " ".join(f"{x:g}" for x in wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])))
    audit.cell(where, "Wilcoxon with a tie and a zero: pairs, T+", "3 1.5",
               " ".join(f"{x:g}" for x in wilcoxon_signed_rank([0.0, 1.0, -1.0, -2.0])[:2]))
    audit.said(SI, "with a relative tolerance\nof \\(10^{-12}\\) on that comparison so that ties to rounding count.")
    audit.said("paper/jctc/scripts/ase_table.py", "from paired_stats import fisher_exact  # noqa: E402")
    audit.said("paper/jctc/scripts/ase_table.py", '"p": {arm: fisher_exact(hits["recommended"], 12, hits[arm], 12)')
    audit.said("scripts/paired_stats.py", "if probability(value) <= observed * (1.0 + 1.0e-12)")
    audit.said("scripts/paired_stats.py", "Z95 = NormalDist().inv_cdf(0.975)")
    audit.said(SI, "\\qquad z = \\Phi^{-1}(0.975),")
    shared = paired_stats_module()
    audit.cell(where, "shared z against the bisected 0.975 normal quantile", f"{bisected_z975():.12f}",
               f"{shared.Z95:.12f}")
    worst = max(abs(shared.fisher_exact(a, 12, b, 12) - fisher(a, 12, b, 12)) for a in range(13) for b in range(13))
    audit.record(worst < 1e-9, where, "the audit and scripts/paired_stats.py agree on Fisher at 12 against 12",
                 "< 1e-9", f"{worst:.1e}")
    literal = re.compile(r"(?<![\d.])1\.9(?:6|59)\d*")
    for path in PAPER_CHAIN:
        found = sorted({m.group(0) for m in literal.finditer(audit.text(path))})
        audit.cell(path, "literal normal quantile", "none", " ".join(found) or "none")
    audit.said(SI, "is 0.0625 at \\(m=5\\) and 0.03125 at \\(m=6\\).")
    audit.cell(SI, "the McNemar floor at five and six discordant seeds", "0.0625 0.03125",
               f"{mcnemar(5, 0)} {mcnemar(6, 0)}")
    audit.said(SI, "There are 2000 replicates, and the interval runs from the type-7 quantile at\n0.025 to the one at "
                   "0.975 of the replicate ratios, with no bias correction.")
    audit.said(SI, "each replicate draws 64 seed indices with\nreplacement, with Python's \\texttt{random.Random(2)}, "
                   "and takes the same\nindices for both arms.")
    rows = audit.rows("bench/gmin-panel/rows.csv")
    budgets = gmin_panel_groups(audit)[0].PAPER_BUDGETS
    misses = [r for r in rows if r["hit"] != "1" and r["system"] in budgets]
    audit.record(all(int(float(r["evals_to_hit"])) == int(r["budget"]) for r in misses), "bench/gmin-panel/rows.csv",
                 "every miss carries the budget as its calls, as the Wilcoxon test counts it", "budget",
                 "budget" if all(int(float(r["evals_to_hit"])) == int(r["budget"]) for r in misses) else "other")
    spec = importlib.util.spec_from_file_location("paired_stats", ROOT / "scripts/paired_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panel, groups = gmin_panel_groups(audit)
    worst = 0.0
    for system in panel.PAPER_BUDGETS:
        mine = gmin_paired(groups, system)
        a = {r["seed"]: r for r in groups[(system, "anneal")]}
        g = {r["seed"]: r for r in groups[(system, "gmin")]}
        seeds = sorted(a)
        table = module.outcome_table([a[k]["hit"] == "1" for k in seeds], [g[k]["hit"] == "1" for k in seeds])
        audit.cell(where, f"{system} outcome table agrees", " ".join(map(str, mine["table"])), " ".join(map(str, table)))
        theirs = [float(module.mcnemar_exact(table[1], table[2])), *module.newcombe_paired(*table)]
        for key in ("calls", "wall"):
            column = "evals_to_hit" if key == "calls" else "wall_s"
            theirs.append(module.wilcoxon_exact([float(a[k][column]) - float(g[k][column]) for k in seeds])[2])
        ours = [mine["mcnemar"], *mine["interval"], mine["calls"][2], mine["wall"][2]]
        worst = max(worst, *(abs(x - y) for x, y in zip(ours, theirs)))
    audit.record(worst < 1e-12, where, "the audit and scripts/paired_stats.py agree on every panel cluster", "< 1e-12",
                 f"{worst:.1e}")
    rerun_panel, rerun = gmin_rerun_groups(audit)
    develop = gmin_develop_groups(audit, rerun_panel)
    worst = 0.0
    for system, rerun in [(s, rerun) for s in GMIN_RERUN_SYSTEMS] + [(s, develop) for s in GMIN_DEVELOP_SYSTEMS]:
        mine = gmin_paired(rerun, system)
        a = {r["seed"]: r for r in rerun[(system, "anneal")]}
        g = {r["seed"]: r for r in rerun[(system, "gmin")]}
        seeds = sorted(a)
        table = module.outcome_table([a[k]["hit"] == "1" for k in seeds], [g[k]["hit"] == "1" for k in seeds])
        theirs = [float(module.mcnemar_exact(table[1], table[2])), *module.newcombe_paired(*table)]
        for column in ("evals_to_hit", "wall_s"):
            theirs.append(module.wilcoxon_exact([float(a[k][column]) - float(g[k][column]) for k in seeds])[2])
        ours = [mine["mcnemar"], *mine["interval"], mine["calls"][2], mine["wall"][2]]
        worst = max(worst, *(abs(x - y) for x, y in zip(ours, theirs)))
    audit.record(worst < 1e-12, where, "the audit and scripts/paired_stats.py agree on every rerun cluster", "< 1e-12",
                 f"{worst:.1e}")


def manuscript_gmin(audit: Audit) -> None:
    """The bench README of the earlier panel against its rows, and tab:gmin against the imported rerun."""
    panel, groups = gmin_panel_groups(audit)
    gmin_panel_summary(audit, panel, groups)
    arms = {"anneal": "anneal", "GMIN": "gmin"}
    by_label = {labels[0]: system for system, labels in panel.LABELS.items()}
    shown = []
    system = None
    for line, row in md_table(audit, GMIN_PANEL_README, "System | Budget | Arm"):
        cells = list(row.values())
        if cells[0]:
            system = by_label.get(cells[0])
        arm = arms.get(cells[2])
        where = f"{GMIN_PANEL_README}:{line} {cells[0] or system} {cells[2]}"
        if system is None or arm is None or len(cells) != 12:
            audit.record(False, where, "table line", " | ".join(cells), "a panel system and arm")
            continue
        shown.append((system, arm))
        gmin_panel_line(audit, where, cells[1:], groups, system, arm, panel.PAPER_BUDGETS[system])
    want = [(system, arm) for system in panel.PAPER_BUDGETS for arm in ("anneal", "gmin")]
    audit.cell(GMIN_PANEL_README, "gmin panel lines", " ".join(f"{s}/{a}" for s, a in shown),
               " ".join(f"{s}/{a}" for s, a in want))
    gmin_panel_readme(audit, panel, groups)
    gmin_rerun(audit)


GMIN_RERUN_REVISION = "eeec8e2c216ae1506768bca8e29846c1c039ca3b"
GMIN_RERUN_SNAPSHOT = "9cbec1a446352bc9ef064bd2b1a5767194655919"
GMIN_RERUN_SYSTEMS = ("lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "m80r6")
TEMPLATE_REMOVAL = "1c1b12384683485cf820d3a244151b48973dc0f7"
SUBMITTED = "61fa67cf380e46876b569da45e7cc091647a775b"
GMIN_DEVELOP = "paper/jctc/data/gmin_develop"
GMIN_DEVELOP_HARNESS = "f009acc96bb95023eeb6e394e863630c752a5d05"
GMIN_DEVELOP_SYSTEMS = ("au55", "m55r6", "m55r14", "m80r6", "m80r14")
GMIN_DEVELOP_HOST = "e06d886d760e0a59870126b963636ff6408201594f2686f72deb66816fccadb3"
GMIN_DEVELOP_SESSION = "20261120T072444Z"


def imported(audit: Audit, folder: str, name: str = "rows.csv", keys: tuple[str, str] = ("source", "blob")) -> str:
    """An imported row file is the blob its PROVENANCE names, at that path of a commit on that record branch."""
    fields = dict(line.split(" ", 1) for line in audit.text(f"{folder}/PROVENANCE").splitlines())
    path = f"{folder}/{name}"
    source, blob = fields[keys[0]], fields[keys[1]]
    local = git("hash-object", str(ROOT / path)).stdout.strip()
    audit.cell(path, "blob of the copy", blob, local)
    audit.cell(path, "blob at the record commit", blob, git("rev-parse", f"{fields['commit']}:{source}").stdout.strip())
    on_branch = git("merge-base", "--is-ancestor", fields["commit"], f"refs/remotes/origin/{fields['branch']}").returncode == 0
    audit.record(on_branch, path, "the record commit is on the record branch of origin", fields["branch"],
                 "on it" if on_branch else "not on it")
    return path


def ancestor_of(audit: Audit, where: str, revision: str, of: str = "HEAD") -> bool:
    ok = git("merge-base", "--is-ancestor", revision, of).returncode == 0
    audit.record(ok, where, f"{revision[:8]} is an ancestor of {of[:8]}", "ancestor", "ancestor" if ok else "not")
    return ok


def gmin_rerun_groups(audit: Audit) -> tuple[object, dict[tuple[str, str], list[dict[str, str]]]]:
    """The four imported shards of the eeec8e2c rerun, checked row by row, grouped by cluster and code in seed order."""
    panel = load_script("gmin_rerun_panel", "scripts/gmin_panel/run_panel.py")
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    rows, bad, first = [], [], defaultdict(lambda: defaultdict(int))
    for shard in GMIN_SHARDS:
        path = imported(audit, f"{GMIN_RERUN}/{shard}")
        seen = set()
        for index, r in enumerate(audit.rows(path), start=2):
            key = (r["system"], r["seed"])
            if key not in seen:
                first[r["system"]][r["code"]] += 1
                seen.add(key)
            budget, ref = panel.PAPER_BUDGETS.get(r["system"]), references.get(r["system"])
            want = GMIN_RERUN_REVISION if r["code"] == "anneal" else GMIN_RERUN_SNAPSHOT
            hit, best = r["hit"] == "1", float(r["best"])
            evals, charged = int(r["evals_to_hit"]), int(r["charged"])
            wall, pot, outside = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
            ok = (budget is not None and int(r["budget"]) == budget and r["hit"] in ("0", "1")
                  and r["code_revision"] == want and r["harness_revision"] == GMIN_RERUN_REVISION
                  and hit == energy_hits(best, ref, panel.HIT_TOL)
                  and (not hit or 0 < evals <= charged <= budget) and (hit or evals == charged == budget)
                  and 0.0 <= pot <= wall + 2e-6 and abs(max(wall - pot, 0.0) - outside) <= 2e-6)
            if not ok:
                bad.append(f"{shard}:{index}")
            rows.append(r)
    audit.cell(GMIN_RERUN, "row invariants and pinned revisions", " ".join(bad) or "ok", "ok")
    keys = [(r["system"], r["code"], r["seed"]) for r in rows]
    audit.cell(GMIN_RERUN, "each cluster, code and seed once", str(len(keys)), float(len(set(keys))))
    groups = {}
    for system in GMIN_RERUN_SYSTEMS:
        for code in ("anneal", "gmin"):
            group = sorted((r for r in rows if r["system"] == system and r["code"] == code), key=lambda r: int(r["seed"]))
            audit.cell(f"{GMIN_RERUN} {system} {code}", "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
            groups[(system, code)] = group
        audit.cell(f"{GMIN_RERUN} {system}", "seeds with this code first and with GMIN first", "6 6",
                   f"{first[system]['anneal']} {first[system]['gmin']}")
    audit.cell(GMIN_RERUN, "clusters", " ".join(GMIN_RERUN_SYSTEMS), " ".join(sorted({r["system"] for r in rows},
               key=GMIN_RERUN_SYSTEMS.index)))
    ancestor_of(audit, GMIN_RERUN, GMIN_RERUN_REVISION)
    ancestor_of(audit, GMIN_RERUN, TEMPLATE_REMOVAL, GMIN_RERUN_REVISION)
    hosts = imported(audit, f"{GMIN_RERUN}/b", "host.csv", ("host_source", "host_blob"))
    machine = {r["system"]: {x["machine_id_sha256"] for x in audit.rows(hosts) if x["system"] == r["system"]}
               for r in audit.rows(hosts)}
    for r in rows:
        for column in ("machine_id_sha256", "machine_id"):
            if column in r:
                machine.setdefault(r["system"], set()).add(r[column])
    ids = {system: " ".join(sorted(v)) for system, v in machine.items()}
    lj150 = "9eb296774121518feb1b187b0f6a109a42de7a5dd39e62ccd37d6e7d329604b8"
    main_host = "3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873"
    for system in GMIN_RERUN_SYSTEMS:
        audit.cell(f"{GMIN_RERUN} {system}", "machine id", lj150 if system == "lj150" else main_host, ids.get(system))
    audit.said(SI, f"LJ\\(_{{104}}\\), and M\\(_{{80}}\\) of the first block ran on\n"
                   f"\\texttt{{\\seqsplit{{{main_host}}}}},\nLJ\\(_{{150}}\\) on\n\\texttt{{\\seqsplit{{{lj150}}}}},")
    audit.said(SI, "pinned to core 0, in alternating order by seed: six seeds with this code\nfirst and six with GMIN first.")
    for shard in GMIN_SHARDS:
        commit = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_RERUN}/{shard}/PROVENANCE").splitlines())["commit"]
        for patch in ("bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
                      "bench/metal-cluster-h2h/gmin-bgupta.patch", "scripts/gmin_panel/build_gmin.sh",
                      "examples/h2h_gmin_lj.data", "examples/h2h_gmin_morse.data"):
            audit.cell(f"{GMIN_RERUN}/{shard}", f"{patch} at the record commit is the one at HEAD",
                       git("rev-parse", f"HEAD:{patch}").stdout.strip(),
                       git("rev-parse", f"{commit}:{patch}").stdout.strip())
    snapshot = {r.get("svnrev", GMIN_RERUN_SNAPSHOT) for r in rows if r["code"] == "gmin"}
    audit.cell(GMIN_RERUN, "GMIN SVNREV", GMIN_RERUN_SNAPSHOT, " ".join(sorted(snapshot)))
    tarballs = {r["tarball_sha256"] for r in rows if r.get("tarball_sha256")}
    audit.cell(GMIN_RERUN, "GMIN tarball", "ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776",
               " ".join(sorted(tarballs)))
    audit.said(SI, f"SVNREV \\texttt{{{GMIN_RERUN_SNAPSHOT}}}, tarball sha256 "
                   "\\texttt{\\seqsplit{ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776}}")
    return panel, groups


def gmin_develop_groups(audit: Audit, panel: object) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The imported session of the submitted revision on gold and Morse, checked row by row."""
    path = imported(audit, GMIN_DEVELOP)
    fields = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_DEVELOP}/PROVENANCE").splitlines())
    audit.cell(f"{GMIN_DEVELOP}/PROVENANCE", "ancestor", SUBMITTED, fields["ancestor"])
    ancestor_of(audit, GMIN_DEVELOP, SUBMITTED)
    ancestor_of(audit, GMIN_DEVELOP, TEMPLATE_REMOVAL, SUBMITTED)
    ancestor_of(audit, GMIN_DEVELOP, SUBMITTED, GMIN_DEVELOP_HARNESS)
    ancestor_of(audit, GMIN_DEVELOP, GMIN_DEVELOP_HARNESS, fields["commit"])
    changed = git("diff", "--name-only", SUBMITTED, GMIN_DEVELOP_HARNESS).stdout.split()
    outside = [f for f in changed if not f.startswith(("scripts/paper_rerun/tuned_gmin/", "bench/paper-rerun/tuned-gmin/"))]
    audit.cell(GMIN_DEVELOP, "files the harness commit changes outside its own folders", "scripts/gmin_panel/build_gmin.sh",
               " ".join(outside))
    build = git("diff", "-U0", SUBMITTED, GMIN_DEVELOP_HARNESS, "--", "scripts/gmin_panel/build_gmin.sh").stdout
    edits = [x for x in build.splitlines() if x[:1] in "+-" and not x.startswith(("+++", "---"))]
    audit.cell(GMIN_DEVELOP, "the build script change", '-    make -j"$(nproc)" GMIN | +    make -j"${GMIN_PANEL_JOBS:-$(nproc)}" GMIN',
               " | ".join(edits))
    references = {row[0]: row[3] for row in panel.SYSTEMS}
    rows, bad, first, seen = audit.rows(path), [], defaultdict(lambda: defaultdict(int)), set()
    for index, r in enumerate(rows, start=2):
        if (r["system"], r["seed"]) not in seen:
            first[r["system"]][r["code"]] += 1
            seen.add((r["system"], r["seed"]))
        budget, ref = panel.PAPER_BUDGETS.get(r["system"]), references.get(r["system"])
        want = SUBMITTED if r["code"] == "anneal" else GMIN_RERUN_SNAPSHOT
        hit, best = r["hit"] == "1", float(r["best"])
        evals, charged = int(r["evals_to_hit"]), int(r["charged"])
        wall, pot, outside_s = float(r["wall_s"]), float(r["pot_s"]), float(r["outside_s"])
        ok = (budget is not None and int(r["budget"]) == budget and r["hit"] in ("0", "1")
              and r["code_revision"] == want and r["harness_revision"] == GMIN_DEVELOP_HARNESS
              and r["host"] == GMIN_DEVELOP_HOST and r["session"] == GMIN_DEVELOP_SESSION
              and hit == energy_hits(best, ref, panel.HIT_TOL)
              and (not hit or 0 < evals <= charged <= budget) and (hit or evals == charged == budget)
              and 0.0 <= pot <= wall + 2e-6 and abs(max(wall - pot, 0.0) - outside_s) <= 2e-6)
        if not ok:
            bad.append(str(index))
    audit.cell(GMIN_DEVELOP, "row invariants, pinned revisions, host and session", " ".join(bad) or "ok", "ok")
    keys = [(r["system"], r["code"], r["seed"]) for r in rows]
    audit.cell(GMIN_DEVELOP, "each cluster, code and seed once", str(len(keys)), float(len(set(keys))))
    audit.cell(GMIN_DEVELOP, "clusters", " ".join(GMIN_DEVELOP_SYSTEMS),
               " ".join(sorted({r["system"] for r in rows}, key=lambda s: GMIN_DEVELOP_SYSTEMS.index(s)
                               if s in GMIN_DEVELOP_SYSTEMS else 99)))
    groups = {}
    for system in GMIN_DEVELOP_SYSTEMS:
        for code in ("anneal", "gmin"):
            group = sorted((r for r in rows if r["system"] == system and r["code"] == code), key=lambda r: int(r["seed"]))
            audit.cell(f"{GMIN_DEVELOP} {system} {code}", "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
            groups[(system, code)] = group
        audit.cell(f"{GMIN_DEVELOP} {system}", "seeds with this code first and with GMIN first", "6 6",
                   f"{first[system]['anneal']} {first[system]['gmin']}")
    for name in ("bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
                 "bench/metal-cluster-h2h/gmin-bgupta.patch", "examples/h2h_gmin_lj.data",
                 "examples/h2h_gmin_morse.data", "scripts/gmin_panel/run_panel.py", "examples/h2h_metal.rs",
                 "examples/h2h_morse.rs"):
        audit.cell(GMIN_DEVELOP, f"{name} at the harness commit is the one of the submitted revision",
                   git("rev-parse", f"{SUBMITTED}:{name}").stdout.strip(),
                   git("rev-parse", f"{GMIN_DEVELOP_HARNESS}:{name}").stdout.strip())
    runner = git("show", f"{GMIN_DEVELOP_HARNESS}:scripts/paper_rerun/tuned_gmin/run.py").stdout
    for needle in ('panel.gmin_data(system, seed, lj_data, morse_data)', 'if key.startswith(("H2H_", "ANNEAL_")):'):
        audit.record(needle in runner, GMIN_DEVELOP, f"the harness runs {needle}", "present",
                     "present" if needle in runner else "absent")
    metal = git("show", f"{SUBMITTED}:examples/h2h_metal.rs").stdout
    audit.record('cfg.basin_stride = std::env::var("H2H_BASIN_STRIDE")' in metal and ".unwrap_or(8)" in metal,
                 GMIN_DEVELOP, "the gold driver of the submitted revision uses a basin stride of 8", "8",
                 "8" if ".unwrap_or(8)" in metal else "other")
    return groups


def gmin_develop_data(audit: Audit, panel: object) -> None:
    """The gold input of the second block, from the panel script of the submitted revision."""
    data = panel.gmin_data("au55", 2000, "", "")
    for line in ("BGUPTAT 55 0.2061 10.229 4.036 1.79 2.884", "TEMPERATURE 0.1", "NEWRESTART 40", "EDIFF 0.003",
                 f"STEP {0.40 * 2.884 / 2 ** (1 / 6):.6f} 0.51", f"AVOID {4.0 * 2.884 / 2 ** (1 / 6):.6f} 20 F",
                 f"RADIUS {panel.gmin_gupta_radius(55):.6f}"):
        audit.record(line in data, "scripts/gmin_panel/run_panel.py", f"the Au55 GMIN input sets {line}", line,
                     "set" if line in data else "absent")
    audit.said(SI, "The Au\\(_{55}\\) row runs GMIN's Gupta potential with the gold row of Darby et al., "
                   "\\(A=0.2061\\)~eV, \\(p=10.229\\), \\(q=4.036\\), \\(\\xi=1.79\\)~eV, and \\(r_0=2.884\\)~\\AA, at "
                   "\\(T=0.1\\)~eV, with step 0.40 and taboo radius 4.0 multiplied by \\(r_0/2^{1/6}\\), GMIN's "
                   "container radius, \\texttt{NEWRESTART 40}, and an energy difference of 0.003.")
    audit.said(TEX, "On Au\\(_{55}\\) GMIN runs the Gupta potential with the gold parameters of Darby et al.\\ "
                    "\\cite{darbyCuAu2002} at \\(T=0.1\\)~eV, with the step and the taboo radius scaled by "
                    "\\(r_0/2^{1/6}\\) and GMIN's container radius, and the gold driver of this code applies the basin "
                    "bias of equation~\\ref{eq:delta} on a stride of 8.")
    audit.said(TEX, "On Au\\(_{55}\\) the gold driver of this code applies the basin bias of equation~\\ref{eq:delta} "
                    "on a stride of 8.")
    refs = {row[0]: row[3] for row in panel.SYSTEMS}
    audit.said(SI, f"\\({refs['au55']:.8f}\\)~eV at Au\\(_{{55}}\\), \\({refs['m55r6']:.6f}\\) and "
                   f"\\({refs['m55r14']:.6f}\\) at M\\(_{{55}}\\) (\\(\\rho=6\\) and 14), and \\({refs['m80r6']:.6f}\\) and "
                   f"\\({refs['m80r14']:.6f}\\) at M\\(_{{80}}\\) (\\(\\rho=6\\) and 14).")
    audit.said(SI, f"and every cluster of the second block on\n\\texttt{{\\seqsplit{{{GMIN_DEVELOP_HOST}}}}}.")
    audit.said(SI, "and for the second by the same script at the record commit \\texttt{f009acc9} on the submitted "
                   "revision, which only sets the number of parallel build jobs,")
    audit.said(SI, f"The rows of this code in the second block are the submitted revision "
                   f"\\texttt{{\\seqsplit{{{SUBMITTED}}}}}, session {GMIN_DEVELOP_SESSION} of "
                   "\\path{paper/jctc/data/gmin_develop}, copied from the record branch its \\texttt{PROVENANCE} file "
                   "names.")
    audit.said(SI, "The GMIN table of the main text uses three GMIN inputs.")


def gmin_develop_prose(audit: Audit, panel: object, groups: dict) -> None:
    """The results, abstract, conclusion and cover sentences of the second block, from its rows."""
    paired = {s: gmin_paired(groups, s) for s in GMIN_DEVELOP_SYSTEMS}
    t = {s: paired[s]["table"] for s in GMIN_DEVELOP_SYSTEMS}
    p = {s: pp(paired[s]["mcnemar"]) for s in GMIN_DEVELOP_SYSTEMS}
    hits = {(s, a): hits_of(groups[(s, a)]) for s in GMIN_DEVELOP_SYSTEMS for a in ("anneal", "gmin")}
    ahead = [s for s in GMIN_DEVELOP_SYSTEMS if t[s][1] > t[s][2] and paired[s]["mcnemar"] < 0.05]
    audit.cell(TEX, "clusters of the second block where this code hits more at p<0.05", "m55r14 m80r14", " ".join(ahead))
    audit.cell(TEX, "GMIN hits on M55 and M80 at rho 14", "0 0", f"{hits[('m55r14', 'gmin')]} {hits[('m80r14', 'gmin')]}")
    a, b = paired["m55r14"]["interval"], paired["m80r14"]["interval"]
    audit.said(TEX, f"At the submitted revision, this code alone hits {t['m55r14'][1]} seeds of M\\(_{{55}}\\) at "
                    f"\\(\\rho=14\\) and GMIN alone none, with \\(p={p['m55r14']}\\) and an interval from {a[0]:.2f} to "
                    f"{a[1]:.2f}, and {t['m80r14'][1]} seeds of M\\(_{{80}}\\) at \\(\\rho=14\\) against none, with "
                    f"\\(p={p['m80r14']}\\) and an interval from {b[0]:.2f} to {b[1]:.2f}.")
    audit.said(TEX, f"On Au\\(_{{55}}\\) this code alone hits {t['au55'][1]} seeds and GMIN alone {t['au55'][2]}, with "
                    f"\\(p={p['au55']}\\).")
    audit.cell(TEX, "both arms hit M55 and M80 at rho 6 on every seed", "12 0 0 0 12 0 0 0",
               " ".join(map(str, t["m55r6"] + t["m80r6"])))
    calls = median([float(r["evals_to_hit"]) for r in groups[("m80r6", "anneal")]])
    audit.said(TEX, "Both arms hit M\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed, and on M\\(_{80}\\) at "
                    f"\\(\\rho=6\\) this code takes the same median of {calls:.0f} calls at both revisions.")
    hits_m = f"{hits[('m55r14', 'anneal')]} and {hits[('m80r14', 'anneal')]} of 12 against none"
    audit.said(TEX, f"Morse M\\(_{{55}}\\) and M\\(_{{80}}\\) at \\(\\rho=14\\) on {hits_m} "
                    f"(\\(p={p['m55r14']}\\) and {p['m80r14']}), and rigid")
    audit.said(TEX, f"on Morse M\\(_{{55}}\\) and M\\(_{{80}}\\) at \\(\\rho=14\\) at the submitted revision, "
                    f"{hits[('m55r14', 'anneal')]} and {hits[('m80r14', 'anneal')]} of 12 against none, and on rigid")
    audit.said(TEX, "Table~\\ref{tab:gmin} runs GMIN against this code at two revisions: revision \\texttt{eeec8e2c}, "
                    "an ancestor of the submitted revision with the structure templates removed, on eight Lennard-Jones "
                    "sizes and Morse M\\(_{80}\\) at \\(\\rho=6\\), and the submitted revision \\texttt{61fa67cf} on "
                    "Gupta Au\\(_{55}\\) and on Morse M\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) and 14.")
    audit.said(TEX, "Each block runs both arms of each cluster on one core of one host, on seeds 2000--2011 in "
                    "alternating order, at the budget in the table, from the start this code writes for the seed.")
    audit.said(COVER, "Paired by seed, the code hits more seeds than GMIN on a Lennard-Jones cluster, on two Morse "
                      "clusters, and on two rigid TIP4P water clusters,")


def gmin_rerun(audit: Audit) -> None:
    """tab:gmin and its prose, recomputed from the imported eeec8e2c shards and the session of the submitted revision."""
    panel, groups = gmin_rerun_groups(audit)
    develop = gmin_develop_groups(audit, panel)
    by_label = {labels[1]: system for system, labels in panel.LABELS.items()}
    shown, system, block, heads = [], None, None, []
    for row in tex_table(audit, TEX, "tab:gmin"):
        cells = [c.strip() for c in row]
        if "\\multicolumn{16}" in cells[0]:
            heads.append(cells[0].replace("\\midrule", "").strip())
            block = "develop" if "61fa67cf" in cells[0] else "rerun"
            system = None
            continue
        if cells[0]:
            system = by_label.get(cells[0])
        arm = {"this work": "anneal", "GMIN": "gmin"}.get(cells[2])
        where = f"{TEX} tab:gmin {block} {system} {cells[2]}"
        source = develop if block == "develop" else groups
        if block is None or system is None or arm is None or len(cells) != 16 or (system, arm) not in source:
            audit.record(False, where, "table line", " & ".join(cells), "a rerun cluster and arm")
            continue
        shown.append((block, system, arm))
        gmin_tex_line(audit, where, cells[1:], source, system, arm, panel.PAPER_BUDGETS[system])
    audit.cell(TEX, "tab:gmin block headings",
               "\\multicolumn{16}{@{}l}{\\emph{Revision \\texttt{eeec8e2c}, an ancestor of the submitted revision}} | "
               "\\multicolumn{16}{@{}l}{\\emph{The submitted revision \\texttt{61fa67cf}}}", " | ".join(heads))
    want = ([("rerun", s, a) for s in GMIN_RERUN_SYSTEMS for a in ("anneal", "gmin")]
            + [("develop", s, a) for s in GMIN_DEVELOP_SYSTEMS for a in ("anneal", "gmin")])
    audit.cell(TEX, "tab:gmin lines", " ".join("/".join(x) for x in shown), " ".join("/".join(x) for x in want))
    gmin_develop_data(audit, panel)
    gmin_develop_prose(audit, panel, develop)
    paired = {s: gmin_paired(groups, s) for s in GMIN_RERUN_SYSTEMS}
    t = {s: paired[s]["table"] for s in GMIN_RERUN_SYSTEMS}
    p = {s: pp(paired[s]["mcnemar"]) for s in GMIN_RERUN_SYSTEMS}

    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    def walls(system: str, arm: str) -> float:
        return median([float(r["wall_s"]) for r in groups[(system, arm)]])

    lo, hi = paired["lj98"]["interval"]
    audit.said(TEX, f"Paired by seed, this code alone hits {t['lj98'][1]} seeds of LJ\\(_{{98}}\\) and GMIN alone "
                    f"{'none' if t['lj98'][2] == 0 else t['lj98'][2]}, with \\(p={p['lj98']}\\) and an interval from {lo:.2f} to {hi:.2f}.")
    audit.cell(TEX, "LJ38 and LJ75: discordant seeds", "5 0 5 0",
               f"{t['lj38'][1]} {t['lj38'][2]} {t['lj75'][1]} {t['lj75'][2]}")
    audit.said(TEX, f"On LJ\\(_{{38}}\\) and LJ\\(_{{75}}\\) this code alone hits {t['lj38'][1]} seeds and GMIN alone "
                    f"{'none' if t['lj38'][2] == 0 else t['lj38'][2]}, with \\(p={p['lj38']}\\), the smallest value the exact test can "
                    f"reach with five discordant seeds; this code hits LJ\\(_{{38}}\\) on {hits('lj38', 'anneal')} and "
                    f"LJ\\(_{{75}}\\) on {hits('lj75', 'anneal')} of 12, and GMIN on {hits('lj38', 'gmin')} and "
                    f"{hits('lj75', 'gmin')}.")
    audit.cell(TEX, "LJ38 and LJ75 p is the five-seed floor", "0.0625 0.0625",
               f"{paired['lj38']['mcnemar']} {paired['lj75']['mcnemar']}")
    big = ("lj102", "lj103", "lj104", "lj150")
    audit.said(TEX, "On LJ\\(_{102}\\), LJ\\(_{103}\\), LJ\\(_{104}\\), and LJ\\(_{150}\\) the seeds that one arm "
                    "alone hits are " + series(f"{t[s][1]} to {t[s][2]}" for s in big) + ", with \\(p\\) of "
                    + series(p[s] for s in big) + ".")
    audit.cell(TEX, "both arms hit LJ55 and M80 at rho 6 on every seed", "12 0 0 0 12 0 0 0",
               " ".join(map(str, t["lj55"] + t["m80r6"])))
    audit.said(TEX, "Both arms hit LJ\\(_{55}\\) and M\\(_{80}\\) at \\(\\rho=6\\) on every seed.")
    lower = [s for s in GMIN_RERUN_SYSTEMS if walls(s, "anneal") < walls(s, "gmin")]
    audit.cell(TEX, "clusters where the median wall of this code is lower", str(len(GMIN_RERUN_SYSTEMS)),
               float(len(lower)))
    faster = [s for s in GMIN_RERUN_SYSTEMS if paired[s]["wall"][2] < 0.05]
    favor = all(paired[s]["wall"][1] < paired[s]["wall"][0] * (paired[s]["wall"][0] + 1) / 4 for s in faster)
    audit.record(favor, TEX, "every paired wall test below 0.05 favors this code", "this code",
                 "this code" if favor else "GMIN on one")
    audit.said(TEX, f"The median wall of a run is lower for this code on all {NUMBER_WORDS[len(lower)]} clusters, and "
                    f"the paired Wilcoxon test on the walls is below 0.05 on {NUMBER_WORDS[len(faster)]} of them: from "
                    f"{panel.show_s(walls('lj38', 'anneal'))}~s against {panel.show_s(walls('lj38', 'gmin'))}~s on "
                    f"LJ\\(_{{38}}\\) to {panel.show_s(walls('lj150', 'anneal'))}~s against "
                    f"{panel.show_s(walls('lj150', 'gmin'))}~s on LJ\\(_{{150}}\\).")
    audit.said(TEX, f"hits LJ\\(_{{98}}\\) on {hits('lj98', 'anneal')} of 12 seeds and GMIN on none")
    ahead = [x for x in GMIN_RERUN_SYSTEMS if hits(x, "anneal") > hits(x, "gmin") and paired[x]["mcnemar"] < 0.05]
    audit.cell(COVER, "Lennard-Jones clusters where the code hits more than GMIN at p<0.05", "lj98", " ".join(ahead))
    audit.said(TEX, f"GMIN on none (\\(p={p['lj98']}\\)), Morse M\\(_{{55}}\\)")
    audit.said(TEX, f"With the templates removed, the code hits more seeds than GMIN on LJ\\(_{{98}}\\), "
                    f"{hits('lj98', 'anneal')} of 12 against none,")
    misses = [float(r["best"]) for s in ("lj75",) for a in ("anneal", "gmin") for r in groups[(s, a)] if r["hit"] == "0"]
    lj75_ref = {row[0]: row[3] for row in panel.SYSTEMS}["lj75"]
    audit.cell(TEX, "every LJ75 miss of tab:gmin ends at -396.2822", "-396.2822",
               f"{min(misses):.4f}" if f"{min(misses):.4f}" == f"{max(misses):.4f}" else f"{min(misses)} to {max(misses)}")
    audit.cell(TEX, "the LJ75 miss gap", "1.210", max(misses) - lj75_ref)
    audit.said(TEX, "Every miss on LJ\\(_{75}\\) in Table~\\ref{tab:gmin}, of either code, ends at the icosahedral "
                    "competitor at \\(-396.2822\\), \\(1.210\\) above the published decahedral minimum at "
                    f"\\({lj75_ref:.6f}\\).")
    lj_data = audit.text("examples/h2h_gmin_lj.data")
    for line in ("TEMPERATURE 1.0", "STEP 0.40 0.51", "NEWRESTART 40", "AVOID 4.0 20 F", "EDIFF 0.003"):
        audit.record(line in lj_data and line in audit.text("examples/h2h_gmin_morse.data"), "examples/h2h_gmin_lj.data",
                     f"both GMIN inputs set {line}", line, "set" if line in lj_data else "absent")
    audit.said(TEX, "file: \\(T=1.0\\), step \\(0.40\\), a taboo list of the last \\(20\\) minima within \\(4.0\\), and a "
                    "restart after \\(40\\) steps without a new lowest minimum; the Morse rows add GMIN's native "
                    "\\texttt{MORSE} keyword and \\(\\rho\\).")
    for path in (TEX, SI):
        part = audit.text(path)
        if path == TEX:
            part = part.split("\\label{sec:protocol}", 1)[1].split("\\label{sec:transfer}", 1)[0]
        else:
            part = part.split("\\label{sec:si-gmin}", 1)[1].split("\\section{Host and commands}", 1)[0]
        for word in ("1ce3ab64", "c2de652d", "prior-on", "template-free", "Fisher", "Mann--Whitney"):
            audit.record(word not in part, path, f"the GMIN comparison does not print {word}", "absent",
                         "absent" if word not in part else "present")
    for path in (TEX, SI):
        audit.record("Mann--Whitney" not in audit.text(path), path, "no Mann--Whitney test is reported", "absent",
                     "absent" if "Mann--Whitney" not in audit.text(path) else "present")
    gmin_patches(audit)


def series(items) -> str:
    """A list with the serial comma: ``a``, ``a and b``, ``a, b, and c``."""
    items = list(items)
    return " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]


def morse_pair(rho: float, r: float) -> float:
    """The Doye-Wales Morse pair with unit well depth and unit pair minimum."""
    e = math.exp(rho * (1.0 - r))
    return e * (e - 2.0)


def trap_minima(energies: list[float], tol: float) -> list[tuple[float, int]]:
    """Sorted best energies, split wherever two neighbors differ by the hit tolerance or more."""
    groups: list[list[float]] = []
    for energy in sorted(energies):
        if groups and energy - groups[-1][-1] < tol:
            groups[-1].append(energy)
        else:
            groups.append([energy])
    return [(g[0], len(g)) for g in groups]


KNOCKOUT_DATA = "paper/jctc/data/knockouts"
PRIOR_SWITCH = "4f5ec0c56779e02b7627b7d3693fbb319f631176"
# The commits whose code each arm of a knockout block ran, with and without the switch.
KNOCKOUT_CODE = {
    "priors": (PRIOR_SWITCH, PRIOR_SWITCH),
    "templates": ("628773d6837ea2d959a27256ff4163b21162fd4d", "1c1b12384683485cf820d3a244151b48973dc0f7"),
    "restart": ("d12f8825", "7192e9f8d5fff6e581c00b4b1e30aa883dad00de"),
    "pair-scale": ("e5883055", "628773d6837ea2d959a27256ff4163b21162fd4d"),
}
# How the rows of each block name the revision of an arm.
KNOCKOUT_REVISION = {"priors": ("code_revision", "f2a628c0", "f2a628c0"),
                     "templates": ("revision", "628773d6", "1c1b1238"),
                     "restart": ("code_revision", "d12f8825", "7192e9f8"),
                     "pair-scale": ("code_revision", "e5883055", "628773d6")}


def interval_of(cell: str) -> tuple[str, str]:
    low, _, high = cell.replace("\\(-\\)", "-").partition(" to ")
    return low.strip(), high.strip()


def knockout_blocks(audit: Audit) -> list[tuple[str, list[list[str]]]]:
    """The heading and the lines of each block of tab:knockouts."""
    blocks: list[tuple[str, list[list[str]]]] = []
    for cells in tex_table(audit, TEX, "tab:knockouts"):
        cells = [c.replace("\\midrule", "").strip() for c in cells]
        if cells[0].startswith("\\multicolumn"):
            blocks.append((re.search(r"\\emph\{(.*)\}\}", cells[0]).group(1), []))
        elif blocks:
            blocks[-1][1].append(cells)
    return blocks


def knockout_cells(audit: Audit, knock: object, knockout: object) -> dict[str, dict]:
    """Each system of a block recomputed from its imported rows: budget, seeds, the paired table, p, interval."""
    path = imported(audit, f"{KNOCKOUT_DATA}/{knockout.key}")
    column, with_rev, without_rev = KNOCKOUT_REVISION[knockout.key]
    arms: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    twice = []
    for r in audit.rows(path):
        if not knockout.keep(r) or r[knockout.column] not in (knockout.with_arm, knockout.without_arm):
            continue
        if r["seed"] in arms[(r["system"], r[knockout.column])]:
            twice.append(f"{r['system']}/{r['seed']}")
        arms[(r["system"], r[knockout.column])][r["seed"]] = r
        want = with_rev if r[knockout.column] == knockout.with_arm else without_rev
        if not r[column].startswith(want):
            twice.append(f"{r['system']}/{r['seed']} at {r[column][:8]}")
    audit.cell(path, f"{knockout.key}: rows at the block revisions, one per system, arm, and seed", "none",
               " ".join(twice) or "none")
    out = {}
    for system in sorted({s for s, _ in arms}):
        first, second = arms.get((system, knockout.with_arm), {}), arms.get((system, knockout.without_arm), {})
        if not first or not second:
            continue
        if sorted(first) != sorted(second):
            audit.record(False, path, f"{knockout.key} {system}: both arms ran the same seeds", "same", "different")
            continue
        hit = knockout.hit
        table = paired_table({k: first[k][hit].strip().lower() in ("1", "true") for k in first},
                             {k: second[k][hit].strip().lower() in ("1", "true") for k in second})
        budgets = {r["budget"] for r in (*first.values(), *second.values())}
        out[system] = {"budget": budgets.pop() if len(budgets) == 1 else "mixed", "n": len(first), "table": table,
                       "p": mcnemar(table[1], table[2]), "interval": newcombe_paired_interval(*table)}
    return out


def manuscript_knockouts(audit: Audit) -> None:
    """tab:knockouts and tab:tip4p-gmin recomputed from the imported rows, the LJ75 rule, and their prose."""
    knock = load_script("knockout_tables_audit", "paper/jctc/scripts/knockout_tables.py")
    blocks = knockout_blocks(audit)
    audit.cell(TEX, "knockout blocks", " | ".join(k.title for k in knock.KNOCKOUTS), " | ".join(t for t, _ in blocks))
    cells_of = {}
    for knockout, (title, lines) in zip(knock.KNOCKOUTS, blocks):
        cells = knockout_cells(audit, knock, knockout)
        cells_of[knockout.key] = cells
        by_name = {knock.name(system): system for system in cells}
        printed = []
        for line in lines:
            system = by_name.get(line[0])
            where = f"{TEX} tab:knockouts {knockout.key} {line[0]}"
            if system is None or len(line) != 9:
                audit.record(False, where, "a system the rows hold", " & ".join(line), "not in the rows")
                continue
            printed.append(system)
            c = cells[system]
            both, only_with, only_without, _ = c["table"]
            audit.cell(where, "budget", line[1], float(c["budget"]) if c["budget"] != "mixed" else "mixed")
            audit.cell(where, "seeds", line[2], float(c["n"]))
            audit.cell(where, "with", line[3], f"{both + only_with}/{c['n']}")
            audit.cell(where, "without", line[4], f"{both + only_without}/{c['n']}")
            audit.cell(where, "only with", line[5], float(only_with))
            audit.cell(where, "only without", line[6], float(only_without))
            audit.cell(where, "exact McNemar p", line[7], c["p"])
            low, high = interval_of(line[8])
            audit.cell(where, "Newcombe lower", low, c["interval"][0])
            audit.cell(where, "Newcombe upper", high, c["interval"][1])
        code = KNOCKOUT_CODE[knockout.key]
        ancestors = all(git("merge-base", "--is-ancestor", rev, SUBMITTED).returncode == 0 for rev in code)
        audit.record(("lj75" in printed) == (ancestors and "lj75" in cells), TEX,
                     f"{knockout.key}: LJ75 is printed only where both arms ran the code of ancestors",
                     "printed" if ancestors and "lj75" in cells else "left out",
                     "printed" if "lj75" in printed else "left out")
        audit.cell(TEX, f"{knockout.key}: printed systems are the systems of the rows less the left-out ones",
                   " ".join(sorted(set(cells) - set(knockout.omitted))), " ".join(sorted(printed)))
        for system in knockout.omitted:
            audit.record(system == "lj75" and not ancestors, TEX, f"{knockout.key}: {system} is left out only for the "
                         "LJ75 rule", "lj75 outside history", f"{system} {'ancestors' if ancestors else 'outside'}")
    audit.said(TEX, "LJ\\(_{75}\\) is printed only in a block whose two arms ran the code of ancestors of the "
                    "submitted revision.")
    base = git("show", "7bdf0d86c327dd1da8a5065d3e4e965669565148:bench/gmin-losses2/baseline-source.txt").stdout
    for line in ("revision=628773d6837ea2d959a27256ff4163b21162fd4d", "template_ablation_patch_sha256=1c5575a2",
                 "patched_tree=a9cef902"):
        audit.record(line in base, "record/gmin-losses bench/gmin-losses2/baseline-source.txt",
                     "the length-unit baseline", line, "present" if line in base else "absent")
    audit.said(TEX, "\\texttt{e5883055} against \\texttt{628773d6} with a patch that removes the structure templates "
                    "(tree \\texttt{a9cef902}), raises the hits on M\\(_{80}\\) at")
    audit.said(SI, "the length-unit block the gate of \\texttt{e5883055} against \\texttt{628773d6} with the patch of "
                   "sha256 \\texttt{1c5575a2} that removes the structure templates, tree \\texttt{a9cef902}.")
    readme = git("show", "7bdf0d86c327dd1da8a5065d3e4e965669565148:bench/gmin-losses2/README.md").stdout
    audit.record("(`baseline` at `628773d6` vs `fix2` at `e5883055`)" in readme, "record/gmin-losses",
                 "the gate names e5883055", "named", "checked")
    session = git("show", "486ef2f6e8150b6cc37e0944c65c8d283cf54852", "--format=%B", "--no-patch").stdout
    audit.record("The deciding session is develop\n7192e9f8 against d12f8825" in session, "record/lj150d-restart",
                 "the restart session is d12f8825 against develop 7192e9f8", "named", "checked")
    audit.said(SI, "The restart block is the session of \\texttt{d12f8825} against develop \\texttt{7192e9f8},")
    audit.said(SI, "Revisions \\texttt{d12f8825} and \\texttt{e5883055} are not in the repository; their records are.")

    pri, tem, res, uni = (cells_of[k] for k in ("priors", "templates", "restart", "pair-scale"))

    def fall(c: dict) -> tuple[int, int]:
        both, only_with, only_without, _ = c["table"]
        return both + only_with, both + only_without

    alloys = ("cu6au32", "cu19au19", "cu27au28", "cu10au45", "cu30au8", "pt12cu11")
    lowered = [a for a in alloys if fall(pri[a])[1] < fall(pri[a])[0] and pri[a]["p"] < 0.05]
    audit.cell(TEX, "alloys whose hits fall at p<0.05 without the priors", " ".join(alloys), " ".join(lowered))
    audit.cell(TEX, "every lowered alloy hits 12 of 12 with the priors", "12", float(min(fall(pri[a])[0] for a in alloys)))
    audit.cell(TEX, "lowest and highest hits without the priors", "0 5",
               f"{min(fall(pri[a])[1] for a in alloys)} {max(fall(pri[a])[1] for a in alloys)}")
    audit.cell(TEX, "largest alloy p", "0.016", max(pri[a]["p"] for a in alloys))
    audit.cell(TEX, "Cu12Au26 is hit by neither arm", "0 0", " ".join(map(str, fall(pri["cu12au26"]))))
    p_ = {s: pp(c["p"]) for s, c in pri.items()}
    audit.said(TEX, f"Cu\\(_6\\)Au\\(_{{32}}\\) and Cu\\(_{{19}}\\)Au\\(_{{19}}\\) fall from 12 of 12 to "
                    f"{fall(pri['cu6au32'])[1]} of 12 (\\(p={p_['cu6au32']}\\)), Cu\\(_{{27}}\\)Au\\(_{{28}}\\) and "
                    f"Pt\\(_{{12}}\\)Cu\\(_{{11}}\\) to {fall(pri['cu27au28'])[1]} of 12 (\\(p={p_['cu27au28']}\\)), "
                    f"Cu\\(_{{10}}\\)Au\\(_{{45}}\\) to {fall(pri['cu10au45'])[1]} of 12 (\\(p={p_['cu10au45']}\\)), "
                    f"and Cu\\(_{{30}}\\)Au\\(_8\\) to {fall(pri['cu30au8'])[1]} of 12 (\\(p={p_['cu30au8']}\\)); "
                    "neither arm hits Cu\\(_{12}\\)Au\\(_{26}\\).")
    audit.cell(TEX, "Cu6Au32 and Cu19Au19 agree", p_["cu6au32"], p_["cu19au19"])
    audit.cell(TEX, "Cu27Au28 and Pt12Cu11 agree", p_["cu27au28"], p_["pt12cu11"])
    audit.said(TEX, f"On Au\\(_{{55}}\\) the hits fall from {fall(pri['au55'])[0]} to {fall(pri['au55'])[1]} of 12 "
                    f"(\\(p={p_['au55']}\\)), and on LJ\\(_{{150}}\\) they rise from {fall(pri['lj150'])[0]} to "
                    f"{fall(pri['lj150'])[1]} of 12, with \\(p={p_['lj150']}\\).")
    audit.said(TEX, f"to between 0 and 5 of 12 (\\(p\\le {pp(max(pri[a]['p'] for a in alloys))}\\)); on "
                    f"LJ\\(_{{150}}\\) the hits rise from {fall(pri['lj150'])[0]} to {fall(pri['lj150'])[1]} of 12 "
                    f"(\\(p={p_['lj150']}\\)).")
    audit.said(TEX, "With the switch built on develop \\texttt{f2a628c0}, turning the structure priors off lowers the "
                    "hits on six of the seven Gupta alloys.")

    m38 = tem["m38r14"]
    lo, hi = m38["interval"]
    audit.said(TEX, f"the hits on M\\(_{{38}}\\) at \\(\\rho=14\\) fall from {fall(m38)[0]} to {fall(m38)[1]} of 12 "
                    f"(\\(p={pp(m38['p'])}\\), interval {lo:.2f} to {hi:.2f}), and on the Sutton--Chen 10--8 cluster of "
                    f"38 atoms from {fall(tem['sc108-38'])[0]} to {fall(tem['sc108-38'])[1]} of 12 "
                    f"(\\(p={pp(tem['sc108-38']['p'])}\\)).")
    audit.cell(TEX, "both revisions hit M38 at rho 6 and SC 12-6 on every seed", "12 12 12 12",
               " ".join(str(x) for s in ("m38r6", "sc126-38") for x in fall(tem[s])))
    audit.said(TEX, f"Removing the structure templates lowers the hits on Morse M\\(_{{38}}\\) at \\(\\rho=14\\) from "
                    f"{fall(m38)[0]} to {fall(m38)[1]} of 12 (\\(p={pp(m38['p'])}\\)).")

    audit.cell(TEX, "the restart block changes no cluster at p<0.05", "none",
               " ".join(s for s, c in res.items() if c["p"] < 0.05) or "none")
    audit.cell(TEX, "clusters of the restart block", "12", float(len(set(res) - {"lj75"})))
    audit.cell(TEX, "restart seeds per cluster", "12 24", " ".join(sorted({str(c["n"]) for c in res.values()})))
    audit.said(TEX, "\\(p<0.05\\) on the twelve clusters of that block of Table~\\ref{tab:knockouts}, with 12 or 24 seeds "
                    "each.")
    gaps = sorted((abs(fall(c)[0] - fall(c)[1]), s) for s, c in res.items() if s != "lj75")[::-1][:3]
    audit.cell(TEX, "the three largest restart differences", "lj150 lj98 m55r14", " ".join(sorted(s for _, s in gaps)))
    audit.said(TEX, f"The largest differences are on LJ\\(_{{150}}\\), {fall(res['lj150'])[0]} against "
                    f"{fall(res['lj150'])[1]} of 24 (\\(p={pp(res['lj150']['p'])}\\)), on LJ\\(_{{98}}\\), "
                    f"{fall(res['lj98'])[0]} against {fall(res['lj98'])[1]} of 24 (\\(p={pp(res['lj98']['p'])}\\)), and "
                    f"on M\\(_{{55}}\\) at \\(\\rho=14\\), {fall(res['m55r14'])[0]} against {fall(res['m55r14'])[1]} of "
                    f"24 (\\(p={pp(res['m55r14']['p'])}\\)).")
    audit.said(TEX, "The restart is off in the preset.")
    config = audit.text("src/methods/cluster_hopping/config.rs")
    audit.record("restart_on_plateau" not in config or "restart_on_plateau: false" in config, PRIOR_CONFIG,
                 "the preset has no plateau restart on", "off", "checked")

    m80 = uni["m80r6"]
    lo, hi = m80["interval"]
    audit.said(TEX, f"raises the hits on M\\(_{{80}}\\) at \\(\\rho=6\\) from {fall(m80)[1]} to {fall(m80)[0]} of 12 "
                    f"(\\(p={pp(m80['p'])}\\), interval {lo:.2f} to {hi:.2f}).")
    same = [s for s, c in uni.items() if s not in ("m80r6", "lj75") and c["table"][1] == c["table"][2] == 0]
    audit.cell(TEX, "the other clusters of the length-unit block hit the same seeds", "11", float(len(same)))
    audit.cell(TEX, "Lennard-Jones, gold, and Morse among them", "7 1 3",
               f"{sum(s.startswith('lj') for s in same)} {sum(s == 'au55' for s in same)} "
               f"{sum(s.startswith('m') for s in same)}")
    audit.said(TEX, "The other eleven clusters of that block of Table~\\ref{tab:knockouts}, seven Lennard-Jones sizes, "
                    "Au\\(_{55}\\), and three Morse clusters, hit the same seeds in both arms.")
    audit.said(TEX, f"at \\(\\rho=6\\) from {fall(m80)[1]} to {fall(m80)[0]} of 12 (\\(p={pp(m80['p'])}\\)) and leaves "
                    "the hits on the eleven other clusters reported for it unchanged.")
    audit.said(TEX, "A restart from a diverse population on a plateau and a Thompson choice among the moves change no "
                    "cluster at \\(p<0.05\\).")
    audit.said(COVER, "Switching off the structure priors lowers the hits on six Cu--Au and Pt--Cu Gupta alloys, "
                      "removing the structure templates lowers them on a Morse cluster, and measuring lengths in units "
                      "of the pair minimum raises them on another; a restart on a plateau and a Thompson choice among "
                      "the moves change no cluster significantly.")

    # TIP4P against GMIN, paired by seed and start.
    path = imported(audit, "paper/jctc/data/tip4p_gmin")
    rows = audit.rows(path)
    arms: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for r in rows:
        arms[(int(r["molecules"]), r["arm"])][r["seed"]] = r
    revisions = {r["anneal_sha"] for r in rows}
    audit.cell(path, "TIP4P rows of this code are revision e1a69a57", "e1a69a57a8dda1b44fdde9413ae31e3fc3af1889",
               revisions.pop() if len(revisions) == 1 else " ".join(sorted(revisions)))
    ancestor_of(audit, path, "e1a69a57a8dda1b44fdde9413ae31e3fc3af1889", SUBMITTED)
    audit.cell(path, "TIP4P rows per arm and size", "12", float(min(len(v) for v in arms.values())))
    shared = all(arms[(n, "anneal")][k]["start_id"] == arms[(n, "gmin")][k]["start_id"]
                 for n in {n for n, _ in arms} for k in arms[(n, "anneal")])
    audit.record(shared, path, "both TIP4P arms of a seed share the start", "shared", "shared" if shared else "not")
    sizes = {"(H\\(_2\\)O)\\(_{8}\\)": 8, "(H\\(_2\\)O)\\(_{15}\\)": 15, "(H\\(_2\\)O)\\(_{20}\\)": 20,
             "(H\\(_2\\)O)\\(_{21}\\)": 21, "(H\\(_2\\)O)\\(_{25}\\)": 25}
    stats = {}
    for line in tex_table(audit, TEX, "tab:tip4p-gmin"):
        n = sizes.get(line[0])
        where = f"{TEX} tab:tip4p-gmin {line[0]}"
        if n is None:
            audit.record(False, where, "a TIP4P size the rows hold", line[0], "unknown")
            continue
        a, g = arms[(n, "anneal")], arms[(n, "gmin")]
        seeds = sorted(a, key=int)
        if n == 25:
            text = " & ".join(line[1:])
            for arm, label in ((a, "this work"), (g, "GMIN")):
                energies = [float(arm[k]["best_energy"]) for k in seeds]
                audit.record(f"{label} \\({min(energies):.3f}\\), median \\({median(energies):.3f}\\)" in text, where,
                             f"{label} lowest and median best energy", f"{min(energies):.3f} {median(energies):.3f}",
                             text[:80])
            continue
        table = paired_table({k: a[k]["hit"] == "1" for k in seeds}, {k: g[k]["hit"] == "1" for k in seeds})
        calls = wilcoxon_signed_rank([float(a[k]["calls"]) - float(g[k]["calls"]) for k in seeds])
        walls = wilcoxon_signed_rank([float(a[k]["wall_s"]) - float(g[k]["wall_s"]) for k in seeds])
        stats[n] = (table, mcnemar(table[1], table[2]))
        lo, hi = newcombe_paired_interval(*table)
        audit.cell(where, "this work", line[1], f"{table[0] + table[1]}/12")
        audit.cell(where, "GMIN", line[2], f"{table[0] + table[2]}/12")
        audit.cell(where, "only this work", line[3], float(table[1]))
        audit.cell(where, "only GMIN", line[4], float(table[2]))
        audit.cell(where, "exact McNemar p", line[5], stats[n][1])
        audit.cell(where, "Newcombe lower", interval_of(line[6])[0], lo)
        audit.cell(where, "Newcombe upper", interval_of(line[6])[1], hi)
        audit.cell(where, "Wilcoxon p on calls", line[7], calls[2])
        audit.cell(where, "Wilcoxon p on wall", line[8], walls[2])
    audit.cell(TEX, "tab:tip4p-gmin sizes", "8 15 20 21", " ".join(map(str, sorted(stats))))
    t = {n: stats[n][0] for n in stats}
    audit.said(TEX, f"Paired by seed, this code alone hits {t[15][1]} seeds of (H\\(_2\\)O)\\(_{{15}}\\) and {t[20][1]} of "
                    f"(H\\(_2\\)O)\\(_{{20}}\\) and GMIN alone none, with \\(p={pp(stats[15][1])}\\) and "
                    f"{pp(stats[20][1])}.")
    audit.cell(TEX, "GMIN alone hits no seed of 15 or 20", "0 0", f"{t[15][2]} {t[20][2]}")
    audit.said(TEX, f"On (H\\(_2\\)O)\\(_8\\) this code hits {t[8][0] + t[8][1]} of 12 and GMIN {t[8][0] + t[8][2]}, "
                    f"with \\(p={pp(stats[8][1])}\\), and neither arm hits (H\\(_2\\)O)\\(_{{21}}\\).")
    audit.cell(TEX, "neither arm hits 21", "0 0 0", " ".join(map(str, t[21][:3])))
    a25 = [float(r["best_energy"]) for r in arms[(25, "anneal")].values()]
    g25 = [float(r["best_energy"]) for r in arms[(25, "gmin")].values()]
    audit.said(TEX, f"lowest energy of this code there is \\({min(a25):.3f}\\)~kJ/mol and that of GMIN "
                    f"\\({min(g25):.3f}\\)~kJ/mol.")
    audit.said(TEX, f"(H\\(_2\\)O)\\(_{{20}}\\) on {t[15][0] + t[15][1]} and {t[20][0] + t[20][1]} of 12 against none "
                    f"(\\(p={pp(stats[15][1])}\\) and {pp(stats[20][1])}).")
    two = [n for n in stats if t[n][1] > t[n][2] and stats[n][1] < 0.05]
    audit.cell(COVER, "TIP4P sizes where the code hits more than GMIN at p<0.05", "15 20", " ".join(map(str, two)))
    audit.said(TEX, "Table~\\ref{tab:tip4p-gmin} puts revision \\texttt{e1a69a57} of this code against GMIN on rigid "
                    "TIP4P clusters of 8, 15, 20, 21, and 25 molecules, at \\(2\\times 10^{6}\\) calls on seeds "
                    "2000--2011, both arms from the same start on each seed in alternating order.")
    orders = {r["arm_order"] for r in rows}
    audit.cell(path, "TIP4P arm orders", "anneal,gmin gmin,anneal", " ".join(sorted(orders)))


def manuscript_priors(audit: Audit) -> None:
    """Section sec:priors read against the source of the removal commit, its parent, and the current preset."""
    removal, lj_rev = "1c1b1238", "c2de652dca0bc75e2b0cb4afffac91fd656404e4"
    subject = git("log", "-1", "--format=%s", removal).stdout.strip()
    audit.cell(TEX, "the removal commit", "fix: remove structure templates from cluster search", subject)
    removed = git("show", removal).stdout
    for switch in ("prior_reseed", "prior_growth", "prior_core", "prior_handover", "prior_motif"):
        audit.record(f"-            && cfg.{switch}" in removed or f"cfg.{switch}" in "".join(
            line for line in removed.splitlines() if line.startswith("-")), TEX,
            f"{removal} removes the {switch} code path", "removed", "removed")
    on = priors_on(f"{removal}^") or []
    audit.cell(TEX, "the six priors are on in the parent of the removal", " ".join(PRIORS), " ".join(on))
    audit.said(TEX, "Up to commit \\texttt{1c1b1238} the preset carried six switches that use the structure of "
                    "close-packed clusters, all on.")
    config = git("show", f"{removal}^:{PRIOR_CONFIG}").stdout
    audit.record("reseed_hops: 5," in config, TEX, "the reseed interval is five hops before the removal", "5",
                 "5" if "reseed_hops: 5," in config else "other")
    audit.said(TEX, "at most once every five hops, closed its core orbits, and adopted the quench as the walker.")
    current = audit.text(PRIOR_CONFIG)
    audit.record("prior_soap" in current, PRIOR_CONFIG, "prior_soap remains in the code", "present",
                 "present" if "prior_soap" in current else "absent")
    audit.said(TEX, "positions, and it remains in the code.")
    switch = "4f5ec0c56779e02b7627b7d3693fbb319f631176"
    added = git("show", switch, "--", PRIOR_CONFIG).stdout
    audit.record("+    pub structural_priors: bool," in added, PRIOR_CONFIG, f"{switch[:8]} adds the structural_priors switch",
                 "added", "added" if "+    pub structural_priors: bool," in added else "absent")
    audit.record(git("merge-base", "--is-ancestor", switch, SUBMITTED).returncode != 0, PRIOR_CONFIG,
                 "the switch is not in the history of the submitted revision", "not", "checked")
    readme = git("show", "refs/remotes/origin/record/review6-on-7192:bench/priors-ablation/README.md").stdout
    audit.record("Develop `f2a628c0` and this series on it give identical pins" in readme, "record/review6-on-7192",
                 "with the switch on, the build gives the pins of develop f2a628c0", "identical pins", "checked")
    audit.said(TEX, "The knockout of Section~\\ref{sec:k-priors} builds one switch, \\texttt{Config::structural\\_priors}, "
                    "on develop \\texttt{f2a628c0}; it turns off every path that builds a structure from a known packing "
                    "or chooses a move by classifying one, \\texttt{prior\\_soap} included, and with the switch on that "
                    "build gives the trajectory pins of develop \\texttt{f2a628c0}.")
    audit.said(SI, "whose binaries are the switch of commit \\texttt{4f5ec0c5} built on develop \\texttt{f2a628c0};")
    audit.said(TEX, "Commit \\texttt{1c1b1238} removed the structure templates, the builders of face-centered cubic, "
                    "polyicosahedral, Mackay, and axial-spine cuts behind the last five priors;")
    table_priors(audit, removal)
    count_gates(audit, lj_rev)


def count_gates(audit: Audit, lj_rev: str) -> None:
    """The atom- and body-count gates the methods name, read from the current source and from c2de652d."""
    config = audit.text(PRIOR_CONFIG)
    tests = re.findall(r"if n_points\s*(<|<=|>|>=|==)\s*(\d+)\s*\{([^}]*)\}", config)
    shown = " ".join(f"{op} {n} {'surfaces' if 'cfg.surfaces = ' in body else 'other'}" for op, n, body in tests)
    audit.cell(PRIOR_CONFIG, "the atom-count tests of the presets", "< 98 surfaces < 98 surfaces", shown)
    audit.cell(PRIOR_CONFIG, "every n_points test is one of those", str(len(tests)),
               float(len(re.findall(r"(?<!self\.)\bn_points\s*(?:<|<=|>|>=|==)\s*\d", config))))
    audit.record("self.n_points >= 2," in config, PRIOR_CONFIG, "the other n_points test is the validation of the size",
                 "present", "present" if "self.n_points >= 2," in config else "absent")
    hop = audit.text("src/methods/cluster_hopping.rs")
    audit.record("n_points < 98" not in hop, "src/methods/cluster_hopping.rs", "the growth hold gate is gone",
                 "absent", "absent" if "n_points < 98" not in hop else "present")
    old = git("show", f"{lj_rev}:src/methods/cluster_hopping.rs").stdout
    audit.record("stall_growth_waits(held, incumbent) && (n_points < 98 || skipped < GROWTH_RETRY_LEAVES)" in old,
                 f"{lj_rev[:8]}:src/methods/cluster_hopping.rs", "the prior-on growth hold releases at 98 atoms",
                 "present", "present")
    bodies = re.search(r"const RIGID_HOT_BODIES: usize = (\d+);", hop)
    audit.cell("src/methods/cluster_hopping.rs", "rigid body-count gate", "10", float(bodies.group(1)) if bodies else None)
    for fragment in ("cfg.move_library.is_rigid_body() && cfg.n_points > RIGID_HOT_BODIES",
                     "&& cfg.n_points <= RIGID_HOT_BODIES"):
        audit.record(fragment in hop, "src/methods/cluster_hopping.rs", "the rigid gate reads the body count",
                     fragment, "present" if fragment in hop else "absent")
    assignment = audit.text("src/assignment.rs")
    audit.record("const SHORTEST_PATH_FROM: usize = 98;" in assignment and "the same optimum" in assignment,
                 "src/assignment.rs", "the assignment switches algorithm at 98 for the same optimum", "98", "98")
    audit.record("minimum_cost_assignment" in audit.text("src/continuous_symmetry.rs"), "src/continuous_symmetry.rs",
                 "the symmetry projections use that assignment", "used", "used")
    audit.said(TEX, "For atomic clusters the preset has one atom-count gate: the diameter penalty of "
                    "equation~\\ref{eq:penalty} is used only below 98 atoms.")
    audit.said(TEX, "The revisions with the structure priors on had a second, the growth hold of "
                    "equation~\\ref{eq:hold}.")
    audit.said(TEX, "A rigid-body run has a gate of its own: at 10 molecules or fewer its step factor may climb higher, "
                    "and above 10 its orientation step follows the acceptance rate.")
    audit.said(TEX, "In the revisions with the structure priors on, lattice growth on a plateau is held when it has "
                    "already failed at the current energy.")
    for phrase in ("two atom-count gates", "Apart from those gates no move reads the atom count"):
        audit.record(phrase not in collapse(audit.text(TEX)), TEX, f"retired gate wording stays out: {phrase}",
                     "absent", "absent" if phrase not in collapse(audit.text(TEX)) else "present")


def table_priors(audit: Audit, removal: str) -> None:
    """The prior state behind each table that the SI gives, read from the source of each revision."""
    for revision, want in (("9a30f319", " ".join(PRIORS)), ("70c0ad6c^", " ".join(PRIORS)),
                           ("3129d1ec^", "soap reseed growth core motif"), ("07092af2^", "")):
        audit.cell(SI, f"priors on in the preset at {revision}", want or "none", " ".join(priors_on(revision) or []) or "none")
        ancestor = git("merge-base", "--is-ancestor", removal, revision).returncode == 0
        audit.record(not ancestor, SI, f"{revision} predates the removal", "not an ancestor",
                     "ancestor" if ancestor else "not an ancestor")
    for revision, absent in (("3129d1ec^", "prior_handover"), ("07092af2^", "prior_")):
        audit.record(absent not in git("show", f"{revision}:{PRIOR_CONFIG}").stdout, SI,
                     f"{revision} has no {absent} switch", "absent", "absent")
    hop = git("show", "3129d1ec^:src/methods/cluster_hopping.rs").stdout
    for switch in ("prior_core", "prior_motif"):
        at = hop.index(f"&& cfg.{switch}")
        guard = hop[hop.rindex("if ", 0, at):hop.index("{", at)]
        audit.record("!cfg.move_library.is_rigid_body()" in guard, SI, f"3129d1ec^ keeps rigid runs out of {switch}",
                     "guarded", "guarded" if "!cfg.move_library.is_rigid_body()" in guard else "unguarded")
    audit.said(SI, "The parent of \\texttt{07092af2} has no prior switch.")


PRIORS = ("soap", "reseed", "growth", "core", "handover", "motif")
PRIOR_CONFIG = "src/methods/cluster_hopping/config.rs"


def priors_on(revision: str) -> list[str] | None:
    """The structure priors the preset at ``revision`` sets on, or None when the revision is not here."""
    shown = git("show", f"{revision}:{PRIOR_CONFIG}")
    if shown.returncode != 0:
        return None
    return [p for p in PRIORS if f"prior_{p}: true," in shown.stdout]


def configuration_of(rows: list[dict[str, str]]) -> str:
    """The name the manuscript gives the configuration of this code behind ``rows``.

    A revision in the repository is prior-on when its preset sets all six priors on;
    one that is not is named by its entry in the list of unreachable revisions."""
    revisions = {r["code_revision"] for r in rows}
    if len(revisions) != 1:
        return "mixed revisions"
    revision = revisions.pop()
    on = priors_on(revision)
    if on is not None:
        return "prior-on" if len(on) == len(PRIORS) else f"{len(on)} priors on"
    return "template-free" if "template-free" in UNPUSHED.get(revision, "") else "unknown"


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
    audit.said(TEX, "The gold driver of this code gives the template-free configuration the copper--gold species "
                    "list with no copper atom and applies the basin bias of equation~\\ref{eq:delta} on a stride of "
                    f"{stride.group(1) if stride else '?'}; it does not touch the GMIN arm.")
    audit.said(TEX, f"On Au\\(_{{55}}\\) the gold driver of this code applies the basin bias of equation~\\ref{{eq:delta}} "
                    f"to this code on a stride of {stride.group(1) if stride else '?'}.")
    audit.said(SI, "The gold driver \\path{examples/h2h_metal.rs} configures only the arm of this code.")
    strides = re.findall(r"(\S*)basin_stride\s*=", metal)
    audit.cell("examples/h2h_metal.rs", "the basin stride is set only on the config of this code", "cfg.",
               " ".join(sorted(set(strides))))
    audit.record("Config::recommended(spec.n)" in metal and "gmin" not in metal.lower(), "examples/h2h_metal.rs",
                 "the gold driver builds a config of this code and never runs GMIN", "this code only",
                 "this code only" if "gmin" not in metal.lower() else "mentions GMIN")
    gmin_patches(audit)


def gmin_patches(audit: Audit) -> None:
    """The GMIN of the panel carries the cap and timer patches and nothing else of substance."""
    build = audit.text("scripts/gmin_panel/build_gmin.sh")
    listed = re.search(r"PATCHES=\((.*?)\)", build, re.S)
    names = re.findall(r'"\$ROOT/([^"]+)"', listed.group(1)) if listed else []
    want = ["bench/lj104-lattice/v2/h2h-gmin.patch", "scripts/morse_h2h/morse-timer.patch",
            "bench/metal-cluster-h2h/gmin-bgupta.patch"]
    audit.cell("scripts/gmin_panel/build_gmin.sh", "GMIN patches", " ".join(want), " ".join(names))
    audit.said(SI, ", ".join(f"\\path{{{n}}}" for n in want[:2]) + f", and \\path{{{want[2]}}}")
    shown = [f"\\texttt{{{n.replace('_', chr(92) + '_')}}}" for n in want]
    audit.said(TEX, "GMIN is the snapshot of SVNREV \\texttt{9cbec1a4} built with three patches, " + ", ".join(shown[:2]) + f", and {shown[2]}, "
                    "which cap the force calls, time the potential, and, in the serial build, set the run index "
                    "(\\texttt{JP=1}) before its first use.")
    audit.record("JP=1" in audit.text(want[0]), want[0], "the first patch sets the serial run index", "JP=1",
                 "JP=1" if "JP=1" in audit.text(want[0]) else "absent")
    audit.record("patched only" not in audit.text(TEX), TEX, "the GMIN caption names every patch", "absent",
                 "absent" if "patched only" not in audit.text(TEX) else "present")
    allowed = {"#ifndef GMIN_MPI", "JP=1", "#endif", "IF (SAVECT) COMPRESST=.TRUE."}
    # What a capped call does instead of evaluating: return a refused energy and stop the quench.
    refusal = {"RETURN", "MFLAG=.FALSE.", "NPCALL = NPCALL - 1", "EREAL = 1.0D20", "IF (GRADT) GRAD(:) = 0.0D0",
               "IF (ENERGY.GT.1.0D19) ENERGY=1.0D6"}
    for name in names:
        text = audit.text(name)
        removed = {line[1:].strip() for line in text.splitlines() if line.startswith("-") and not line.startswith("---")}
        target = ""
        stray = []
        block: list[str] | None = None
        for line in text.splitlines():
            if line.startswith("+++ "):
                target = line[4:]
                continue
            if not line.startswith("+") or target == "b/source/h2h_cap.f90":
                continue
            body = line[1:].strip()
            if block is not None:
                if body != "END IF":
                    block.append(body)
                    continue
                head, rest = block[0], block[1:]
                guarded = "H2H" in head or any("H2H" in b for b in rest)
                rewritten = head.endswith(" THEN") and head[:-len(" THEN")] + " RETURN" in removed
                plain = [b for b in rest if "H2H" not in b]
                if not (guarded and (rewritten or "H2H" in head) and set(plain) <= refusal):
                    stray.append(" / ".join(block))
                block = None
                continue
            if re.fullmatch(r"IF \(.*\) THEN", body):
                block = [body]
            elif body and "H2H" not in body and body not in allowed:
                stray.append(body)
        audit.cell(name, "added GMIN lines outside the cap and timers", "none", "; ".join(stray) or "none")
    audit.said(SI, "which cap the force calls, time the potential, and, in the serial build, set the run index before "
                   "its first use; no other line of GMIN changes.")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap driver prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")


ASE_PANEL = ("lj38", "lj55", "lj98", "lj104", "au55")
MORSE_PANEL = ("m38r6", "m38r14", "m55r6", "m55r14")
ASE_ARMS = ("ase_bh", "ase_mh")
ASE_DATA = "paper/jctc/data/ase_rerun"
ASE_REVISIONS = {"lj": "61fa67cf380e46876b569da45e7cc091647a775b",
                 "morse": "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"}
PANEL_SEEDS = " ".join(str(s) for s in range(2000, 2012))


def load_script(name: str, path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ase_groups(audit: Audit) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The rows of each system and arm of the two ASE tables, in seed order."""
    groups = {}
    folders = [(s, (s,), ("anneal", "recommended", *ASE_ARMS), "lj") for s in ASE_PANEL]
    folders.append(("morse", MORSE_PANEL, ("for_cluster", "recommended", *ASE_ARMS), "morse"))
    for folder, systems, arms, kind in folders:
        rows = audit.rows(f"{ASE_DATA}/{folder}/rows.csv")
        for system in systems:
            for arm in arms:
                group = sorted(pick(rows, system=system, arm=arm), key=lambda r: int(r["seed"]))
                where = f"ase panel {system} {arm}"
                audit.cell(where, "seeds", " ".join(r["seed"] for r in group), PANEL_SEEDS)
                audit.cell(where, "revision", " ".join(sorted({r["code_revision"] for r in group})),
                           ASE_REVISIONS[kind])
                groups[(system, arm)] = group
    return groups


def ase_cap(audit: Audit, groups: dict, system: str) -> float:
    """The one wall cap of a cluster, checked to be three times the slowest cluster-preset wall."""
    base = "for_cluster" if system in MORSE_PANEL else "anneal"
    caps = sorted({r["cap_s"] for arm in ("recommended", *ASE_ARMS) for r in groups[(system, arm)]})
    slowest = max(float(r["wall_s"]) for r in groups[(system, base)])
    audit.cell(f"ase panel {system}", "one wall cap", "1", float(len(caps)))
    audit.record(abs(float(caps[0]) - 3.0 * slowest) <= 1e-5, f"ase panel {system}", "cap is three times the slowest "
                 f"{base} wall", caps[0], f"{3.0 * slowest:.6f}")
    return float(caps[0])


def manuscript_ase(audit: Audit) -> None:
    """Tables tab:ase and tab:ase-morse, the imported rows they come from, and every sentence that reads them."""
    table = load_script("ase_table_audit", "paper/jctc/scripts/ase_table.py")
    for folder in (*ASE_PANEL, "morse"):
        path = imported(audit, f"{ASE_DATA}/{folder}")
        fields = dict(line.split(" ", 1) for line in audit.text(f"{ASE_DATA}/{folder}/PROVENANCE").splitlines())
        ancestor_of(audit, path, fields["ancestor"])
        audit.cell(path, "every row is the PROVENANCE ancestor", fields["ancestor"],
                   " ".join(sorted({r["code_revision"] for r in audit.rows(path)})))
    audit.record(not (ROOT / ASE_DATA / "lj150").exists(), ASE_DATA, "LJ150 has no imported rows", "absent", "checked")
    groups = ase_groups(audit)

    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    def stops(system: str, arm: str, stop: str = "wall") -> int:
        return sum(r["stop"] == stop for r in groups[(system, arm)])

    def p(system: str, arm: str) -> float:
        a, b = groups[(system, "recommended")], groups[(system, arm)]
        return fisher(hits_of(a), len(a), hits_of(b), len(b))

    budgets = {s: v[1] for s, v in (table.LJ | table.MORSE).items()}
    references = {s: v[2] for s, v in (table.LJ | table.MORSE).items()}
    by_atoms = {38: 400_000, 55: 1_000_000}
    atoms = {"lj38": 38, "lj55": 55, "lj98": 98, "lj104": 104, "au55": 55,
             "m38r6": 38, "m38r14": 38, "m55r6": 55, "m55r14": 55}
    for (system, arm), group in groups.items():
        audit.cell(f"ase panel {system} {arm}", "budget", " ".join(sorted({r["budget"] for r in group})),
                   str(budgets[system]))
        audit.cell(f"ase panel {system} {arm}", "budget by atom count", str(by_atoms.get(atoms[system], 4_000_000)),
                   float(budgets[system]))
        worst = max((abs(float(r["best"]) - references[system]) for r in group if r["hit"] == "1"), default=0.0)
        audit.record(worst < DEFAULT_HIT_TOL, f"ase panel {system} {arm}", "every hit within 1e-3 of its reference",
                     f"{worst:.3g}", "below 0.001")
    for system in (*ASE_PANEL, *MORSE_PANEL):
        audit.cell(f"ase panel {system}", "recommended wall stops", "0", float(stops(system, "recommended")))

    for label, systems, labels in (("tab:ase", ASE_PANEL, table.LJ), ("tab:ase-morse", MORSE_PANEL, table.MORSE)):
        by_label = {v[3]: s for s, v in labels.items()}
        shown = []
        for cells in tex_table(audit, TEX, label):
            system = by_label.get(cells[0])
            where = f"{TEX} {label} {system}"
            if system is None or len(cells) != 13:
                audit.record(False, where, "table line", " & ".join(cells), "a panel system")
                continue
            shown.append(system)
            audit.cell(where, "budget", cells[1], float(budgets[system]))
            audit.cell(where, "cap", cells[2], ase_cap(audit, groups, system))
            for cell, arm in zip(cells[3:6], ("recommended", *ASE_ARMS)):
                audit.cell(where, f"{arm} hits", cell, frac(groups[(system, arm)]))
            for cell, arm in zip(cells[6:8], ASE_ARMS):
                audit.cell(where, f"{arm} wall stops", cell, float(stops(system, arm)))
            for cell, arm in zip(cells[8:11], ("recommended", *ASE_ARMS)):
                audit.cell(where, f"{arm} median wall", cell, median([float(r["wall_s"]) for r in groups[(system, arm)]]))
            for cell, arm in zip(cells[11:13], ASE_ARMS):
                audit.cell(where, f"Fisher p against {arm}", cell, p(system, arm))
        audit.cell(TEX, f"{label} lines", " ".join(shown), " ".join(systems))

    lj_rev, morse_rev = ASE_REVISIONS["lj"][:8], ASE_REVISIONS["morse"][:8]
    audit.said(TEX, f"\\caption{{This code, \\texttt{{Config::recommended}} at revision \\texttt{{{lj_rev}}}, against "
                    "the basin hopping (BH) and minima hopping (MH) of ASE 3.29.0 at its constructor defaults, at the "
                    "effort each run reports.")
    audit.said(TEX, f"Table~\\ref{{tab:ase}} runs revision \\texttt{{{lj_rev}}} on four Lennard-Jones clusters and "
                    f"Au\\(_{{55}}\\), and Table~\\ref{{tab:ase-morse}} runs revision \\texttt{{{morse_rev}}} on four "
                    "Morse clusters; both are ancestors of the submitted revision.")
    audit.said(TEX, "These runs do not share a start with this code, so their hits are compared by Fisher's test, and "
                    "their effort is not equal: every run stopped at a hit, at the budget, or at a wall cap of three "
                    "times the slowest wall of the cluster preset of this code on that cluster.")
    mh_all = all(r["stop"] == "wall" for s in (*ASE_PANEL, *MORSE_PANEL) for r in groups[(s, "ase_mh")])
    audit.record(mh_all, TEX, "every ASE minima-hopping run ended at the cap", "every", "every" if mh_all else "not")
    audit.said(TEX, "ASE minima hopping is wall-capped: every one of its runs in both tables ended at the cap with "
                    "fewer calls than the budget.")
    bh_lj = sum(stops(s, "ase_bh") for s in ASE_PANEL)
    runs_lj = sum(len(groups[(s, "ase_bh")]) for s in ASE_PANEL)
    bh_morse = all(r["stop"] == "wall" for s in MORSE_PANEL for r in groups[(s, "ase_bh")])
    audit.record(bh_morse, TEX, "basin hopping reached the cap on every Morse run", "every", "every" if bh_morse else "not")
    audit.said(TEX, f"Basin hopping reached the cap on {bh_lj} of the {runs_lj} runs of Table~\\ref{{tab:ase}} and on "
                    "every run of Table~\\ref{tab:ase-morse}, and no run of this code reached it.")
    r = {s: hits(s, "recommended") for s in (*ASE_PANEL, *MORSE_PANEL)}
    audit.said(TEX, f"Under that policy this code hits LJ\\(_{{38}}\\) and LJ\\(_{{55}}\\) on {r['lj38']} of 12, "
                    f"LJ\\(_{{98}}\\) on {r['lj98']} of 12, and LJ\\(_{{104}}\\) on {r['lj104']} of 12, while ASE basin "
                    f"hopping hits LJ\\(_{{55}}\\) on {hits('lj55', 'ase_bh')} of 12 and no other cluster, and ASE "
                    "minima hopping hits none; no arm hits Au\\(_{55}\\).")
    audit.cell(TEX, "LJ38 and LJ55 share one count", "12 12", f"{r['lj38']} {r['lj55']}")
    bh_hits = [s for s in ASE_PANEL if hits(s, "ase_bh")]
    audit.cell(TEX, "basin hopping hits one cluster", "lj55", " ".join(bh_hits))
    audit.cell(TEX, "minima hopping hits none", "0", float(sum(hits(s, "ase_mh") for s in ASE_PANEL)))
    audit.cell(TEX, "no arm hits Au55", "0", float(sum(hits_of(groups[("au55", a)]) for a in
                                                       ("anneal", "recommended", *ASE_ARMS))))
    audit.cell(TEX, "Morse at rho=6 on 12 of 12", "12 12", f"{r['m38r6']} {r['m55r6']}")
    morse_ase = [r_ for s in MORSE_PANEL for a in ASE_ARMS for r_ in groups[(s, a)]]
    audit.cell(TEX, "Morse ASE hits", "0", float(hits_of(morse_ase)))
    charged = [int(r_["charged"]) for r_ in morse_ase]
    audit.said(TEX, f"On the Morse clusters this code hits M\\(_{{38}}\\) and M\\(_{{55}}\\) at \\(\\rho=6\\) on "
                    f"{r['m38r6']} of 12 and at \\(\\rho=14\\) on {r['m38r14']} and {r['m55r14']} of 12, and neither "
                    f"ASE arm hits any; each ASE run there stopped at the cap after {min(charged)} to {max(charged)} "
                    "force calls.")

    # The Supporting Information.
    audit.said(SI, "on seeds 2000--2011, at \\(4\\times 10^{5}\\) calls for 38 atoms, \\(1\\times 10^{6}\\) for 55, "
                    "and \\(4\\times 10^{6}\\) for 98 atoms and above, on one core of one host.")
    audit.said(SI, "Environment 3.29.0 at its constructor defaults")
    audit.said(SI, "\\texttt{\\seqsplit{" + ASE_REVISIONS["lj"] + "}}, and those of the Morse ASE table revision "
                   "\\texttt{\\seqsplit{" + ASE_REVISIONS["morse"] + "}}, at the length scale of the Morse pair; both "
                   "are ancestors of the submitted revision.")
    preset = [str(hits(s, "anneal")) for s in ASE_PANEL]
    morse_preset = [str(hits(s, "for_cluster")) for s in MORSE_PANEL]
    audit.said(SI, f"Its hits are {series(preset)} of 12 on LJ\\(_{{38}}\\), LJ\\(_{{55}}\\), LJ\\(_{{98}}\\), "
                   f"LJ\\(_{{104}}\\), and Au\\(_{{55}}\\), and {series(morse_preset)} of 12 on M\\(_{{38}}\\) and "
                   "M\\(_{55}\\) at \\(\\rho=6\\) and 14 in the order of the Morse table.")
    below = [s for s in ASE_PANEL[:4] if max(p(s, a) for a in ASE_ARMS) < 0.05]
    audit.said(SI, f"Fisher's test against each ASE arm is below 0.05 on all {len(below)} Lennard-Jones clusters.")
    audit.cell(SI, "Fisher below 0.05 on every Lennard-Jones cluster", " ".join(ASE_PANEL[:4]), " ".join(below))
    bh_all = [s for s in (*ASE_PANEL, *MORSE_PANEL) if stops(s, "ase_bh") == 12]
    audit.cell(SI, "basin hopping at the cap on every run", "lj38 au55 m38r6 m38r14 m55r6 m55r14", " ".join(bh_all))
    audit.said(SI, f"Minima hopping reached its cap on every run of both tables, and basin hopping on every run of "
                   f"LJ\\(_{{38}}\\), Au\\(_{{55}}\\), and the Morse clusters and on {stops('lj55', 'ase_bh')} of 12 on "
                   "LJ\\(_{55}\\).")
    spent = [s for s in ASE_PANEL if stops(s, "ase_bh", "budget") == 12]
    audit.cell(SI, "basin hopping spent the budget on every seed", "lj98 lj104", " ".join(spent))
    audit.said(SI, "On LJ\\(_{98}\\) and LJ\\(_{104}\\) basin hopping spent the full \\(4\\times 10^{6}\\) calls on "
                   "every seed and missed on every seed.")
    gaps = [f"{median([float(r_['best']) for r_ in groups[('au55', a)]]) - references['au55']:.3f}"
            for a in ("recommended", *ASE_ARMS)]
    audit.said(SI, f"On Au\\(_{{55}}\\) the median best energy lies {gaps[0]}~eV above the reference for this code, "
                   f"{gaps[1]}~eV for basin hopping, and {gaps[2]}~eV for minima hopping.")
    audit.said(SI, "No run of this code reached its cap.")
    audit.said(SI, "the random generators of the two codes do not make a shared start from one seed")
    audit.said(SI, "\\path{paper/jctc/scripts/ase_table.py} writes both ASE tables from them.")
    audit.said(SI, "# ASE: rewrite both tables from the imported rows\npython3 paper/jctc/scripts/ase_table.py")

    # Fisher only where arms do not pair, and no rank test.
    text = audit.text(TEX)
    section = text.split("\\label{sec:ase}", 1)[1].split("% ase-nonlj table: end", 1)[0]
    audit.record("Mann" not in section, TEX, "sec:ase reports Fisher only", "no rank test",
                 "no rank test" if "Mann" not in section else "a rank test")
    conclusions = text.split("\\section{Conclusions}", 1)[1].split("\\appendix", 1)[0]
    audit.record("Mann" not in conclusions, TEX, "the conclusions report no rank test", "no rank test",
                 "no rank test" if "Mann" not in conclusions else "a rank test")
    audit.said(TEX, "Arms that do not share a start, the ASE arms, are compared on hits by the two-sided Fisher exact "
                    "test \\cite{fisherLogicInductiveInference1935}.")
    audit.said(TEX, "the test is unpaired because the random generators of the two codes do not make a shared start "
                    "from one seed.")
    stats = text.split("\\label{sec:stats}", 1)[1].split("\\section{", 1)[0]
    outside = text.replace(section, "").replace(stats, "")
    audit.record("Fisher" not in outside.split("\\begin{suppinfo}")[0], TEX,
                 "the Fisher test appears only in the unpaired ASE comparisons", "only there",
                 "only there" if "Fisher" not in outside.split("\\begin{suppinfo}")[0] else "elsewhere too")
    for path in ("scripts/ase_panel/summarize.py", "scripts/ase_nonlj/summarize.py"):
        source = audit.text(path)
        audit.record("mann_whitney" not in source and "continuity" not in source, path,
                     "no continuity-corrected rank test", "absent",
                     "absent" if "mann_whitney" not in source else "present")
    for gone in ("gofee", "AGOX", "1ce3ab64", "b49341e3", "equal-call"):
        found = [path for path in (TEX, SI, "paper/jctc/cover-letter.tex") if gone in audit.text(path)]
        audit.record(not found, " ".join(found) or TEX, f"no {gone} in the paper", "absent", " ".join(found) or "absent")
    step = 1e-4

    def curvature(pair) -> float:
        return (pair(1.0 + step) - 2.0 * pair(1.0) + pair(1.0 - step)) / step**2

    morse6 = curvature(lambda x: morse_pair(6.0, x))
    lennard_jones = curvature(lambda x: x**-12 - 2.0 * x**-6)
    audit.record(abs(morse6 - lennard_jones) < 1e-3, TEX, "Morse rho=6 curvature at the pair minimum",
                 f"{lennard_jones:.3f}", f"{morse6:.3f}")
    gmin_panel, gmin = gmin_panel_groups(audit)
    lj75 = gmin[("lj75", "anneal")], gmin[("lj75", "gmin")]
    audit.said("paper/jctc/NUMBERS_TODO.md", f"In tab:gmin, seeds 2000-2011, the recommended preset hits "
                                             f"{hits_of(lj75[0])} of {len(lj75[0])}.")
    audit.said("paper/jctc/NUMBERS_TODO.md", f"And the Wales-group basin-hopping program (GMIN) hits {hits_of(lj75[1])} "
                                             f"of {len(lj75[1])}.")


GMIN_RERUN = "paper/jctc/data/gmin_rerun"
GMIN_SHARDS = ("a", "b", "c", "d")


def panel_references(audit: Audit) -> None:
    """The panel references of Section sec:hit, and every panel hit inside the window of equation eq:abs."""
    gmin = load_script("gmin_panel_refs", "scripts/gmin_panel/run_panel.py")
    ase = load_script("ase_table_refs", "paper/jctc/scripts/ase_table.py")
    refs: dict[str, set[float]] = defaultdict(set)
    for row in gmin.SYSTEMS:
        refs[row[0]].add(row[3])
    for system, spec in (ase.LJ | ase.MORSE).items():
        refs[system].add(spec[2])
    for system, values in sorted(refs.items()):
        audit.cell("panel references", f"{system} one reference", "1", float(len(values)))
    ref = {s: next(iter(v)) for s, v in refs.items()}
    for path, rows in (("bench/gmin-panel/rows.csv", audit.rows("bench/gmin-panel/rows.csv")),
                       *((f"{ASE_DATA}/{s}/rows.csv", audit.rows(f"{ASE_DATA}/{s}/rows.csv"))
                         for s in (*ASE_PANEL, "morse")),
                       *((f"{GMIN_RERUN}/{shard}/rows.csv", audit.rows(f"{GMIN_RERUN}/{shard}/rows.csv"))
                         for shard in GMIN_SHARDS)):
        worst = max((abs(float(r["best"]) - ref[r["system"]]) for r in rows if hits_of([r])), default=0.0)
        audit.record(
            worst < DEFAULT_HIT_TOL,
            path,
            "every hit within 1e-3 of its reference",
            f"{worst:.3g}",
            "below 0.001",
        )

    def f6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    audit.said(TEX, f"The other tables add {f6('lj102')} at 102 atoms, "
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

    gmin = load_script("gmin_table_tex", "paper/jctc/scripts/gmin_table.py")
    knock = load_script("knockout_tables_tex", "paper/jctc/scripts/knockout_tables.py")
    ase = load_script("ase_table_tex", "paper/jctc/scripts/ase_table.py")
    ablation = load_script("ablation_tex", "paper/jctc/scripts/ablation_table.py")
    for marks, path in ((("% gofee table: start", "% gofee table: end"), TEX),
                        (("% gofee-si table: start", "% gofee-si table: end"), SI)):
        audit.record(marks[0] not in audit.text(path), path, "the GOFEE table is not printed", "absent",
                     "absent" if marks[0] not in audit.text(path) else "present")
    for label, marks, written in (("tab:ablation", ablation.TEX_MARKS, ablation.tex_table()),
                                  ("tab:gmin", gmin.MARKS, gmin.table()),
                                  ("tab:knockouts", knock.KNOCKOUT_MARKS, knock.knockout_table()),
                                  ("tab:tip4p-gmin", knock.TIP4P_MARKS, knock.tip4p_table()),
                                  ("tab:ase", ase.MARKS["ase"], ase.ase_block()),
                                  ("tab:ase-morse", ase.MARKS["ase-morse"], ase.morse_block())):
        ok = block(marks) == written
        audit.record(ok, TEX, f"{label} is the summariser's block", "as written" if ok else "edited by hand", "as written")
    for marks in (("% gmin-panel table: start", "% gmin-panel table: end"),):
        audit.record(text.count(marks[0]) == 1, TEX, "one GMIN block", "1", str(text.count(marks[0])))
    audit.record("% trap table: start" not in text, TEX, "the endpoint-energy trap table is not printed", "absent",
                 "absent" if "% trap table: start" not in text else "present")


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
    _panel, groups = gmin_rerun_groups(audit)
    g = groups[("lj98", "gmin")]
    audit.cell(SI, "rerun LJ98 GMIN budget", "4000000", float(g[0]["budget"]))
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


def resolve_named_hit_tol(expr: str, names: dict[str, float]) -> float | None:
    """Resolve a tolerance expression through ``names`` or as a numeric literal.

    ``${DEFAULT_HIT_TOL}`` and ``DEFAULT_HIT_TOL`` both follow ``names``.
    A bare number is parsed. Anything else is unresolved, so a check that
    compares the two sides cannot pass by reading the constant twice.
    """
    token = expr.strip().strip("\"'")
    if token.startswith("${") and token.endswith("}"):
        token = token[2:-1]
    if token in names:
        return float(names[token])
    try:
        return float(token)
    except ValueError:
        return None


def jcc_target_lines(tally_text: str, script_text: str) -> tuple[int | None, int | None]:
    """Line the tally names, and the script line that sets the copper target."""
    claimed = re.search(
        r"scripts/elja_jcc_molslab_ensemble\.sh line (\d+) sets that target\.",
        tally_text,
    )
    claimed_line = int(claimed.group(1)) if claimed else None
    actual = next(
        (
            index
            for index, text in enumerate(script_text.splitlines(), 1)
            if text.strip() == "TARGET=-415.971529"
        ),
        None,
    )
    return claimed_line, actual


def supporting_numbers(audit: Audit) -> None:
    """Every number of the SI, read from the row, log, archive or source file that holds it."""
    gmin = load_script("gmin_panel_si", "scripts/gmin_panel/run_panel.py")
    ref = {row[0]: row[3] for row in gmin.SYSTEMS}

    def r6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    # Board constants.
    target_tol_py = audit.text("scripts/target_tol.py")
    target_tol_rs = audit.text("src/target_tol.rs")
    py_default_hit_tol = float(
        re.search(r"DEFAULT_HIT_TOL:\s*float\s*=\s*([0-9.eE+-]+)", target_tol_py).group(1)
    )
    rs_default_hit_tol = float(
        re.search(r"pub const DEFAULT_HIT_TOL: f64 = ([0-9.eE+-]+);", target_tol_rs).group(1)
    )
    audit.cell(
        SI,
        "hit tolerance of the h2h examples",
        "0.001",
        py_default_hit_tol,
    )
    audit.cell(SI, "python and rust default hit tolerances agree", "1", float(py_default_hit_tol == rs_default_hit_tol))
    audit.said(
        SI,
        "We read a hit as an energy at or below the reference plus \\(0.001\\) "
        "(inclusive \\(E \\le E_{\\mathrm{ref}} + 10^{-3}\\)),",
    )
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
    si_text = audit.text(SI)
    for gone in ("campaign of 200 seeds", "lj75-source-", "lj75_ecdf", "lj75_landscape", "occupancy_landfold_lj75",
                 "anneal core developers"):
        audit.record(gone not in si_text, SI, f"the SI leaves out {gone}", "absent",
                     "absent" if gone not in si_text else "present")

    # Hardware and reproduction.
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
    audit.said(SI, f"The best-so-far energy of the seed-0 basin-hopping run ends \\({gap['lj38_bh_s0']}\\) above it, "
                   f"and that of the seed-0 random structure search \\({gap['lj38_rss_s0']}\\) above it.")
    audit.said(SI, f"The best-so-far energy of the seed-0 recommended run ends \\({gap['lj98_recommended_s0']}\\) "
                   f"above it, and that of the seed-0 basin-hopping run \\({gap['lj98_bh_s0']}\\) above it.")
    audit.said(SI, f"reaches the fcc minimum and that of basin hopping ends \\({gap['lj38_bh_s0']}\\) above it.")
    audit.said(SI, f"ends \\({gap['lj98_recommended_s0']}\\) above the Leary minimum and that of basin hopping "
                   f"\\({gap['lj98_bh_s0']}\\) above it.")
    report = "paper/jctc/REPORT.md"
    audit.said(report, f"Basin hopping stops {gap['lj38_bh_s0']} above it.")

    # Copper slab.
    script = "scripts/elja_hq_sci_molslab_bank.sh"
    script_lines = audit.text(script).splitlines()
    energy_line = next(
        i
        for i, text in enumerate(script_lines, 1)
        if text == "export TARGET_ENERGY=-415.971529"
    )
    tol_line = next(
        i
        for i, text in enumerate(script_lines, 1)
        if i > energy_line and text.startswith("export TARGET_TOL=")
    )
    target_energy = float(script_lines[energy_line - 1].split("=", 1)[1])
    names = {"DEFAULT_HIT_TOL": py_default_hit_tol}
    target_tol = resolve_named_hit_tol(script_lines[tol_line - 1].split("=", 1)[1], names)
    audit.said(
        SI,
        f"Its line {energy_line} sets \\texttt{{TARGET\\_ENERGY}} to \\(-415.971529\\) and its line {tol_line} "
        "sets \\texttt{TARGET\\_TOL} from \\texttt{DEFAULT\\_HIT\\_TOL} (\\(10^{-3}\\)~eV).",
    )
    tally = "paper/jctc/scripts/hq_sci_tally.py"
    tally_text = audit.text(tally)
    tally_reference = float(re.search(r"^REFERENCE_EV = ([-+0-9.eE]+)$", tally_text, re.M).group(1))
    tally_tol_match = re.search(r"^REFERENCE_TOL_EV = (.+)$", tally_text, re.M)
    tally_tolerance = (
        resolve_named_hit_tol(tally_tol_match.group(1), names) if tally_tol_match else None
    )
    audit.cell(SI, "TARGET_ENERGY value", "-415.971529", target_energy)
    audit.cell(SI, "TARGET_ENERGY matches tally reference", f"{tally_reference}", target_energy)
    if target_tol is None:
        audit.record(False, SI, "TARGET_TOL value", "unresolved", "0.001")
    else:
        audit.cell(SI, "TARGET_TOL value", "0.001", target_tol)
    if target_tol is None or tally_tolerance is None:
        audit.record(
            False,
            SI,
            "TARGET_TOL matches tally tolerance",
            "unresolved",
            target_tol if target_tol is not None else "unresolved",
        )
    else:
        audit.cell(
            SI,
            "TARGET_TOL matches tally tolerance",
            f"{tally_tolerance:.6g}",
            target_tol,
        )
    audit.said(tally, f"scripts/elja_hq_sci_molslab_bank.sh line {energy_line} sets it too.")
    claimed_line, jcc_target_line = jcc_target_lines(
        tally_text, audit.text("scripts/elja_jcc_molslab_ensemble.sh")
    )
    if claimed_line is None or jcc_target_line is None:
        audit.record(
            False,
            tally,
            "the jcc script's target line",
            "missing" if claimed_line is None else str(claimed_line),
            jcc_target_line if jcc_target_line is not None else "missing",
        )
    else:
        audit.cell(tally, "the jcc script's target line", str(claimed_line), float(jcc_target_line))
    if jcc_target_line is not None:
        audit.said(tally, f"scripts/elja_jcc_molslab_ensemble.sh line {jcc_target_line} sets that target.")
    bests = []
    for index in range(8):
        found = re.search(r"seed \d+: best (\S+) eV", audit.text(f"paper/jctc/logs/cuh2_fcc_rec_{index}.out"))
        bests.append(float(found.group(1)))
    gaps = [b + 415.971529 for b in bests]
    audit.cell(SI, "smallest copper gap", "8e-6", min(gaps))
    audit.cell(SI, "largest copper gap", "3.68e-4", max(gaps))
    audit.cell(SI, "copper seeds inside the tolerance", "8", float(sum(energy_hits(b, -415.971529, DEFAULT_HIT_TOL) for b in bests)))
    audit.cell(SI, "copper seeds inside 1e-6 eV", "0", float(sum(g < 1e-6 for g in gaps)))
    audit.said(SI, "The per-seed gaps run from \\(8\\times 10^{-6}\\)~eV to \\(3.68\\times 10^{-4}\\)~eV above that reference.")
    audit.said(SI, "No seed reached the reference within \\(10^{-6}\\)~eV.")
    audit.said(SI, "Eight seeds lie inside the 0.001~eV tolerance.")
    audit.said(SI, "The hit tolerance is 0.001~eV.")


def manuscript_tip4p(audit: Audit) -> None:
    rows = audit.rows("bench/tip4p-water-rates/rates-a24ba760.csv")
    ten = sorted(pick(rows, n="10", budget="500000"), key=lambda r: int(r["seed"]))
    audit.said(SI, f"\\((\\mathrm{{H_2O}})_{{10}}\\) hits on {frac(ten).replace('/', ' of ')} seeds.")
    calls = [r["calls_to_first_hit"] for r in ten]
    audit.said(SI, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    for n in ("15", "20"):
        group = pick(rows, n=n, budget="500000")
        audit.cell(SI, f"tip4p ({n}) hits at 5e5", "0/6", frac(group))
        gaps = [float(r["best_kj"]) - float(r["reference_kj"]) for r in group]
        word = {"15": "fifteen", "20": "twenty"}[n]
        audit.said(SI, f"The best {word}-molecule energies lie \\({min(gaps):.10f}\\) to \\({max(gaps):.10f}\\)~kJ/mol above")


def sci(p: float) -> str:
    """A p below 0.01 as the manuscript prints it, two significant figures times a power of ten."""
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa}\\times 10^{{{int(exponent)}}}"


WATER_RERUN = "paper/jctc/data/water_v45b/rerun-9a30f319"


def water_receipts(audit: Audit) -> dict[str, dict[int, dict]]:
    """The 384 receipts of the hexamer rerun, each held to SHA256SUMS, paired by seed and start."""
    import hashlib
    import json
    base = ROOT / WATER_RERUN
    sums = {}
    for line in (base / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.lstrip("*").removeprefix("./")] = digest
    out: dict[str, dict[int, dict]] = {}
    for arm in ("anneal", "crest"):
        out[arm] = {}
        for path in sorted((base / arm).glob("seed-*.json")):
            name = f"{arm}/{path.name}"
            audit.record(sums.get(name) == hashlib.sha256(path.read_bytes()).hexdigest(), WATER_RERUN,
                         f"{name} matches SHA256SUMS", sums.get(name, "absent"), "checked")
            row = json.loads(path.read_text())
            out[arm][row["seed"]] = row
    audit.cell(WATER_RERUN, "receipts in SHA256SUMS", "384", float(len(sums)))
    audit.cell(WATER_RERUN, "seeds paired on one start", "192",
               float(sum(out["anneal"][s]["start_sha256"] == out["crest"][s]["start_sha256"]
                         for s in out["anneal"] if s in out["crest"])))
    readme = f"{WATER_RERUN}/README.md"
    audit.said(readme, "`9a30f3190b3bb7040c675a904f8388eb68224151`")
    audit.record(git("cat-file", "-e", "9a30f3190b3bb7040c675a904f8388eb68224151^{commit}").returncode == 0, readme,
                 "the record revision is in this repository", "9a30f319", "checked")
    return out


def manuscript_hexamer(audit: Audit) -> None:
    numbers = "paper/jctc/logs/water/NUMBERS.tsv"
    lines = [line.rstrip("\n").split("\t") for line in (ROOT / numbers).read_text().splitlines()]
    header = next(cells for cells in lines if cells[0] == "# seed")
    header[0] = "seed"
    seeds = [dict(zip(header, cells)) for cells in lines if cells[0].isdigit()]
    audit.read[numbers] = seeds
    by = {(r["seed"], r["arm"]): r for r in seeds}
    for row in tex_table(audit, SI, "tab:crest-counters"):
        seed = row[0]
        log = ROOT / f"paper/jctc/logs/water/crest-hexamer-seed{seed}.log"
        engrad = re.search(r"Total number of energy\+grad calls:\s*(\d+)", log.read_text()).group(1)
        audit.cell(f"{SI} tab:crest-counters seed {seed}", "old counter", row[1], float(by[(seed, "crest")]["pes_calls"]))
        audit.cell(f"{SI} tab:crest-counters seed {seed}", "energy+gradient", row[2], float(engrad))
        audit.cell(f"{SI} tab:crest-counters seed {seed}", "fair cap = old counter", row[1], float(by[(seed, "fair")]["pes_calls"]))
    for row in tex_table(audit, SI, "tab:hexamer-gaps"):
        seed = row[0]
        where = f"{SI} tab:hexamer-gaps seed {seed}"
        audit.cell(where, "gap 4000", row[1], float(by[(seed, "paper4000")]["signed_dE_polished_Eh"]))
        audit.cell(where, "gap old-counter cap", row[2], float(by[(seed, "fair")]["signed_dE_polished_Eh"]))
        audit.cell(where, "deviation 4000", row[3], float(by[(seed, "paper4000")]["rmsd_label_free_A"]))
        audit.cell(where, "deviation old-counter cap", row[4], float(by[(seed, "fair")]["rmsd_label_free_A"]))
    old = [float(by[(str(s), "crest")]["pes_calls"]) for s in range(6)]
    audit.said(SI, f"from {min(old):.0f} to {max(old):.0f} evaluations.")
    engrads = [float(re.search(r"energy\+grad calls:\s*(\d+)", (ROOT / f"paper/jctc/logs/water/crest-hexamer-seed{s}.log").read_text()).group(1)) for s in range(6)]
    audit.said(SI, f"CREST's energy and gradient totals run from {min(engrads):.0f} to {max(engrads):.0f}.")
    audit.cell(SI, "the old counter is smaller on every seed", "6", float(sum(o < e for o, e in zip(old, engrads))))
    audit.cell(SI, "the 4000-call arm spends 4000", "4000", max(float(by[(str(s), "paper4000")]["pes_calls"]) for s in range(6)))
    within = [s for s in range(6) if all(abs(float(by[(str(s), a)]["signed_dE_polished_Eh"])) <= 1e-7 for a in ("paper4000", "fair"))]
    audit.cell(SI, "seeds within 1e-7 Eh on both ledgers", "5", float(len(within)))
    audit.cell(SI, "the seed outside 1e-7 Eh", "1", float(next(s for s in range(6) if s not in within)))
    dev = max(float(by[(str(s), a)]["rmsd_label_free_A"]) for s in range(6) for a in ("paper4000", "fair"))
    audit.said(SI, f"most \\({dev:.5f}\\)~\\AA.")
    for arm, text in (("crest", "\\(544.0\\)~s for CREST"), ("paper4000", "\\(18.8\\)~s for the 4000-call arm"), ("fair", "\\(195.5\\)~s for the arm")):
        audit.cell(SI, f"{arm} wall median", re.search(r"\\\((\d+\.\d)\\\)", text).group(1), median([float(by[(str(s), arm)]["wall_s"]) for s in range(6)]))
        audit.said(SI, text)

    repolish = audit.rows("paper/jctc/data/hexamer-xtb680-repolish.csv")
    shifts = sorted((abs(float(r["new_energy_Eh"]) - float(r["old_energy_Eh"])), r["structure"], r["seed"]) for r in repolish)
    audit.cell(SI, "xtb 6.8.0 finals", "18", float(len(shifts)))
    audit.cell(SI, "largest xtb 6.8.0 shift", "1.08e-7", shifts[-1][0])
    audit.cell(SI, "largest shift is the CREST final of seed 1", "crest 1", f"{shifts[-1][1]} {shifts[-1][2]}")
    audit.cell(SI, "smallest of the other seventeen shifts", "0.9e-8", shifts[0][0])
    audit.cell(SI, "largest of the other seventeen shifts", "2.0e-8", shifts[-2][0])
    gap = {(r["structure"], r["seed"]): float(r["gap_Eh"]) for r in repolish}
    audit.cell(SI, "xtb 6.8.0 seed 1 gap, 4000 calls", "-4.35e-8", gap[("paper4000", "1")])
    audit.cell(SI, "xtb 6.8.0 seed 1 gap, old counter", "-4.34e-8", gap[("fair", "1")])
    for arm in ("paper4000", "fair"):
        inside = sum(abs(gap[(arm, str(s))]) <= 1e-7 for s in range(6))
        audit.cell(SI, f"xtb 6.8.0 {arm} seeds within 1e-7 Eh", "6", float(inside))
    for text in ("\\(-4.35\\times 10^{-8}\\)~Eh on the 4000-call ledger", "\\(-4.34\\times 10^{-8}\\)~Eh on the ledger whose cap equals the old",
                 "\\(1.08\\times 10^{-7}\\)~Eh.", "\\(0.9\\times 10^{-8}\\) to \\(2.0\\times 10^{-8}\\)~Eh"):
        audit.said(SI, text)

    receipts = water_receipts(audit)
    flags = {arm: {seed: bool(r["hit"]) for seed, r in receipts[arm].items()} for arm in ("anneal", "crest")}
    hits = {arm: sum(flags[arm].values()) for arm in flags}
    n = len(receipts["anneal"])
    audit.cell(TEX, "hexamer rerun seeds", "192", float(n))
    hex_table = paired_table(flags["anneal"], flags["crest"])
    lo, hi = newcombe_paired_interval(*hex_table)
    p_hex = mcnemar(hex_table[1], hex_table[2])
    audit.cell(TEX, "hexamer rerun paired outcome table", "59 62 35 36", " ".join(map(str, hex_table)))
    audit.cell(TEX, "hexamer rerun McNemar p", "0.00796", p_hex)
    walls = {arm: median([float(r["wall_seconds"]) for r in receipts[arm].values()]) for arm in receipts}
    frames = {r["frames_scored"] for r in receipts["crest"].values()}
    gates = {r["isomer_gate_A"] for arm in receipts for r in receipts[arm].values()}
    classifier = {r["classifier_sha256"] for r in receipts["crest"].values()}
    binary = {r["binary_sha256"] for r in receipts["anneal"].values()}
    audit.cell(TEX, "hexamer rerun: one classifier, the anneal binary, for both arms", "1",
               float(len(classifier) == 1 and classifier == binary))
    audit.cell(TEX, "hexamer rerun: CREST frames scored", "180", frames.pop() if len(frames) == 1 else frames)
    audit.cell(TEX, "hexamer rerun: isomer gate", "0.784112", gates.pop() if len(gates) == 1 else gates)
    audit.cell(TEX, "hexamer rerun: CREST runs at the cap", str(n),
               float(sum(r["calls"] == 4000 and r["engrad_cap_reached"] for r in receipts["crest"].values())))
    audit.cell(TEX, "hexamer rerun: anneal runs spending 4000", str(n),
               float(sum(r["force_count"] == 4000 for r in receipts["anneal"].values())))
    audit.said(TEX, f"it hits {hits['anneal']} of {n} seeds and CREST {hits['crest']}")
    audit.record(hits["anneal"] > hits["crest"] and p_hex < 0.05, COVER, "the code hits more hexamer seeds than CREST",
                 "more, p<0.05", f"{hits['anneal']} {hits['crest']} {p_hex}")
    audit.said(COVER, "and more than CREST on the GFN2-xTB water hexamer with both codes stopped at the same number of "
                      "energy and gradient calls.")
    audit.said(TEX, f"This code hits {hits['anneal']} of {n} seeds and CREST {hits['crest']}.")
    audit.said(TEX, f"Paired by seed, both hit {hex_table[0]} seeds, this code alone {hex_table[1]}, CREST alone "
                    f"{hex_table[2]}, and neither {hex_table[3]}, with an exact McNemar \\(p\\) of "
                    f"\\({sci(p_hex)}\\) and a paired 95\\% interval of the difference in hit rate from "
                    f"{lo:.3f} to {hi:.3f}.")
    audit.said(TEX, f"(\\(p={sci(p_hex)}\\)).")
    audit.said(TEX, f"The median walls are {walls['anneal']:.3f}~s for this code and {walls['crest']:.3f}~s for CREST.")
    audit.said(TEX, "places inside an isomer gate of 0.784112~\\AA\\ around the lowest hexamer; a CREST run hits "
                    "when any of the 180 frames it scores lies inside the gate.")
    audit.said(TEX, f"more seeds of the GFN2-xTB water hexamer than CREST, {hits['anneal']} of {n} against {hits['crest']}.")
    audit.said(SI, f"{hits['anneal']} of {n} against {hits['crest']}, both {hex_table[0]}, this code alone "
                   f"{hex_table[1]}, CREST alone {hex_table[2]}, neither {hex_table[3]}, McNemar \\(p={p_hex:.5f}\\), "
                   f"and the paired interval {lo:.3f} to {hi:.3f}.")
    audit.said(TEX, f"Run without the cap on six seeds, CREST spends {min(engrads):.0f} to {max(engrads):.0f} energy and "
                    "gradient calls")
    six = sum(all(abs(gap[(arm, str(s))]) <= 1e-7 for arm in ("paper4000", "fair")) for s in range(6))
    audit.said(TEX, f"\\(10^{{-7}}\\)~Eh on {NUMBER_WORDS[len(within)]} seeds at xtb 6.7.1 and on all {NUMBER_WORDS[six]} at xtb 6.8.0;")


def manuscript_cross_field(audit: Audit) -> None:
    rows = audit.rows("results/cross-field-hits.csv")
    stalls = audit.rows("results/lj-funnel-stalls.csv")
    rates = audit.rows("bench/tip4p-water-rates/rates.csv")
    lj98 = sorted(pick(rows, system="lj", n="98", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    calls = [r["calls_to_first_hit"] for r in lj98]
    audit.said(SI, f"LJ\\(_{{98}}\\) hit on {frac(lj98).replace('/', ' of ')}.")
    audit.said(SI, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    audit.said(SI, f"The median is {median([float(c) for c in calls]):.0f}.")
    retry98 = pick(stalls, phase="retry", n="98")
    lj104 = sorted(pick(rows, system="lj", n="104", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    hit104 = [r for r in lj104 if r["hit"] == "1"]
    retry104 = sorted((r["seed"] for r in pick(stalls, phase="retry", n="104") if r["hit"] == "1"), key=int)
    retry_median = median([float(r["calls_to_hit"]) for r in retry98])
    arm_median = median([float(c) for c in calls])
    audit.record(retry_median < arm_median, SI, "the retry median on LJ98 is lower than the arm median",
                 "lower", "lower" if retry_median < arm_median else "not lower")
    audit.said(SI, f"By comparison, the pentagonal-retry baseline also hits {frac(retry98).replace('/', ' of ')} on "
                    f"LJ\\(_{{98}}\\) with a lower median of {retry_median:.0f} calls, and on LJ\\(_{{104}}\\) reaches "
                    f"{len(retry104)} of 6 seeds ({retry104[0]}, {retry104[1]}, and {retry104[2]}) versus "
                    f"{len(hit104)} of {len(lj104)} ({hit104[0]['seed']} and {hit104[1]['seed']}, at "
                    f"{hit104[0]['calls_to_first_hit']} and {hit104[1]['calls_to_first_hit']} calls) for the BIPOP variant.")
    only_arm = sorted({r["seed"] for r in hit104} - set(retry104))
    only_retry = sorted(set(retry104) - {r["seed"] for r in hit104})
    audit.cell(SI, "the hit the retry rows miss", "6401", " ".join(only_arm))
    audit.cell(SI, "hits the retry rows record and this arm misses", "6402 6404", " ".join(only_retry))
    for n, budget in (("15", "500000"), ("20", "2000000")):
        trust = pick(rows, system="tip4p", n=n, arm="cage-trust")
        audit.cell(SI, f"cage-trust ({n}) hits", "0/6", frac(trust))
        audit.cell(SI, f"cage-trust ({n}) budget", budget, float(trust[0]["budget"]))
        rate = pick(rates, n=n, budget=budget)
        audit.cell(SI, f"rate file ({n}) at {budget}", "1/6", frac(rate))
        audit.cell(SI, f"rate file ({n}) hit seed", {"15": "3", "20": "2"}[n], " ".join(r["seed"] for r in rate if r["hit"] == "1"))
    best = min(pick(rows, system="tip4p", n="20", arm="cage-trust"), key=lambda r: float(r["best"]))
    audit.said(SI, f"\\({float(best['best']):.6f}\\)~kJ/mol, which is \\({float(best['best']) - float(best['reference']):.3f}\\)~kJ/mol above")


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
           "hits more seeds on 9 of 13 and GMIN hits more on 2", "168.9~s", "6 of the 9 wins",
           "lower for this code on 12 of the 13 clusters", "12/13")
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
    for name in (n for n in names if not index.audited_exists(n)):
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
        remote_refs = [f"refs/remotes/origin/{branch}" for branch in
                       (*AUDIT_REMOTE_BRANCHES, *AUDIT_OPTIONAL_REMOTES)]
        audited_refs = ["HEAD", *(ref for ref in remote_refs
                                   if git("show-ref", "--verify", "--quiet", ref).returncode == 0),
                        *git("for-each-ref", "--format=%(refname)", "refs/tags").stdout.split()]
        audited_heads = git("rev-parse", *audited_refs).stdout.split()
        self.commit = {n: line.split()[0] for n, line in zip(asked, out)
                       if line.split()[1:2] == ["commit"]}
        self.parents: dict[str, list[str]] = {}
        roots = sorted({*audited_refs, *self.commit.values()})
        for line in git("rev-list", "--parents", *roots).stdout.splitlines():
            commit, *parents = line.split()
            self.parents[commit] = parents
        self.audited = self.closure(audited_heads)
        remote = git("for-each-ref", "--format=%(objectname)", *remote_refs).stdout.split()
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

    def audited_exists(self, revision: str) -> bool:
        """The revision exists in the histories the paper audit accepts."""
        return self.resolve(revision) in self.audited

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
        checks = [("local revision is not in this repository", not index.audited_exists(local))]
        for column in ("pushed_revision", "branch_revision"):
            revision = row[column]
            if not revision and column == "branch_revision":
                continue
            checks.append((f"{column} {revision[:8]} is in this repository", index.audited_exists(revision)))
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
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
    tag = f"v{version}"
    if git("show-ref", "--verify", "--quiet", f"refs/tags/{tag}").returncode != 0:
        problems.append(f"the release tag {tag} is not present; fetch tags from origin")
    for branch in AUDIT_REMOTE_BRANCHES:
        remote_ref = f"refs/remotes/origin/{branch}"
        if git("show-ref", "--verify", "--quiet", remote_ref).returncode != 0:
            problems.append(f"the remote ref origin/{branch} is not present; fetch {branch} from origin")
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
    manuscript = collapsed(re.sub(r"~?\\todo\{[^}]*\}", "", audit.text(TEX).split("\\begin{abstract}", 1)[1]
                                  .split("\\end{abstract}", 1)[0]))
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
        audit.record(found, COVER, "a cover-letter sentence with a number is a sentence of the abstract",
                     sentence[:70], sentence[:70] if found else "not in the manuscript")
    audit.record(checked == len(COVER_OTHER), COVER,
                 "the only cover-letter sentences with a number are the preprint and graphic lines",
                 str(len(COVER_OTHER)), str(checked))
    for word in ("\\rerun", "Zenodo", "DOI", "pending", "being run", "rerun"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter carries no process note ({word})",
                     "absent", "absent" if word not in audit.text(COVER) else "present")
    toc = audit.text(COVER_TOC)
    audit.record("paperwidth=3.25in,paperheight=1.75in" in toc, COVER_TOC, "the graphic slot is 3.25 in by 1.75 in",
                 "3.25in 1.75in", "3.25in 1.75in")
    audit.said(COVER, "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")
    for word in ("TODO", "\\todo"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter has no {word}", "none", "none")
    for word in ("Science Institute", "University of Iceland, Reykjavik, Iceland\\\\"):
        audit.record(word not in audit.text(COVER), COVER, f"the cover letter does not list {word}", "absent",
                     "absent" if word not in audit.text(COVER) else "present")
    audit.said(COVER, "Parts of the search with no such run are described as design, with no claim about their "
                      "effect.")
    audit.said(COVER, "The comparisons with the Atomic Simulation Environment are reported at the effort each run "
                      "spent.")
    audit.said(COVER, "Every printed number is computed from rows committed to the repository.")


REFS_BLOB = "d646763ac2b4a1f8b7aab78452616c4400d53d10"
# Wording the review of the first rewrite showed to overclaim; none may come back.
OVERCLAIMS = ("same trap", "confined to one funnel", "cannot become the stored best", "cannot win",
              "basin-hopping setting", "one setting", "adds nothing", "does no better", "this setting",
              "crosses funnels more often", "the setting of", "that setting", "does not come from choosing among the moves", "rebuilds a facet",
              "completes the patch", "never restarts", "is equivalent to", "across funnels")


def science_claims(audit: Audit) -> None:
    """Overclaiming wording stays out, endpoint prose stays on energies, and every citation resolves."""
    for path in (TEX, SI, COVER):
        flat = collapse(audit.text(path)).lower()
        for phrase in OVERCLAIMS:
            audit.record(phrase not in flat, path, f"no overclaim: {phrase}", "absent",
                         "absent" if phrase not in flat else "present")
    text = audit.text(TEX)
    audit.record("\\label{sec:endpoints}" not in text, TEX, "the endpoint-energy section is not printed", "absent",
                 "absent" if "\\label{sec:endpoints}" not in text else "present")
    results = text.split("\\label{sec:results}", 1)[1].split("\\begin{acknowledgement}", 1)[0]
    visible = re.sub(r"\\(?:label|ref)\{[^}]*\}", "", re.sub(r"(?m)^%.*$", "", results)).lower()
    for word in ("funnel", "trap", "confin", "hypothes"):
        audit.record(word not in visible, TEX, f"the results and conclusions do not say {word}", "absent",
                     "absent" if word not in visible else "present")
    audit.record("hypothes" not in text.lower(), TEX, "the manuscript states no hypothesis", "absent",
                 "absent" if "hypothes" not in text.lower() else "present")
    audit.said(TEX, "Five parts of the search have a run with and without them, and Section~\\ref{sec:knockouts} reports "
                    "those runs: the structure priors and the structure templates (Section~\\ref{sec:priors}), the "
                    "restart on a plateau (Section~\\ref{sec:stall}), the length unit (Section~\\ref{sec:units}), and "
                    "the choice among the moves (Section~\\ref{sec:posterior}).")
    audit.said(TEX, "The packing moves, the symmetry proposals, the basin bias, the tabu list, the screens, and the "
                    "atom-count gates have no such run, and the trail step has one only on LJ\\(_{75}\\) at a revision "
                    "outside the history of the submitted revision, which the article does not print.")
    audit.said(TEX, "They are described here as the design of the search, and the article makes no claim about what "
                    "any of them contributes.")
    trail = "666591a0"
    audit.record(git("merge-base", "--is-ancestor", trail, SUBMITTED).returncode != 0, TEX,
                 "the trail record is outside the history of the submitted revision", "outside", "checked")
    audit.record(bool(git("branch", "-r", "--contains", trail).stdout.strip()), TEX, "the trail record is on origin",
                 "present", "checked")
    audit.said(TEX, "Every test is two-sided and exact, so no continuity correction enters any of them.")
    audit.said(TEX, "No adjustment for multiple comparisons is made: every \\(p\\) is that of one cluster and one pair of "
                    "arms, every cluster run is reported, and a difference in hits is called significant only when its "
                    "own exact test is below 0.05.")
    audit.said(TEX, "No equivalence margin was set before the runs, so the table supports no claim that the two draws "
                    "are equivalent.")
    audit.said(SI, "No equivalence margin was set before any run, so no table supports a claim that two arms are "
                   "equivalent.")

    # Citations: the frozen bibliography, the statistics file, and every key.
    blob = git("hash-object", "paper/jctc/refs.bib").stdout.strip()
    audit.cell("paper/jctc/refs.bib", "blob", REFS_BLOB, blob)
    keys = {}
    for bib in ("paper/jctc/refs.bib", "paper/jctc/refs-stats.bib"):
        keys[bib] = set(re.findall(r"^@\w+\{([^,\s]+),", audit.text(bib), re.M))
    both = keys["paper/jctc/refs.bib"] & keys["paper/jctc/refs-stats.bib"]
    audit.cell("paper/jctc/refs-stats.bib", "keys also in refs.bib", "none", " ".join(sorted(both)) or "none")
    known = set.union(*keys.values())
    for path in (TEX, SI):
        audit.said(path, "\\bibliography{refs,refs-stats}")
        cited = {k.strip() for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", audit.text(path)) for k in group.split(",")}
        missing = sorted(cited - known)
        audit.cell(path, "cited keys without an entry", "none", " ".join(missing) or "none")
    unused = sorted(keys["paper/jctc/refs-stats.bib"] - {
        k.strip() for path in (TEX, SI) for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", audit.text(path))
        for k in group.split(",")})
    audit.cell("paper/jctc/refs-stats.bib", "entries no text cites", "none", " ".join(unused) or "none")
    audit.said(TEX, "two-sided Fisher exact test \\cite{fisherLogicInductiveInference1935}.")
    audit.said(TEX, "\\cite{newcombeImprovedConfidenceIntervals1998}, built from Wilson intervals "
                    "\\cite{wilsonProbableInferenceLaw1927}.")
    audit.said(TEX, "The exact McNemar test \\cite{mcnemarNoteSamplingError1947}")
    audit.said(TEX, "the exact Wilcoxon signed-rank test\n\\cite{wilcoxonIndividualComparisons1945}")
    for key, doi in (("mcnemarNoteSamplingError1947", "10.1007/bf02295996"),
                     ("wilcoxonIndividualComparisons1945", "10.2307/3001968"),
                     ("newcombeImprovedConfidenceIntervals1998",
                      "10.1002/(sici)1097-0258(19981130)17:22<2635::aid-sim954>3.0.co;2-c"),
                     ("fisherLogicInductiveInference1935", "10.2307/2342435"),
                     ("hyndmanSampleQuantiles1996", "10.1080/00031305.1996.10473566"),
                     ("efronBootstrap1979", "10.1214/aos/1176344552")):
        bib = "paper/jctc/refs-stats.bib" if key in audit.text("paper/jctc/refs-stats.bib") else "paper/jctc/refs.bib"
        entry = audit.text(bib).split("{" + key + ",", 1)[-1].split("\n}", 1)[0].lower()
        audit.record(doi in entry, bib, f"{key} carries its Crossref DOI", doi,
                     doi if doi in entry else "missing")
    audit.said(TEX, "spirit of the compression of Doye \\cite{doye2000}.")
    for key in ("wilsonProbableInferenceLaw1927", "newcombeImprovedConfidenceIntervals1998",
                "mcnemarNoteSamplingError1947", "wilcoxonIndividualComparisons1945", "hyndmanSampleQuantiles1996",
                "efronBootstrap1979", "fisherLogicInductiveInference1935"):
        audit.record(key in audit.text(SI), SI, f"the statistical methods cite {key}", "cited",
                     "cited" if key in audit.text(SI) else "uncited")

    for word in ("prior-on", "template-free", "c2de652d"):
        for path in (TEX, SI, COVER):
            audit.record(word not in audit.text(path), path, f"no row is labelled {word}", "absent",
                         "absent" if word not in audit.text(path) else "present")


def manuscript_new(audit: Audit) -> None:
    """The abstract split, the added panels, the definitions, the priors, and the host paragraph."""
    panel, groups = gmin_panel_groups(audit)

    def name(system: str) -> str:
        return panel.LABELS[system][1].replace(", \\(\\rho=", " at \\(\\rho=")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(groups[(system, arm)])} of {len(groups[(system, arm)])}"

    def p_of(system: str) -> float:
        return gmin_paired(groups, system)["mcnemar"]

    def wall(system: str, arm: str) -> float:
        return median([float(r["wall_s"]) for r in groups[(system, arm)]])

    # Constants of the definitions and of the priors, read from the source.
    preset = "src/methods/cluster_hopping/preset.rs"
    for fragment in ("TEMPERATURE: f64 = 0.8;", "BIAS_HEIGHT: f64 = 0.25;", "MERGE_RADIUS: f64 = 0.7;",
                     "NEIGHBOUR_CUTOFF: f64 = 1.6;", "SYMMETRISE_CUTOFF: f64 = 2.5;", "SOAP_RMSD: f64 = 0.35;",
                     "SOAP_CUTOFF: f64 = 3.5;", "SCREEN_MARGIN: f64 = 2.0;"):
        audit.said(preset, fragment)
    config = "src/methods/cluster_hopping/config.rs"
    for fragment in ("bias_gamma: 5.0,", "tabu_capacity: 8,", "ContinuousSymmetry::Measure { interval: 10 };",
                     "prior_soap: true,", "calibrated_screen: true,"):
        audit.said(config, fragment)
    config_src = audit.text(config)
    for removed in ("reseed_hops: 5,", "pub prior_reseed:", "pub prior_growth:", "pub prior_core:",
                    "pub prior_handover:", "pub prior_motif:"):
        audit.record(removed not in config_src, config, f"inert {removed.rstrip(':,')} is removed",
                     "absent", "absent" if removed not in config_src else "present")
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
    audit.said("src/csm.rs", "Lowest-measure average among `C_i`, `C_5`, and `D_5h`.")
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
    text = audit.text(SI)
    build = audit.text("scripts/gmin_panel/build_gmin.sh")
    for flag in ("-DWITH_ALIGN=OFF", "-DWITH_MPI=OFF", "-DWITH_OPENMP=OFF", "-DWITH_MYBLAS=OFF", "-DWITH_MYLAPACK=OFF"):
        audit.record(flag in build, "scripts/gmin_panel/build_gmin.sh", f"the GMIN build sets {flag}", "set",
                     "set" if flag in build else "absent")
    sessions = ""
    for shard in GMIN_SHARDS:
        prov = dict(line.split(" ", 1) for line in audit.text(f"{GMIN_RERUN}/{shard}/PROVENANCE").splitlines()
                    if " " in line)
        session = prov["source"].rsplit("/", 1)[0] + "/session.txt"
        sessions += git("show", f"{prov['commit']}:{session}").stdout
        audit.record(bool(git("show", f"{prov['commit']}:{session}").stdout), session, "the shard session is committed",
                     "present", "present" if sessions else "absent")
    audit.record("gfortran" not in sessions, "bench/gmin-panel/rerun-20261119", "the rerun sessions give no compiler",
                 "absent", "absent" if "gfortran" not in sessions else "present")
    audit.said(SI, "GMIN is built by \\path{scripts/gmin_panel/build_gmin.sh} with ALIGN, MPI, and OpenMP off, against "
                   "the system BLAS and LAPACK; the record does not give the compiler version.")
    audit.record("13.3.0" not in text.split("\\label{sec:host}", 1)[1].split("\\section", 1)[0], SI,
                 "the host paragraph prints no compiler version", "absent", "checked")
    commands = {
        "bash scripts/gmin_panel/build_gmin.sh": "scripts/gmin_panel/build_gmin.sh",
        "snakemake -s workflow/Snakefile gmin_panel": "workflow/Snakefile",
        "python3 paper/jctc/scripts/gmin_table.py": "paper/jctc/scripts/gmin_table.py",
        "python3 paper/jctc/scripts/knockout_tables.py": "paper/jctc/scripts/knockout_tables.py",
        "python3 paper/jctc/scripts/water_rerun.py": "paper/jctc/scripts/water_rerun.py",
        "python3 paper/jctc/scripts/ase_table.py": "paper/jctc/scripts/ase_table.py",
        "python3 paper/jctc/scripts/water_h2h.py": "paper/jctc/scripts/water_h2h.py",
    }
    for command, path in commands.items():
        audit.record(command in text and (ROOT / path).exists(), SI, "the command names a script that exists",
                     command, path)
    audit.said("workflow/Snakefile", "rule gmin_panel:")
    audit.said("paper/jctc/scripts/water_h2h.py", 'command = f"gfn2_water_h2h {MOLECULES} {budget} {seed} {outdir.name}"')
    audit.said("paper/jctc/scripts/water_h2h.py", 'paper_dir = LOGS / f"anneal-paper4000-seed{seed}"')
    audit.said("paper/jctc/logs/water/crest-hexamer-seed0.log", "$ path start.xyz -T 1")
    audit.said("paper/jctc/logs/water/VERSIONS", "polish=xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500")
    audit.said(SI, "crest start.xyz -T 1\nxtb --gfn 2 --opt vtight --acc 0.01 --iterations 500\ngfn2_water_h2h 6 4000 0 anneal-paper4000-seed0")
    audit.said(SI, "\\texttt{paper/jctc/scripts/ablation\\_table.py} writes the ablation table of the main text")
    for script in ("ablation_table.py",):
        audit.record((ROOT / "paper/jctc/scripts" / script).exists(), SI, f"{script} exists", "exists", "exists")
    for path in (TEX, SI):
        audit.record("\\section{Several chains}" not in audit.text(path) and "eq:rank" not in audit.text(path), path,
                     "the dropped several-chains section stays out", "absent", "absent")

    # Title, affiliation, spelling, and the pending DOI.
    cover = "paper/jctc/cover-letter.tex"
    title = ("Seed-paired knockouts of a basin-hopping search: structure priors, templates, stall rules, and the "
             "length unit, with comparisons to GMIN, CREST, and the Atomic Simulation Environment")
    audit.said(TEX, "\\title{" + title + "}")
    audit.said(SI, "\\title{" + title + "}")
    audit.said(cover, "``" + title)
    for path in (TEX, SI, cover):
        audit.record("Packing moves, symmetry proposals" not in audit.text(path), path, "the r3 title is gone", "absent",
                     "absent" if "Packing moves, symmetry proposals" not in audit.text(path) else "present")
    for path in (TEX, SI, cover):
        audit.record("basin-hopping setting" not in audit.text(path), path, "the old title is gone", "absent",
                     "absent" if "basin-hopping setting" not in audit.text(path) else "present")
    for path in (TEX, SI):
        audit.said(path, "\\affiliation{SURF B.V., Amsterdam}")
        audit.record("SURF BV" not in audit.text(path), path, "the affiliation is spelled one way", "SURF B.V.", "SURF B.V.")
        audit.record("Science Institute" not in audit.text(path), path, "the affiliations are SURF B.V. and TurtleTech ehf.",
                     "absent", "absent" if "Science Institute" not in audit.text(path) else "present")
        audit.said(path, "\\affiliation{SURF B.V., Amsterdam}\n\\alsoaffiliation{TurtleTech ehf., Reykjavik, Iceland}")
    availability = audit.text(TEX).split("\\section*{Data and software availability}", 1)[1]
    audit.said(TEX, "Each imported row file under \\path{paper/jctc/data} carries a \\texttt{PROVENANCE} file that names "
                    "the record branch, the commit, and the blob it was copied from.")
    audit.said(TEX, "Several revisions that produced rows are not in the repository, and the Supporting Information "
                    "lists them; their rows can be audited and cannot be rerun from the repository.")
    audit.record("not yet" not in availability and "\\todo" not in availability, TEX,
                 "the data statement carries no pending item", "absent", "checked")
    audit.record("Zenodo record for this submission" not in availability, TEX,
                 "the data statement carries no pending-record note", "absent",
                 "absent" if "Zenodo record for this submission" not in availability else "present")
    for path in (TEX, SI, cover):
        found = sorted({m.group(0) for m in BRITISH.finditer(audit.text(path))})
        audit.record(not found, path, "American spelling", "none", " ".join(found) or "none")
    for figure in ("landscapes.pdf", "landfold_maps", "fig:minima-maps"):
        audit.record(figure not in audit.text(TEX), TEX, f"the main text leaves out {figure}", "absent",
                     "absent" if figure not in audit.text(TEX) else "present")


RERUN_ITEMS: dict[str, str] = {}
RERUN_TAG = re.compile(r"\\rerun\{([^}]*)\}")


def rerun_tags(text: str) -> list[tuple[int, str]]:
    """The line number and item of each [RERUN: item N] tag in a source text."""
    return [(number, found.group(1)) for number, line in enumerate(text.splitlines(), 1)
            for found in RERUN_TAG.finditer(line) if not line.lstrip().startswith("%")]


# Results of the third rewrite that a rerun replaced. None may come back, fenced or not.
REPLACED_CLAIMS = (
    (SI, ("hit LJ\\(_{75}\\) on 0 of 12 seeds", "this code hits more seeds on 9 of 13 clusters",
          "Paired by seed, the exact McNemar test is below 0.05 on 5 clusters", "The 4 are LJ\\(_{75}\\)",
          "The other 5 wins of this code are", "9 of 12 against 5 of 12 and 12 of 12 against 2 of 12",
          "In the panel GMIN hits LJ\\(_{75}\\) and LJ\\(_{98}\\) on 0 of 12 seeds",
          "Minima hopping stopped at the cap on all 72 runs", "Both ASE arms hit 0 of 12 on all 4 Morse clusters",
          "ASE minima hopping hits LJ\\(_{38}\\) on 1 of 6, LJ\\(_{55}\\) on 6 of 6",
          "hits 121 of 192 seeds for the default hop and 97 of 192 for CREST",
          "CREST with the polish under the same budget",
          "\\section{Results that wait on reruns}")),
    (TEX, ("so the data do not attribute the hits to any single mechanism",
           "it does not yet test any of the four hypotheses separately",
           "No row was run with the shell and lattice moves switched off",
           "No row of this code was run with its symmetry projections switched off",
           "No row was run with the stall rules replaced by a random restart",
           "No row was run with the basin bias or the tabu list switched off",
           "Which component carries them, if any single one does, is not yet measured",
           "does not separate the methods from their settings",
           "Whether they hold against GMIN with symmetrization and an input for each size is not yet measured",
           "needs the endpoint structures and the occupancy along each run",
           "the occupancy trajectories will test it",
           "No Lennard-Jones row was run at the current preset, which has no structure template",
           "Whether the Lennard-Jones gains hold at the current preset, without the structure priors, is not yet measured",
           "is not yet deposited", "the per-seed rows of a rerun will allow the paired test",
           "and the results that wait on reruns.")),
)
TODO_ITEMS: dict[str, str] = {}
TODO_TAG = re.compile(r"\\todo\{([^}]*)\}")
# Wording that defers a claim to a run not made, or fences off a part of the paper as its limits.
HEDGES = ("not yet", "will test", "will show", "will allow", "future work", "in future", "remains to be",
          "\\section{Limits}", "\\section{Limitations}", "\\section*{Limits}", "\\subsection{Limits}")


def rerun_claims(audit: Audit) -> None:
    """No result that a rerun replaced is printed again, fenced or not."""
    for path, fragments in REPLACED_CLAIMS:
        flat = collapse(audit.text(path))
        for fragment in fragments:
            found = flat.count(collapse(fragment))
            audit.record(found == 0, path, "a replaced result is not printed", fragment, f"{found} found")


def rerun_check(audit: Audit, pre_submission: bool = False) -> None:
    """No source carries a rerun tag; every tag names an item that waits on a rerun, and none does."""
    for path in RERUN_FILES:
        for number, item in rerun_tags(audit.text(path)):
            audit.record(item in RERUN_ITEMS, f"{path}:{number}", "the tag names an item that waits on a rerun",
                         " ".join(RERUN_ITEMS) or "none", item)
            audit.record(False, f"{path}:{number}", f"[RERUN: item {item}] remains ({RERUN_ITEMS.get(item, 'unknown')})",
                         "no tag", "tag present")


def todo_check(audit: Audit, pre_submission: bool = False) -> None:
    """No source carries a todo marker or a hedge that defers a claim to a run not made."""
    for path in (TEX, SI, COVER):
        text = audit.text(path)
        live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
        for m in TODO_TAG.finditer(live):
            number = live[:m.start()].count("\n") + 1
            audit.record(False, f"{path}:{number}", f"[TODO: {m.group(1)}] remains "
                         f"({TODO_ITEMS.get(m.group(1), 'unknown')})", "no marker", "marker present")
        audit.cell(path, "todo markers", "0", float(len(TODO_TAG.findall(live))))
        audit.cell(path, "rerun tags", "0", float(len(rerun_tags(text))))
        flat = collapse(live)
        for hedge in HEDGES:
            found = flat.lower().count(hedge.lower())
            audit.record(found == 0, path, f"no hedge: {hedge}", "absent", f"{found} found")


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
    sym = audit.rows("paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv")
    calls = sorted(int(float(r["force_to_hit"])) for r in sym if r["hit"].strip().lower() in ("1", "true", "yes"))
    audit.record(len(calls) == len(sym), TEX, "symmetrized GMIN hits LJ75 on every seed", f"{len(sym)}", f"{len(calls)}")
    audit.said(TEX, f"With symmetrization and the published LJ\\(_{{75}}\\) input, from other starts, GMIN hit "
                    f"LJ\\(_{{75}}\\) on {len(calls)} of {len(sym)} seeds in a median of {int(median(calls))} force calls, "
                    "so the table compares these two configurations as run.")
    audit.said("paper/jctc/data/lj75_gmin/provenance.txt", "SYMMETRISE")
    audit.said(SI, "The committed \\path{paper/jctc/data/lj75_gmin/gmin-lj75-sym-starts-1-200.csv} campaign gave GMIN "
                   "the published LJ\\(_{75}\\) input with symmetrization and the OptBench starts.")
    audit.said(SI, "The Lennard-Jones rows run the Wales-group LJ\\(_{38}\\) introduction file at every Lennard-Jones size,")
    audit.said(SI, "The Morse rows add GMIN's native \\texttt{MORSE} keyword and \\(\\rho\\) to the same numerical settings.")
    audit.said(SI, "# GMIN: LJ/Morse use TEMPERATURE 1.0 and NEWRESTART 40")
    audit.said(SI, "with \\(T=1.0\\), \\texttt{NEWRESTART 40}, and an energy difference of 0.003")
    data = "examples/h2h_gmin_lj.data"
    for fragment in ("TEMPERATURE 1.0", "NEWRESTART 40", "EDIFF 0.003"):
        audit.said(data, fragment)
    audit.record("SYMMETRISE" not in audit.text(data), data, "the table input has no symmetrization keyword", "absent",
                 "absent" if "SYMMETRISE" not in audit.text(data) else "present")
    for gone in ("Au55 Gupta uses TEMPERATURE", "template-free", "Darby gold parameters"):
        for path in (TEX, SI):
            audit.record(gone not in audit.text(path), path, f"{gone} is not printed", "absent",
                         "absent" if gone not in audit.text(path) else "present")
    audit.record("Mackay cap" not in audit.text(TEX), TEX, "Mackay-cap disclosure prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(TEX) else "present")
    audit.record(not (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists(),
                 "src/methods/cluster_hopping/mackay_cap_tests.rs", "Mackay cap test module is removed",
                 "absent", "present" if (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists() else "absent")
    audit.said("paper/jctc/data/ase_rerun/morse/PROVENANCE", "source bench/ase-nonlj/rerun-20261119/rows.csv")
    audit.record("two-shell icosahedron" not in audit.text(TEX), TEX, "cap-disclosure prose is removed",
                 "absent", "present" if "two-shell icosahedron" in audit.text(TEX) else "absent")
    audit.record("largest cap" not in audit.text(TEX), TEX, "cap-size prose is removed",
                 "absent", "present" if "largest cap" in audit.text(TEX) else "absent")


def run(report: bool = False, write_map: bool = False, pre_submission: bool = False) -> int:
    problems = unauditable()
    if problems:
        for problem in problems:
            print(f"paper_audit: cannot check revisions: {problem}", file=sys.stderr)
        return 2
    audit = Audit()
    for check in (paired_statistics, manuscript_ablation, selection_tables, manuscript_retry, manuscript_gmin, manuscript_tip4p, manuscript_hexamer,
                  manuscript_cross_field, supporting_information, release, paper_0120, gad_published, openconf,
                  head_to_head, sc96_78_howto, stale_claims, tip4p_panel, covalent_panels, covalent_gfn2, ase_gofee, lj_large, bench_prose,
                  manuscript_ase, manuscript_knockouts, manuscript_priors, panel_references, generated_tables, supporting_gmin, supporting_numbers, manuscript_new, manuscript_followups, cover_letter, science_claims):
        check(audit)
    if write_map:
        (ROOT / REVISION_MAP).write_text(revision_map_text(revision_map_rows(audit)))
    revision_map(audit)
    rerun_claims(audit)
    rerun_check(audit, pre_submission)
    todo_check(audit, pre_submission)
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
                        help="fail while any [RERUN: item N] tag or [TODO: Gn] marker remains")
    args = parser.parse_args(argv)
    return run(args.report, args.write_revision_map, args.pre_submission)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
