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
without them, or whose revision is not on a pushed branch, must be listed in
``KNOWN_GAPS``. Whether a cited revision is pushed is decided by reachability:
a ref fetched from the remote contains it; or it was rebuilt on push and its
alias in ``ALIASES`` verifies; or the ``code_tree`` its rows record is the tree
of a commit on the remote. A revision none of these resolves is listed in
``UNPUSHED_FILE`` with the reason, and the list is checked against the
reachability, not the other way round. Printed numbers no
committed row backs are listed in ``UNTRACED``; their text must still be where
the list says. The revision checks need the full history and the remote
branches; a clone without them is refused, with exit status 2.

    python3 scripts/paper_audit.py            # exit 1 on any drift
    python3 scripts/paper_audit.py --report   # also print every check
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import functools
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import struct
import subprocess
import tarfile
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from codata2018 import BOLTZMANN_EV_PER_K, KJ_PER_KCAL  # noqa: E402
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402

TEX = "paper/jctc/manuscript.tex"
SI = "paper/jctc/si.tex"
# The manuscript followed by the Supporting Information: a claim is checked wherever the two place it.
PAPER = "paper/jctc/manuscript.tex+si.tex"

REVISION_COLUMNS = (
    "code_revision", "anneal_head", "anneal_sha",
    "release_revision", "head", "harness_revision",
)
HOST_COLUMNS = ("host", "machine_id")
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = {"aac1548a8aec13b53ad12a8d6904625081685d2e": "GMIN source/SVNREV"}
# Local revisions that rows cite, each with the pushed commit it was rebuilt as.
ALIASES = "bench/revision-aliases.csv"
# Cited revisions that no remote ref contains, no verified alias covers and no
# recorded code_tree resolves, each with the reason. The file only explains an
# unresolved revision: reachability decides which revisions are unresolved.
UNPUSHED_FILE = "bench/unpushed-revisions.csv"
# Row groups whose revision column is empty or "unknown", with why each cannot be recovered.
UNRECORDED_FILE = "bench/unrecorded-revisions.csv"
UNRECORDED = "unrecorded"
# The column that holds the tree of the commit a revision column names.
TREE_COLUMNS = {"code_revision": "code_tree"}

KNOWN_GAPS = {
    "data/sc96_78_hits.csv": "code_revision unrecorded on some rows; harness_revision unrecorded on some rows; host "
                             "empty or unknown on some rows",
    "bench/depth-posterior-ablation/results.csv": "revision unrecorded (no revision column)",
    "results/size-tiered-selection.csv": "revision and host unrecorded (no revision or host column)",
    "results/size-tiered-selection-provenance.csv": "code_revision unrecorded on some rows",
    "results/lj-funnel-stalls.csv": "revision and host unrecorded (no revision or host column)",
    "results/cross-field-hits.csv": "revision and host unrecorded (no revision or host column)",
    "bench/tip4p-water-rates/rates-a24ba760.csv": "revision unrecorded (no revision column)",
    "bench/tip4p-water-rates/rates.csv": "revision unrecorded (no revision column)",
    "paper/jctc/data/system_visuals/references.csv": "revision and host unrecorded (no revision or host column)",
    "bench/gad-published/summary.csv": "revision recorded; no host column",
    "bench/morse-panel/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/morse-h2h/seeds.csv": "revision and host unrecorded (no revision or host column)",
    "bench/fp-fingerprint/rows.csv": "revision and host unrecorded (no revision or host column)",
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
    "bench/ase-gofee-h2h/lj-large-seeds.csv": "revision 89de725a is not in this repository",
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
    "paper/jctc/data/literature.tsv": "revision and host unrecorded (no revision or host column)",
}

UNTRACED = []


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, str, str]] = []
        self.read: dict[str, list[dict[str, str]]] = {}
        self.text_cache: dict[str, str] = {}
        self.revisions: Revisions | None = None
        self.revisions_key: tuple[str, ...] | None = None

    def record(self, ok: bool, where: str, label: str, shown: object, value: object) -> None:
        self.results.append(("OK" if ok else "DRIFT", where, label, str(shown), str(value)))

    def rows(self, path: str, delimiter: str = ",") -> list[dict[str, str]]:
        if path not in self.read:
            with (ROOT / path).open(newline="") as handle:
                lines = [line for line in handle if line.strip() and not line.startswith("#")]
            self.read[path] = list(csv.DictReader(lines, delimiter=delimiter))
        return self.read[path]

    def text(self, path: str) -> str:
        if path == PAPER:
            return self.text(TEX) + "\n" + self.text(SI)
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


_COMMIT_HELD: dict[str, bool] = {}


def commit_held(rev: str) -> bool:
    """The revision is a commit object in this repository."""
    if rev not in _COMMIT_HELD:
        _COMMIT_HELD[rev] = git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0
    return _COMMIT_HELD[rev]


def panel_rows_held(rows: list[dict[str, str]], *, anneal: bool) -> bool:
    """Anneal rows need their code and harness commits. GMIN rows need the harness commit."""
    if not rows:
        return False
    for row in rows:
        if not commit_held(row["harness_revision"]):
            return False
        if anneal and not commit_held(row["code_revision"]):
            return False
    return True


def quoted_systems(panel: object, groups: dict) -> list[str]:
    """Panel systems whose scored rows name commits this repository holds."""
    return [
        system for system in panel.PAPER_BUDGETS
        if panel_rows_held(groups[(system, "anneal")], anneal=True)
        and panel_rows_held(groups[(system, "gmin")], anneal=False)
    ]


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


def mann_whitney_exact_no_ties(a: list[float], b: list[float]) -> float:
    """Exact two-sided rank-sum tail for two samples whose pooled values do not tie."""
    if len(set(a + b)) != len(a) + len(b):
        raise ValueError("exact no-ties Mann-Whitney received a tie")
    n_a, n_b = len(a), len(b)
    observed = sum(x > y for x in a for y in b)
    smaller = min(observed, n_a * n_b - observed)
    n = n_a + n_b
    max_sum = n_a * (2 * n - n_a + 1) // 2
    ways = [[0] * (max_sum + 1) for _ in range(n_a + 1)]
    ways[0][0] = 1
    for rank in range(1, n + 1):
        for count in range(min(n_a, rank), 0, -1):
            for total in range(max_sum, rank - 1, -1):
                ways[count][total] += ways[count - 1][total - rank]
    base = n_a * (n_a + 1) / 2
    extreme = sum(count for total, count in enumerate(ways[n_a])
                  if min(total - base, n_a * n_b - (total - base)) <= smaller)
    return extreme / math.comb(n, n_a)


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
    provenance = audit.rows("results/size-tiered-selection-provenance.csv")
    row_keys = {(r["arm"], r["n"], r["seed"]) for r in second}
    provenance_keys = {(r["arm"], r["n"], r["seed"]) for r in provenance}
    audit.record(row_keys == provenance_keys and len(second) == len(provenance),
                 "results/size-tiered-selection-provenance.csv",
                 "each held-out result has one sidecar provenance row",
                 f"{len(second)} rows", f"{len(provenance)} rows, keys {'match' if row_keys == provenance_keys else 'differ'}")
    budgets = {38: 400000, 55: 1000000, 75: 4000000, 98: 4000000, 104: 4000000}
    counts = {}
    mw_p = {}
    for row in tex_table(audit, PAPER, "tab:ablation"):
        size = CLUSTER[row[0]]
        arms = {}
        for arm in ("posterior", "uniform"):
            group = pick(first, system=f"lj{size}", arm=arm)
            if size >= 75:
                group += pick(second, n=str(size), arm=arm)
            arms[arm] = group
            for r in group:
                audit.cell(PAPER, f"ablation lj{size} {arm} seed {r['seed']} budget", str(budgets[size]), float(r.get("budget") or budgets[size]))
        where = f"{PAPER} tab:ablation lj{size}"
        audit.cell(where, "budget", row[1], budgets[size])
        audit.cell(where, "posterior", row[2], frac(arms["posterior"]))
        audit.cell(where, "uniform", row[3], frac(arms["uniform"]))
        p = fisher(hits_of(arms["posterior"]), len(arms["posterior"]), hits_of(arms["uniform"]), len(arms["uniform"]))
        audit.cell(where, "Fisher p", row[4], p)
        counts[size] = (hits_of(arms["posterior"]), len(arms["posterior"]), hits_of(arms["uniform"]), len(arms["uniform"]))

        def calls(group: list[dict]) -> list[float]:
            return [float(r.get("force_calls_to_hit") or r.get("calls_to_hit")) if hits_of([r]) else float(budgets[size])
                    for r in group]

        mw_p[size] = mann_whitney(calls(arms["posterior"]), calls(arms["uniform"]), continuity=True)
    for size, want in zip(sorted(mw_p), ("0.23", "0.47", "0.80", "0.68", "0.83")):
        audit.cell(PAPER, f"ablation lj{size} Mann-Whitney p on calls, a miss at the budget", want, mw_p[size])
    audit.said(PAPER, "The two-sided continuity-corrected asymptotic Mann--Whitney test on the calls\n"
                      "to the minimum, with a miss at the full budget, gives \\(p\\) of 0.23, 0.47,\n"
                      "0.80, 0.68, and 0.83 from LJ\\(_{38}\\) to LJ\\(_{104}\\).")
    audit.cell(PAPER, "abstract: the arms hit the same number of seeds to within one", "1",
               float(max(abs(c[0] - c[2]) for c in counts.values())))
    audit.cell(PAPER, "Fisher p of 12 of 12 against 8 of 12", "0.093", fisher(12, 12, 8, 12))
    audit.cell(PAPER, "Fisher p of 12 of 12 against 7 of 12", "0.037", fisher(12, 12, 7, 12))
    snakefile = audit.text("workflow/Snakefile")
    audit.record("bench/depth-posterior-ablation/results.csv" in snakefile
                 and "rule size_tiered_selection:" in snakefile
                 and "enter_bench.sh --held-out" in snakefile, "workflow/Snakefile", "a rule writes each ablation file",
                 "both", "both")
    audit.record("--results \"$ROOT/results/size-tiered-selection.csv\"" in audit.text(
        "scripts/depth_posterior_ablation/enter_bench.sh"), "scripts/depth_posterior_ablation/enter_bench.sh",
        "the held-out rule appends to the held-out file", "results/size-tiered-selection.csv", "checked")
    audit.said(PAPER, "\\texttt{size\\_tiered\\_selection} rule of \\path{workflow/Snakefile} appends the\ntwo arms and puts their code revision and host in\n\\path{results/size-tiered-selection-provenance.csv}.")
    committed = [r for r in provenance if r["code_revision"] == "unknown"]
    audit.cell(PAPER, "the committed held-out rows: host and revision", "bench unknown 54",
               f"{' '.join(sorted({r['host'] for r in committed}))} {' '.join(sorted({r['code_revision'] for r in committed}))} "
               f"{len(committed)}")
    listed = {r["arm"] for r in committed_revision_map()
              if r["row_file"] == "results/size-tiered-selection-provenance.csv" and r["revision"] == UNRECORDED}
    audit.cell(PAPER, "the unrecorded held-out arms in the revision map", "posterior tiered uniform", " ".join(sorted(listed)))
    audit.said(
        PAPER,
        "The committed rows record host \\texttt{bench}.\n"
        "\\path{bench/unreachable-revisions.csv} indexes the same rows.",
    )
    p75, p98, p104 = counts[75], counts[98], counts[104]
    audit.said(PAPER, f"the posterior hits LJ\\(_{{75}}\\) on {p75[0]} of {p75[1]} and the uniform draw hits {p75[2]} of {p75[3]}.")
    audit.said(PAPER, f"Both hit LJ\\(_{{98}}\\) on {p98[0]} of {p98[1]}.")
    audit.said(PAPER, f"The posterior hits LJ\\(_{{104}}\\) on {p104[0]} of {p104[1]} and the uniform draw hits {p104[2]} of {p104[3]}.")
    audit.said(PAPER, f"LJ\\(_{{75}}\\) is {p75[0]} of {p75[1]} against {p75[2]} of {p75[3]},")
    audit.said(PAPER, f"LJ\\(_{{104}}\\) is {p104[0]} of {p104[1]} against {p104[2]} of {p104[3]}.")


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
    audit.said(PAPER, f"LJ\\(_{{98}}\\) hits on {frac(retry98).replace('/', ' of ')} inside \\(4\\times 10^{{6}}\\) calls.")
    audit.said(PAPER, f"the eighth-leave retry hits {frac(after98).replace('/', ' of ')}.")
    miss = [r for r in after98 if r["hit"] == "0"]
    audit.cell(PAPER, "retry: the hold-only miss is seed 6404", "6404", miss[0]["seed"] if len(miss) == 1 else None)
    audit.said(PAPER, f"Seed 6404 stays at a gap of \\({miss[0]['gap']}\\) above \\(-543.665361\\).")
    for size in ("38", "55", "75"):
        audit.cell(PAPER, f"retry off below 98: lj{size} on 6400-6405", "6/6", frac(pick(rows, phase="after", n=size)))
    retry104 = pick(rows, phase="retry", n="104")
    audit.said(PAPER, f"The retry rows stored for that leave hit {hits_of(retry104)} of {len(retry104)} on LJ\\(_{{104}}\\)")
    audit.said(PAPER, f"{frac(retry98).replace('/', ' of ')} on LJ\\(_{{98}}\\).")


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


def _between(text: str, start: str, end: str) -> str:
    begin = text.find(start)
    if begin < 0:
        return ""
    stop = text.find(end, begin + len(start))
    return text[begin:] if stop < 0 else text[begin:stop]


def _sentences(text: str) -> list[str]:
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.S)
    flat, found, start = collapse(prose), [], 0
    for end in [match.end() for match in SENTENCE_END.finditer(flat)] + [len(flat)]:
        sentence, start = flat[start:end].strip(), end
        if sentence:
            found.append(sentence)
    return found


def _system_named(sentence: str, system: str) -> bool:
    if system == "au55":
        return "Au\\(_{55}\\)" in sentence
    if system.startswith("lj"):
        return f"LJ\\(_{{{system[2:]}}}\\)" in sentence
    cluster = "55" if system.startswith("m55") else "80"
    rho = "6" if system.endswith("r6") else "14"
    return f"M\\(_{{{cluster}}}\\)" in sentence and f"\\rho={rho}" in sentence


_GENERIC_TOKENS = {"0", "1", "0.0", "0.00", "0.000"}
HELD_NUMBER = "a printed number traces to a commit this repository holds"


def _ase_sentence(sentence: str) -> bool:
    return "Atomic Simulation" in sentence or re.search(r"\bASE\b", sentence) is not None or "tab:ase" in sentence


def _result_tokens(panel: object, groups: dict, system: str) -> set[str]:
    """Formats in which this system's GMIN-panel results are printed."""
    tokens = {"loses that row"}
    for arm in ("anneal", "gmin"):
        rows = groups[(system, arm)]
        hits, n = hits_of(rows), len(rows)
        calls = [float(row["evals_to_hit"]) for row in rows]
        tokens.update({
            f"{hits}/{n}",
            f"{hits} of {n}",
            panel.show_calls(median(calls)),
            panel.show_calls(iqr(calls)),
            panel.show_s(median([float(row["wall_s"]) for row in rows])),
            panel.show_s(median([float(row["pot_s"]) for row in rows])),
            panel.show_s(median([float(row["outside_s"]) for row in rows])),
        })
    anneal, gmin = groups[(system, "anneal")], groups[(system, "gmin")]
    budget = panel.PAPER_BUDGETS[system]
    p_fisher = fisher(hits_of(anneal), len(anneal), hits_of(gmin), len(gmin))
    tokens.add(panel.show_p(p_fisher, tex=True))
    tokens.add(panel.show_p(p_fisher, tex=False))
    rc_a, rc_g = gmin_ranked_calls(anneal, budget), gmin_ranked_calls(gmin, budget)
    if rc_a is not None and rc_g is not None:
        p_calls = mann_whitney(rc_a, rc_g, continuity=True)
        if p_calls is not None:
            tokens.add(panel.show_p(p_calls, tex=True))
            tokens.add(panel.show_p(p_calls, tex=False))
    rw = gmin_ranked_wall(anneal, gmin)
    if rw is not None:
        p_wall = mann_whitney(*rw, continuity=True)
        if p_wall is not None:
            tokens.add(panel.show_p(p_wall, tex=True))
            tokens.add(panel.show_p(p_wall, tex=False))
    return {token for token in tokens if token not in _GENERIC_TOKENS}


def _gmin_quote_spans(audit: Audit) -> list[tuple[str, str]]:
    """The passages that quote the GMIN panel: abstract, introduction, results, discussion, conclusions, SI, cover."""
    tex, si = audit.text(TEX), audit.text(SI)
    return [
        (TEX, _between(tex, r"\begin{abstract}", r"\end{abstract}")),
        (TEX, _between(tex, r"\section{Introduction}", r"\section{")),
        (TEX, _between(tex, r"\label{sec:gmin}", r"\label{sec:sc}")),
        (TEX, _between(tex, r"\section{Discussion}", r"\section{Conclusions}")),
        (TEX, _between(tex, r"\section{Conclusions}", r"\section*{Author contributions}")),
        (SI, _between(si, r"\label{sec:si-gmin}", r"\section{Sutton")),
        (COVER, audit.text(COVER)),
    ]


def _gmin_caption(audit: Audit) -> str:
    text = audit.text(TEX)
    start = text.find("% gmin-panel table: start")
    label = text.find(r"\label{tab:gmin}", start if start >= 0 else 0)
    if start < 0 or label < 0:
        return ""
    return text[start:label]


_UPSTREAM_GMIN = "aac1548a8aec13b53ad12a8d6904625081685d2e"
_REVISION_COLUMNS = frozenset({
    "code_revision", "harness_revision", "anneal_sha", "anneal_head", "anneal_revision", "head",
})


def unheld_revision_prefixes(audit: Audit) -> set[str]:
    """Eight-character prefixes of anneal or harness commits this repository does not hold.

    The upstream GMIN revision is not an anneal commit. A row may cite it beside a harness this
    repository does hold.
    """
    prefixes: set[str] = set()
    for path in (ROOT / "bench").rglob("*.csv"):
        try:
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(line for line in handle if line.strip() and not line.startswith("#")))
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
        for row in rows:
            for column, value in row.items():
                if column not in _REVISION_COLUMNS or not re.fullmatch(r"[0-9a-f]{8,40}", value or ""):
                    continue
                if value == _UPSTREAM_GMIN or _UPSTREAM_GMIN.startswith(value):
                    continue
                if not commit_held(value):
                    prefixes.add(value[:8])
    return prefixes


def _window_has_token(text: str, label: str, tokens: set[str]) -> str | None:
    flat = collapse(text)
    for match in re.finditer(re.escape(label), flat):
        window = flat[max(0, match.start() - 160):match.end() + 220]
        for token in sorted(tokens, key=len, reverse=True):
            if re.search(rf"(?<!\d){re.escape(token)}", window):
                return token
    return None


def unheld_bench_quotes(audit: Audit) -> None:
    """Every printed number from a bench record traces to commits this repository holds.

    The GMIN panel has its own span check. This one covers the other records the paper quotes:
    ASE, CREST, AGOX, GOFEE, TIP4P, Sutton--Chen, and the Morse panel, and any printed revision.
    """
    documents = "\n".join(audit.text(path) for path in (TEX, SI, COVER))
    for prefix in sorted(unheld_revision_prefixes(audit)):
        if prefix in documents:
            audit.record(False, PAPER, HELD_NUMBER, "absent", prefix)
    panel = load_script("ase_quote_panel", "scripts/ase_panel/summarize.py")
    groups = ase_groups(audit)
    ase_span = _between(audit.text(SI), r"\label{sec:ase}", r"\section{AGOX")
    ase_span += "\n" + "\n".join(
        sentence for sentence in _sentences(documents) if _ase_sentence(sentence)
    )
    for system in panel.SYSTEMS:
        arms = ("recommended", "ase_bh", "ase_mh", "anneal")
        rows = [row for arm in arms for row in groups.get((system, arm), [])]
        if not rows or all(commit_held(row["code_revision"]) for row in rows):
            continue
        tokens = set()
        for arm in arms:
            arm_rows = groups.get((system, arm), [])
            if not arm_rows:
                continue
            tokens.add(frac(arm_rows).replace(" ", ""))
            tokens.add(frac(arm_rows))
            tokens.add(f"{hits_of(arm_rows)} of {len(arm_rows)}")
        hit = _window_has_token(ase_span, panel.TEX_LABELS[system], tokens)
        if hit:
            audit.record(False, f"{SI} {system}", HELD_NUMBER, "absent", hit)
    h2h = load_script("h2h_quote", "paper/jctc/scripts/h2h_tables.py")
    if not h2h.sc_gmin_quotable():
        sc_span = _between(audit.text(TEX), r"\label{sec:sc}", r"\label{sec:other-panels}")
        sc_span += "\n" + _between(audit.text(SI), r"\section{Sutton", r"\section{The Atomic")
        for anneal, gmin in h2h.sc_pairs():
            if commit_held(gmin["harness_revision"]):
                continue
            wall = h2h.show_s(float(gmin["wall_median_s"]))
            anneal_wall = h2h.show_s(float(anneal["wall_median_s"]))
            tokens = {wall, f"{gmin['hits']}/12", f"{gmin['hits']} of 12"} - {anneal_wall, f"{anneal['hits']}/12",
                                                                              f"{anneal['hits']} of 12"}
            hit = _window_has_token(sc_span, h2h.sc_label(anneal["system"]), tokens)
            if hit:
                audit.record(False, f"{TEX} {anneal['system']}", HELD_NUMBER, "absent", hit)
    gofee_rows = audit.rows("bench/ase-gofee-h2h/results.csv")
    if gofee_rows and any(not commit_held(row["head"]) for row in gofee_rows):
        summary = {(row["system"], row["arm"]): row for row in audit.rows("bench/ase-gofee-h2h/summary.csv")}
        gofee_span = _between(audit.text(SI), r"\label{sec:gofee}", r"\section{")
        gofee_span += "\n" + "\n".join(sentence for sentence in _sentences(documents) if "GOFEE" in sentence)
        labels = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj75": "LJ\\(_{75}\\)"}
        for system, label in labels.items():
            tokens = set()
            for arm in ("paper", "ase_bh", "ase_mh", "agox_gofee"):
                row = summary.get((system, arm))
                if row is None:
                    continue
                tokens.add(f"{row['hits']}/{row['n']}")
                tokens.add(f"{row['hits']} of {row['n']}")
                tokens.add(f"{float(row['wall_median_s']):.1f}")
            hit = _window_has_token(gofee_span, label, tokens)
            if hit:
                audit.record(False, f"{SI} {system}", HELD_NUMBER, "absent", hit)
    if not any(row[2] == HELD_NUMBER and row[0] == "DRIFT" for row in audit.results):
        audit.record(True, PAPER, HELD_NUMBER, "absent", "absent")


def unheld_panel_numbers(audit: Audit, panel: object, groups: dict) -> None:
    """A printed GMIN-panel count, call total, wall, or test comes from a commit this repository holds.

    A mark around the number does not exempt it.
    """
    quoted = set(quoted_systems(panel, groups))
    unquoted = [system for system in panel.PAPER_BUDGETS if system not in quoted]
    labels = {panel.LABELS[system][1]: system for system in unquoted}
    quoted_in_table: set[str] = set()
    try:
        table_rows = tex_table(audit, TEX, "tab:gmin")
    except ValueError:
        table_rows = []
    current = ""
    for cells in table_rows:
        if cells and cells[0]:
            current = cells[0]
        system = next((name for label, name in labels.items() if label in current), None)
        if system is None:
            continue
        quoted_in_table.add(system)
        audit.record(False, f"{TEX} tab:gmin {system}", HELD_NUMBER, "absent", cells[3] if len(cells) > 3 else current)
    caption = _gmin_caption(audit)
    for system in unquoted:
        revision = groups[(system, "anneal")][0]["code_revision"][:8]
        if revision and revision in caption:
            quoted_in_table.add(system)
            audit.record(False, f"{TEX} tab:gmin {system}", HELD_NUMBER, "absent", revision)
    for system in unquoted:
        tokens = _result_tokens(panel, groups, system)
        found = False
        for path, text in _gmin_quote_spans(audit):
            for sentence in _sentences(text):
                if _ase_sentence(sentence) or not _system_named(sentence, system):
                    continue
                hit = sorted(token for token in tokens if re.search(rf"(?<!\d){re.escape(token)}", sentence))
                if not hit:
                    continue
                found = True
                audit.record(False, f"{path} {system}", HELD_NUMBER, "absent", hit[0])
        if not found and system not in quoted_in_table:
            audit.record(True, f"{TEX} {system}", HELD_NUMBER, "absent", "absent")
    unheld_bench_quotes(audit)


def gmin_panel_line(audit: Audit, where: str, cells: list[str], groups: dict, system: str, arm: str,
                    budget: int) -> None:
    """One table line: Budget, Arm, Hits, Median calls, IQR, Wall, Inside, Outside, Fisher p, MW p calls, MW p wall."""
    group = groups[(system, arm)]
    other = groups[(system, "gmin" if arm == "anneal" else "anneal")]

    def col(rows: list[dict[str, str]], key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    audit.cell(where, "hits", cells[2], frac(group))
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
        (PAPER, 1, [(None, row) for row in tex_table(audit, PAPER, "tab:gmin")]),
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
        source = quoted_systems(panel, groups) if path == PAPER else list(panel.PAPER_BUDGETS)
        want = [(system, arm) for system in source for arm in ("anneal", "gmin")]
        audit.cell(path, "gmin panel lines", " ".join(f"{s}/{a}" for s, a in shown), " ".join(f"{s}/{a}" for s, a in want))
    gmin_panel_readme(audit, panel, groups)
    gmin_panel_prose(audit, panel, groups)
    unheld_panel_numbers(audit, panel, groups)


def gmin_cluster_account(panel: object, groups: dict) -> list[str]:
    """One sentence group per quoted cluster, in panel order, from the committed rows."""
    sentences = []
    for system in quoted_systems(panel, groups):
        anneal = groups[(system, "anneal")]
        gmin = groups[(system, "gmin")]
        name = panel.LABELS[system][1]

        def med(rows: list[dict[str, str]], key: str) -> float:
            return median([float(r[key]) for r in rows])

        def spread(rows: list[dict[str, str]]) -> float:
            return iqr([float(r["evals_to_hit"]) for r in rows])

        fisher_p = fisher(hits_of(anneal), len(anneal), hits_of(gmin), len(gmin))
        calls = panel.ranked_calls(anneal, panel.PAPER_BUDGETS[system])
        gmin_calls = panel.ranked_calls(gmin, panel.PAPER_BUDGETS[system])
        calls_p = panel.mann_whitney(calls, gmin_calls)[1] if calls is not None and gmin_calls is not None else None
        walls = panel.ranked_wall(anneal, gmin)
        wall_p = panel.mann_whitney(*walls)[1] if walls is not None else None
        test = f"Fisher's exact \\(p\\) on the hits is {panel.show_p(fisher_p, True)}"
        if calls_p is not None:
            test += f", and the Mann--Whitney \\(p\\) on calls is {panel.show_p(calls_p, True)}"
            if wall_p is not None:
                test += f", and on wall {panel.show_p(wall_p, True)}"
        sentences.append(
            f"On {name} this setting hits {hits_of(anneal)} of {len(anneal)} and GMIN hits "
            f"{hits_of(gmin)} of {len(gmin)}. "
            f"The median calls are {panel.show_calls(med(anneal, 'evals_to_hit'))} against "
            f"{panel.show_calls(med(gmin, 'evals_to_hit'))}, "
            f"and the interquartile ranges of those calls are {panel.show_calls(spread(anneal))} and "
            f"{panel.show_calls(spread(gmin))}. "
            f"The median wall is {panel.show_s(med(anneal, 'wall_s'))}~s against "
            f"{panel.show_s(med(gmin, 'wall_s'))}~s, "
            f"the median time inside the potential is {panel.show_s(med(anneal, 'pot_s'))}~s against "
            f"{panel.show_s(med(gmin, 'pot_s'))}~s, "
            f"and the median time outside the potential is {panel.show_s(med(anneal, 'outside_s'))}~s against "
            f"{panel.show_s(med(gmin, 'outside_s'))}~s. "
            f"{test}."
        )
    return sentences


def gmin_panel_prose(audit: Audit, panel: object, groups: dict) -> None:
    def name(system: str) -> str:
        label = panel.LABELS[system][1]
        return label.replace(", \\(\\rho=", " at \\(\\rho=")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(groups[(system, arm)])} of {len(groups[(system, arm)])}"

    references = {row[0]: row[3] for row in panel.SYSTEMS}
    systems = quoted_systems(panel, groups)
    ahead = [s for s in systems if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")])]
    behind = [s for s in systems if hits_of(groups[(s, "anneal")]) < hits_of(groups[(s, "gmin")])]
    tied = [s for s in systems if s not in ahead and s not in behind]
    significant = [s for s in systems if fisher(hits_of(groups[(s, "anneal")]), len(groups[(s, "anneal")]),
                                                hits_of(groups[(s, "gmin")]), len(groups[(s, "gmin")])) < 0.05]
    total = len(systems)
    audit.said(PAPER, f"this setting hits\nmore seeds on {len(ahead)} of {total} and GMIN hits more on {spelled(len(behind))}.")

    def ps(system: str) -> str:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return panel.show_p(fisher(hits_of(a), len(a), hits_of(g), len(g)), tex=True).replace("\\(", "").replace("\\)", "")

    audit.said(PAPER, f"This setting hits more seeds than GMIN on {len(ahead)} of the {total} clusters.")
    seed_span = f"{panel.SEEDS[0]}--{panel.SEEDS[-1]}"
    rad = (2.0 + (3.0 * 55 / 17.77153175) ** (1.0 / 3.0)) * 2.884
    by_revision: dict[str, list[str]] = {}
    for s in systems:
        for rev in sorted({r["code_revision"][:8] for r in groups[(s, "anneal")]}):
            by_revision.setdefault(rev, []).append(panel.LABELS[s][1].replace(", ", " at "))
    shown_rev = "; ".join(f"\\texttt{{{rev}}} on {series(names)}" for rev, names in by_revision.items())
    audit.said(PAPER, f"Revisions: anneal {shown_rev}, GMIN \\texttt{{{panel.GMIN_REV[:8]}}}.")
    audit.said(PAPER, f"Seeds {seed_span}, one core, budget in force calls.")
    audit.said(PAPER, f"Both arms use seeds {seed_span}, the budget in the table and the start this code writes for the seed.")
    audit.said(PAPER, f"GMIN's nearest-neighbor container \\({rad:.6f}\\)~\\AA, step and taboo distances scaled by "
                    "\\(r_0/2^{1/6}\\) (\\(r_0=2.884\\)~\\AA), and \\(T=0.1\\)~eV from the Wales Gupta example.")
    audit.said(PAPER,
               f"\\({references['lj102']:.6f}\\) at LJ\\(_{{102}}\\), "
               f"\\({references['lj103']:.6f}\\) at LJ\\(_{{103}}\\), "
               f"\\({references['lj150']:.6f}\\) at LJ\\(_{{150}}\\), "
               f"\\({references['au55']:.8f}\\)~eV at Au\\(_{{55}}\\), "
               f"\\({references['m55r6']:.6f}\\) and \\({references['m55r14']:.6f}\\) at M\\(_{{55}}\\) (\\(\\rho=6,14\\)), "
               f"and \\({references['m80r6']:.6f}\\) and \\({references['m80r14']:.6f}\\) at M\\(_{{80}}\\) (\\(\\rho=6,14\\)).")
    audit.said(
        PAPER,
        "The hit is a quenched energy at or below the Cambridge Cluster Database minimum plus 0.001:",
    )
    audit.said(PAPER, "GMIN hits more on " + ", and on ".join(f"{name(s)}, {of(s, 'gmin')} against {of(s, 'anneal')}"
                                                       for s in behind) + ".")
    audit.said(PAPER, f"The two tie on {' and '.join(name(s) for s in tied)}.")
    ours = [s for s in significant if s in ahead]
    theirs = [s for s in significant if s in behind]
    if theirs:
        audit.said(PAPER, f"Fisher's test on hits is below 0.05 on {len(significant)} clusters: {len(ours)} for this work "
                        f"and {' and '.join(name(s) for s in theirs)} for GMIN.")
    else:
        audit.said(PAPER, f"Fisher's test on hits is below 0.05 on {len(significant)} clusters, all for this work.")
    counts = " and ".join(f"{of(s, 'gmin')} against {of(s, 'anneal')}" for s in behind)
    audit.said(PAPER, f"Against GMIN on seeds {seed_span}, this setting hits more seeds on {len(ahead)} of {total} clusters, "
                    f"and GMIN hits more on {' and on '.join(name(s) for s in behind)}, {counts}.")
    audit.said(PAPER, "Section~\\ref{sec:gmin} scores it on the clusters of Table~\\ref{tab:gmin}.")
    audit.said(PAPER, f"GMIN hits more seeds on {len(behind)} of the {total}, {series(name(s) for s in behind)}.")

    def show_p(p: float) -> str:
        return panel.show_p(p, tex=True)

    def p_of(system: str) -> float:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return fisher(hits_of(a), len(a), hits_of(g), len(g))

    def col(system: str, arm: str, key: str) -> float:
        return median([float(r[key]) for r in groups[(system, arm)]])

    for system in behind:
        budget = panel.PAPER_BUDGETS[system]
        audit.cell(PAPER, f"{system}: this setting's median calls are the full budget", str(budget),
                   col(system, "anneal", "evals_to_hit"))
        audit.cell(PAPER, f"{system}: the full budget is 4e6", "4000000", float(budget))
    audit.cell(PAPER, "the clusters where GMIN hits more", " ".join(behind), " ".join(behind))
    if "m80r6" in behind:
        audit.said(PAPER, f"On {name('m80r6')} GMIN reaches the minimum in a median of "
                        f"{panel.show_calls(col('m80r6', 'gmin', 'evals_to_hit'))} calls and "
                        f"{panel.show_s(col('m80r6', 'gmin', 'wall_s'))}~s, and this setting spends the full "
                        f"\\(4\\times 10^{{6}}\\) calls in a median of {panel.show_s(col('m80r6', 'anneal', 'wall_s'))}~s.")
    lj150 = "lj150"
    audit.said(PAPER, f"On {name(lj150)} the median calls are {panel.show_calls(col(lj150, 'gmin', 'evals_to_hit'))} for "
                    f"GMIN and the full \\(4\\times 10^{{6}}\\) for this setting, and Fisher's \\(p\\) is "
                    f"{show_p(p_of(lj150))}.")
    audit.said(PAPER, f"GMIN hits more seeds than this setting on {len(behind)} of the {total} clusters of "
                    "Table~\\ref{tab:gmin}.")
    for system in behind:
        audit.said(PAPER, f"On {name(system)} GMIN hits {of(system, 'gmin')} and this setting {of(system, 'anneal')}, "
                        f"and Fisher's \\(p\\) is {show_p(p_of(system))}.")
    both = {f"{hits_of(groups[(s, 'anneal')])}/{len(groups[(s, 'anneal')])}" for s in tied}
    audit.cell(PAPER, "the tied clusters share one count", "1", float(len(both)))
    audit.said(PAPER, f"The two tie on {' and '.join(name(s) for s in tied)}, where both hit "
                    f"{next(iter(both)).replace('/', ' of ')}.")
    for sentence in gmin_cluster_account(panel, groups):
        audit.said(PAPER, sentence)
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
    audit.said(PAPER, "The gold driver gives it the copper--gold species list with no copper atom, applies the basin "
                    "bias of equation~\\msref{eq:delta} to every downhill or accepted proposal and to one other proposal "
                    f"in {stride.group(1) if stride else '?'}, and decides the rest on the bare energy.")
    audit.record("Mackay cap" not in audit.text(PAPER), PAPER, "Mackay-cap driver prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(PAPER) else "present")


ASE_PANEL = ("lj38", "lj55", "lj98", "lj104", "lj150", "au55")
MORSE_PANEL = ("m38r6", "m38r14", "m55r6", "m55r14")
MORSE_REVISION = "9a30f3190b3bb7040c675a904f8388eb68224151"
ASE_ARMS = ("ase_bh", "ase_mh")
PANEL_SEEDS = " ".join(str(s) for s in range(2000, 2012))


def load_script(name: str, path: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
    audit.said(PAPER, "on seeds 2000--2011, at \\(4\\times 10^{5}\\) calls for 38 atoms and \\(1\\times 10^{6}\\) for 55, "
                    "on one core of one host.")
    audit.said("scripts/ase_panel/run_one.py", 'ASE_VERSION = "3.29.0"')
    audit.said("bench/ase-nonlj/README.md", "ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults")
    audit.said(PAPER, "Environment 3.29.0 at its constructor defaults")
    audit.said("scripts/ase_panel/run_one.py", "Au55 is Gupta::copper_gold through asg_eval serve, the same potential "
                                               "the anneal arms evaluate.")

    # tab:ase and the combined recommended summary. The table is absent when a quoted arm
    # cites a commit this repository does not hold; the summary files stay.
    revision = sorted({r["code_revision"] for s in ASE_PANEL for r in groups[(s, "recommended")]})
    audit.record(len(revision) >= 1, PAPER, "tab:ase recommended revisions", "at least one", str(len(revision)))
    ase_held = bool(panel.tex_table())
    by_label = {label: s for s, label in panel.TEX_LABELS.items()}
    shown = []
    if ase_held:
        rev_ase_text = ", ".join(revision)
        audit.said(PAPER, f"This work is anneal \\texttt{{{rev_ase_text}}}.")
    else:
        present = "\\label{tab:ase}" in audit.text(SI) or "\\label{tab:ase}" in audit.text(TEX)
        audit.record(not present, SI, "tab:ase is absent when its commits are not held", "absent",
                     "present" if present else "absent")
    for cells in (tex_table(audit, PAPER, "tab:ase") if ase_held else []):
        system = by_label.get(cells[0])
        where = f"{PAPER} tab:ase {system}"
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
    if ase_held:
        audit.cell(PAPER, "tab:ase lines", " ".join(shown), " ".join(ASE_PANEL))
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
    audit.record(len(revision) >= 1, PAPER, "tab:ase-morse recommended revisions", "at least one", str(len(revision)))
    rev_morse_short = revision[0][:8]
    audit.said(PAPER, f"This work is anneal \\texttt{{{rev_morse_short}}} at the length scale of the Morse pair.")
    audit.said("bench/ase-nonlj/README.md",
               "Anneal `recommended` uses `Config::recommended_with_scales` at `PairKind::length_scale()`.")
    audit.said("examples/ase_panel.rs", "Config::recommended_with_scales(n, length, well)")
    by_label = {label: s for s, label in nonlj.TEX_LABELS.items()}
    shown = []
    for cells in tex_table(audit, PAPER, "tab:ase-morse"):
        system = by_label.get(cells[0])
        where = f"{PAPER} tab:ase-morse {system}"
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
    audit.cell(PAPER, "tab:ase-morse lines", " ".join(shown), " ".join(MORSE_PANEL))
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
    audit.cell(PAPER, "Morse energy agreement", "6.8e-12", energy)
    audit.cell(PAPER, "Morse force agreement", "3.6e-12", force)
    audit.cell(PAPER, "Morse agreement at the seed-2000 start", "2000", " ".join(sorted({r["seed"] for r in identity})))
    audit.said(PAPER, "agrees with this code at the seed-2000 start to \\(6.8\\times 10^{-12}\\) in energy and "
                    "\\(3.6\\times 10^{-12}\\) in force.")

    # Stops: no run of this setting at a cap, the caps of each ASE arm.
    for system in (*ASE_PANEL, *MORSE_PANEL):
        audit.cell(f"ase panel {system}", "recommended wall stops", "0", float(stops(system, "recommended")))
        uncapped = sum(not r["cap_s"] for a in ("recommended", *ASE_ARMS) for r in groups[(system, a)])
        audit.cell(f"ase panel {system}", "table runs without a cap", "0", float(uncapped))
    audit.said(PAPER, "Each run stops at a hit, at the budget, or at a wall cap. The cap of the basin-hopping and "
                    "minima-hopping runs is three times the slowest wall of the panel's cluster-preset arm on that "
                    "cluster.")
    audit.said(PAPER, "No run of this setting reached its cap.")
    lj = ASE_PANEL[:5]

    def label(system: str) -> str:
        if system in panel.TEX_LABELS:
            return panel.TEX_LABELS[system]
        return nonlj.TEX_LABELS[system].replace(", \\(\\rho=", " at \\(\\rho=")

    mh_runs = sum(len(groups[(s, "ase_mh")]) for s in ASE_PANEL)
    mh_walls = sum(stops(s, "ase_mh") for s in ASE_PANEL)
    audit.cell(PAPER, "tab:ase minima-hopping runs at the cap", str(mh_runs), float(mh_walls))
    bh = [(s, stops(s, "ase_bh")) for s in ASE_PANEL if stops(s, "ase_bh")]
    spent = [s for s in ASE_PANEL if stops(s, "ase_bh", "budget") == len(groups[(s, "ase_bh")])]
    for s in spent:
        audit.cell(PAPER, f"{s} basin hopping charged the budget on every seed", "4000000",
                   float(min(int(r["charged"]) for r in groups[(s, "ase_bh")])))
    runs = []
    for s in lj:
        if runs and runs[-1][1] == hits(s, "recommended"):
            runs[-1][0].append(s)
        else:
            runs.append(([s], hits(s, "recommended")))
    bh_hits = [s for s in ASE_PANEL if hits(s, "ase_bh")]
    mh_hits = [s for s in ASE_PANEL if hits(s, "ase_mh")]
    audit.cell(PAPER, "basin hopping hits one cluster", "lj55", " ".join(bh_hits))
    audit.cell(PAPER, "minima hopping hits none", "", " ".join(mh_hits))
    below = [s for s in lj if max(p(s, a) for a in ASE_ARMS) < 0.05]
    audit.cell(PAPER, "Fisher below 0.05 on every Lennard-Jones cluster", " ".join(lj), " ".join(below))
    au_hits = [hits("au55", a) for a in ("anneal", "recommended", *ASE_ARMS)]
    audit.cell(PAPER, "no arm of the ASE panel hits Au55", "0", float(sum(au_hits)))
    au_ref = references["au55"]
    au_gaps = [f"{gap('au55', a, au_ref):.3f}" for a in ("recommended", *ASE_ARMS)]
    if ase_held:
        audit.said(PAPER, f"Minima hopping stopped at the cap on all {mh_runs} runs of Table~\\ref{{tab:ase}}, and basin "
                        f"hopping on {sum(n for _, n in bh)}: {series(f'{n} on {label(s)}' for s, n in bh)}.")
        audit.said(PAPER, f"On {series(label(s) for s in spent)} basin hopping spent the full \\(4\\times 10^{{6}}\\) calls "
                        "on every seed.")
        audit.said(PAPER, "This setting hits " + series(f"{' and '.join(label(s) for s in names)} on {n} of 12"
                                                      for names, n in runs) + ".")
        audit.said(PAPER, f"Basin hopping hits {label('lj55')} on {hits('lj55', 'ase_bh')} of 12 and no other cluster, and "
                        "minima hopping hits none.")
        audit.said(PAPER, f"Fisher's test against each ASE arm is below 0.05 on all {len(below)} Lennard-Jones clusters.")
        audit.said(PAPER, f"No arm hits {label('au55')}.")
        audit.said(PAPER, f"There the median best energy lies {au_gaps[0]}~eV above the reference for this setting, "
                        f"{au_gaps[1]}~eV for basin hopping, and {au_gaps[2]}~eV for minima hopping.")
    morse_walls = all(r["stop"] == "wall" for s in MORSE_PANEL for a in ASE_ARMS for r in groups[(s, a)])
    audit.record(morse_walls, PAPER, "every Morse ASE run stopped at the cap", "every", "every" if morse_walls else "not every")
    audit.said(PAPER, "Every basin-hopping and minima-hopping run of Table~\\ref{tab:ase-morse} stopped at the cap.")
    m = [hits(s, "recommended") for s in MORSE_PANEL]
    audit.cell(PAPER, "M38 at both ranges and M55 at rho=6 share one count", "1", float(len(set(m[:3]))))
    morse_sentence = (f"{label('m38r6').split(' at ')[0]} at \\(\\rho=6\\) and \\(\\rho=14\\) and {label('m55r6')} on "
                      f"{m[0]} of 12")
    audit.said(PAPER, f"This setting hits {morse_sentence}, and {label('m55r14')} on {m[3]} of 12.")
    morse_ase = {hits(s, a) for s in MORSE_PANEL for a in ASE_ARMS}
    audit.cell(PAPER, "Morse ASE hits", "0", float(max(morse_ase)))
    audit.said(PAPER, f"Both ASE arms hit {max(morse_ase)} of 12 on all {len(MORSE_PANEL)}.")
    morse_p = [{show_p(p(s, a)) for a in ASE_ARMS} for s in MORSE_PANEL]
    audit.cell(PAPER, "one Fisher p on the first three Morse clusters", "1", float(len(set.union(*morse_p[:3]))))
    audit.said(PAPER, f"Fisher's \\(p\\) against each ASE arm is {next(iter(morse_p[0]))} on the first three and "
                    f"{next(iter(morse_p[3]))} on {label('m55r14')}.")
    section = audit.text(SI).split("\\label{sec:ase}", 1)[1].split("\\section{AGOX GOFEE}", 1)[0]
    audit.record("Mann" not in section, PAPER, "sec:ase reports Fisher only", "no rank test", "no rank test"
                 if "Mann" not in section else "a rank test")
    discussion = audit.text(PAPER).split("\\label{sec:discussion}", 1)[1].split("\\section{Conclusions}", 1)[0]
    audit.record("Mann" not in discussion, PAPER, "the discussion reports Fisher only", "no rank test", "no rank test"
                 if "Mann" not in discussion else "a rank test")

    # Abstract, introduction, discussion and conclusions.
    gmin_au = gmin[("au55", "anneal")], gmin[("au55", "gmin")]
    rev_gmin = {r["code_revision"] for r in gmin_au[0]}
    rev_ase = {r["code_revision"] for r in groups[("au55", "recommended")]}
    rev_ase_short = sorted(rev_ase)[0][:8]
    overlap = rev_ase.intersection(rev_gmin)
    audit.record(bool(overlap), PAPER, "the two Au55 counts share a revision",
                 "overlap", "overlap" if overlap else "disjoint")
    au_gmin = f"{hits_of(gmin_au[0])} of {len(gmin_au[0])}", f"{hits_of(gmin_au[1])} of {len(gmin_au[1])}"
    audit.cell(PAPER, "Morse hits of this setting against ASE", "12 12 12 5", " ".join(str(n) for n in m))
    audit.cell(PAPER, "Morse hits of the ASE arms", "0", float(max(morse_ase)))
    if ase_held:
        audit.said(PAPER, f"and against the basin hopping and the minima hopping of the Atomic Simulation Environment on "
                        f"{len(lj)} Lennard-Jones clusters, Gupta {label('au55')}, and {len(MORSE_PANEL)} Morse clusters "
                        "\\cite{wales1997,larsenAse2017,goedecker2004}.")
    else:
        audit.said(PAPER, "and against the basin hopping and the minima hopping of the Atomic Simulation Environment on "
                        f"{len(MORSE_PANEL)} Morse clusters \\cite{{wales1997,larsenAse2017,goedecker2004}}.")
    quoted = quoted_systems(gmin_panel, gmin)
    kinds = [name for name, present in (
        ("Lennard-Jones", any(s.startswith("lj") for s in quoted)),
        ("Gupta", "au55" in quoted),
        ("Morse", any(s.startswith("m") for s in quoted)),
    ) if present]
    audit.said(PAPER, f"Section~\\ref{{sec:results}} runs this setting against GMIN on {len(quoted)} {series(kinds)} clusters,")
    mh_all = all(r["stop"] == "wall" for s in (*ASE_PANEL, *MORSE_PANEL) for r in groups[(s, "ase_mh")])
    audit.record(mh_all, PAPER, "minima hopping reached its cap on every run", "every", "every" if mh_all else "not every")
    def partial(system: str) -> str:
        return f"on {stops(system, 'ase_bh')} of 12 on {label(system)}"

    full = [s for s in ASE_PANEL if stops(s, "ase_bh") == len(groups[(s, "ase_bh")])]
    part = [s for s in ASE_PANEL if 0 < stops(s, "ase_bh") < len(groups[(s, "ase_bh")])]
    if ase_held:
        audit.said(PAPER, "Minima hopping reached its cap on every run, and basin hopping on every run of "
                        f"{series([*(label(s) for s in full), 'the Morse clusters'])} and "
                        f"{' and '.join(partial(s) for s in part)}.")
        audit.said(PAPER, f"On {series(label(s) for s in spent)} basin hopping spent the budget before the cap and missed on "
                        "every seed.")
    missed = [s for s in spent if hits(s, "ase_bh") == 0]
    audit.cell(PAPER, "basin hopping spent the budget and missed", " ".join(spent), " ".join(missed))
    audit.said(PAPER, f"Against the Atomic Simulation Environment this setting hits {morse_sentence} and "
                    f"{label('m55r14')} on {m[3]} of 12, where neither ASE arm hits.")

    def of(system: str, arm: str) -> str:
        return f"{hits_of(gmin[(system, arm)])} of {len(gmin[(system, arm)])}"

    def name(system: str) -> str:
        return gmin_panel.LABELS[system][1].replace(", \\(\\rho=", " at \\(\\rho=")

    morse_gmin = [s for s in quoted_systems(gmin_panel, gmin) if s.startswith("m")]
    if morse_gmin:
        behind = [s for s in morse_gmin if hits_of(gmin[(s, "anneal")]) < hits_of(gmin[(s, "gmin")])]
        level = [s for s in morse_gmin if s not in behind]

        def ours(system: str) -> str:
            return f"{name(system)} on {of(system, 'anneal')}"

        audit.said(PAPER, f"Against GMIN it hits {series(ours(s) for s in level)}, where "
                        f"GMIN hits {series(str(hits_of(gmin[(s, 'gmin')])) for s in level)}, and it hits "
                        f"{series(ours(s) for s in behind)}, where GMIN hits "
                        f"{series(str(hits_of(gmin[(s, 'gmin')])) for s in behind)}.")
        if "m55r14" in morse_gmin:
            audit.said(PAPER, f"{name('m55r14')} is in both panels: {hits('m55r14', 'recommended')} of 12 on revision "
                            f"\\texttt{{{rev_morse_short}}} and {of('m55r14', 'anneal')} on "
                            f"\\texttt{{{sorted(rev_gmin)[0][:8]}}}.")
    rev_gmin_text = sorted(rev_gmin)[0][:8]
    audit.cell(PAPER, "the GMIN panel revision", "1ce3ab64", rev_gmin_text)
    if ase_held:
        audit.said(PAPER, f"On {label('au55')} the ASE panel hits {hits('au55', 'recommended')} of 12 for this setting, "
                        f"as do both ASE arms, and the median best energy of this setting lies {au_gaps[0]}~eV above "
                        "the reference.")
    audit.cell(PAPER, "revisions behind the two Au55 counts", "1", float(len(rev_gmin | rev_ase)))
    step = 1e-4

    def curvature(pair) -> float:
        return (pair(1.0 + step) - 2.0 * pair(1.0) + pair(1.0 - step)) / step**2

    morse6 = curvature(lambda r: morse_pair(6.0, r))
    lennard_jones = curvature(lambda r: r**-12 - 2.0 * r**-6)
    audit.record(abs(morse6 - lennard_jones) < 1e-3, PAPER, "Morse rho=6 curvature at the pair minimum",
                 f"{lennard_jones:.3f}", f"{morse6:.3f}")
    audit.said(PAPER, "the Morse pair at \\(\\rho=6\\), whose well has the curvature of the Lennard-Jones well, and at "
                    "the shorter range \\(\\rho=14\\), and the many-body Gupta potential of gold.")
    ahead_lj = [s for s in lj if hits(s, "recommended") > max(hits(s, a) for a in ASE_ARMS)]
    ahead_morse = [s for s in MORSE_PANEL if hits(s, "recommended") > max(hits(s, a) for a in ASE_ARMS)]
    audit.cell(PAPER, "ahead of both ASE arms on every Lennard-Jones and Morse cluster",
               " ".join((*lj, *MORSE_PANEL)), " ".join((*ahead_lj, *ahead_morse)))
    if ase_held:
        audit.said(PAPER, f"this setting hits more seeds on all {len(ahead_lj)} Lennard-Jones clusters and all "
                        f"{len(ahead_morse)} Morse clusters, and no arm hits {label('au55')}.")
    else:
        audit.said(PAPER, f"this setting hits more seeds on all {len(ahead_morse)} Morse clusters")


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
        audit.record(
            worst < DEFAULT_HIT_TOL,
            path,
            "every hit within 1e-3 of its reference",
            f"{worst:.3g}",
            "below 0.001",
        )

    def f6(system: str) -> str:
        return f"\\({ref[system]:.6f}\\)"

    audit.said(PAPER, f"The panels of Sections~\\msref{{sec:gmin}} and~\\ref{{sec:ase}} add {f6('lj102')} at 102 atoms, "
                    f"{f6('lj103')} at 103, and {f6('lj150')} at 150, \\({ref['au55']:.7f}\\)~eV for gold at 55 atoms "
                    f"in the copper--gold Gupta archive, and from the Morse table {f6('m38r6')} and {f6('m38r14')} for "
                    f"M\\(_{{38}}\\) at \\(\\rho=6\\) and \\(14\\), {f6('m55r6')} and {f6('m55r14')} for M\\(_{{55}}\\), "
                    f"and {f6('m80r6')} and {f6('m80r14')} for M\\(_{{80}}\\).")
    for system, text in (("lj38", "\\(-173.928427\\) at 38"), ("lj55", "\\(-279.248470\\) at 55"),
                         ("lj75", "\\(-397.492331\\) at 75"), ("lj98", "\\(-543.665361\\) at"),
                         ("lj104", "\\(-582.086642\\) at 104")):
        audit.cell(PAPER, f"{system} reference", re.search(r"\\\((-[\d.]+)\\\)", text).group(1), ref[system])
        audit.said(PAPER, text)


def generated_tables(audit: Audit) -> None:
    """Each panel table in the manuscript is the block its summariser writes from the committed rows."""
    text = audit.text(PAPER)

    def block(marks: tuple[str, str]) -> str:
        start, end = marks
        if text.count(start) != 1 or text.count(end) != 1:
            return ""
        return text.split(start, 1)[1].split(end, 1)[0].strip("\n")

    gmin = load_script("gmin_panel_tex", "scripts/gmin_panel/run_panel.py")
    rows = gmin.read_rows(ROOT / "bench/gmin-panel/rows.csv")
    want = gmin.tex_table(gmin.quoted_summary(gmin.summarise(rows, "latest")), "latest")
    panel = load_script("ase_panel_tex", "scripts/ase_panel/summarize.py")
    nonlj = load_script("ase_nonlj_tex", "scripts/ase_nonlj/summarize.py")
    ablation = load_script("ablation_tex", "paper/jctc/scripts/ablation_table.py")
    h2h = load_script("h2h_tables_tex", "paper/jctc/scripts/h2h_tables.py")
    si_text = audit.text(SI)
    for label, marks, written, source in (
            ("tab:gofee", h2h.MARKS["gofee"], h2h.gofee_block(), text),
            ("sec:tip4p-gmin", h2h.MARKS["tip4p-prose"], h2h.tip4p_prose(), si_text),
            ("tab:tip4p-gmin", h2h.MARKS["tip4p"], h2h.tip4p_block(), text),
            ("tab:gofee-si", h2h.MARKS["gofee-si"], h2h.gofee_si_block(), si_text),
            ("tab:sc", h2h.MARKS["sc"], h2h.sc_block(), text),
            ("tab:sc-si", h2h.MARKS["sc-si"], h2h.sc_si_block(), si_text)):
        found = source.split(marks[0], 1)[1].split(marks[1], 1)[0].strip("\n") if marks[0] in source else ""
        ok = (marks[0] not in source) if not written else found == written
        audit.record(ok, SI if source is si_text else PAPER, f"{label} is the generator's block",
                     "as written" if ok else "edited by hand", "as written")
    for label, marks, written in (("tab:ablation", ablation.TEX_MARKS, ablation.tex_table()),
                                  ("tab:gmin", gmin.TEX_MARKS, want),
                                  ("tab:ase", panel.TEX_MARKS, panel.tex_table()),
                                  ("tab:ase-morse", nonlj.TEX_MARKS, nonlj.tex_table(nonlj.load_rows()))):
        ok = (marks[0] not in text and marks[0] not in si_text) if not written else block(marks) == written
        audit.record(ok, PAPER, f"{label} is the summariser's block", "as written" if ok else "edited by hand", "as written")


def sutton_chen(audit: Audit) -> None:
    """The Sutton-Chen counts in the prose, from the rows the table is written from."""
    h2h = load_script("h2h_tables_sc", "paper/jctc/scripts/h2h_tables.py")
    pairs = h2h.sc_pairs()
    ahead = sum(int(a["hits"]) > int(g["hits"]) for a, g in pairs)
    behind = sum(int(a["hits"]) < int(g["hits"]) for a, g in pairs)
    tied = sum(int(a["hits"]) == int(g["hits"]) for a, g in pairs)
    tests = h2h.sc_tests()
    significant = [a["system"] for a, g in pairs if float(tests[a["system"]]["fisher_p"]) < 0.05]
    gmin_faster = sum(float(g["wall_median_s"]) < float(a["wall_median_s"]) for a, g in pairs)
    ratios = [float(a["pot_per_eval_s"]) / float(g["pot_per_eval_s"]) for a, g in pairs]
    audit.cell(PAPER, "Sutton-Chen clusters where this setting hits more", "5", float(ahead))
    audit.cell(PAPER, "Sutton-Chen clusters tied on hits", "4", float(tied))
    audit.cell(PAPER, "Sutton-Chen clusters where GMIN hits more", "0", float(behind))
    audit.cell(PAPER, "Sutton-Chen clusters with Fisher p below 0.05", "sc108-38 sc126-80 sc96-80", " ".join(significant))
    audit.cell(PAPER, "Sutton-Chen clusters where GMIN has the lower median wall", "7", float(gmin_faster))
    audit.cell(PAPER, "Sutton-Chen per-call ratio inside the potential, low", "3.4", min(ratios))
    audit.cell(PAPER, "Sutton-Chen per-call ratio inside the potential, high", "4.9", max(ratios))
    if h2h.sc_gmin_quotable():
        for path in (PAPER, COVER):
            audit.said(path, f"On {spelled(len(pairs))} Sutton--Chen clusters, this setting hits at least as many seeds as "
                             f"GMIN on each, while GMIN has the lower median wall on {spelled(gmin_faster)}.")
        audit.said(PAPER, "This setting hits more seeds on 5 of the 9 clusters and the two tie on 4.")
        audit.said(PAPER, "Per call, this setting spends 3.4 to 4.9 times as long inside the potential as GMIN;")
    else:
        audit.said(PAPER, "This setting hits 12 of 12 on the three forms at 38 atoms and on the 12-6 and "
                          "9-6 forms at 55 atoms.")
        audit.said(PAPER, "It hits 5 of 12 on the 10-8 form at 55 atoms and on the 12-6 form at 80 atoms, "
                          "4 of 12 on the 10-8 form at 80 atoms, and 7 of 12 on the 9-6 form at 80 atoms.")
    audit.said(PAPER, "\\path{bench/metal-cluster-h2h/sc-n12/seeds.csv} holds the 324 seed rows")
    sc_seeds = audit.rows("bench/metal-cluster-h2h/sc-n12/seeds.csv")
    audit.cell(PAPER, "Sutton-Chen seed rows", "324", float(len(sc_seeds)))
    gmin_panel = load_script("gmin_panel_sc_fmt", "scripts/gmin_panel/run_panel.py")
    by_label = {h2h.sc_label(a["system"]): (a, g) for a, g in pairs}
    for cells in tex_table(audit, PAPER, "tab:sc"):
        if cells[0] not in by_label:
            continue
        a_row, g_row = by_label[cells[0]]
        sys_name = a_row["system"]
        a_calls = [float(r["evals_to_hit"]) for r in sc_seeds
                   if r["system"] == sys_name and r["arm"] == "anneal"
                   and r["harness_revision"] == a_row["harness_revision"]]
        g_calls = [float(r["evals_to_hit"]) for r in sc_seeds
                   if r["system"] == sys_name and r["arm"] == "gmin"
                   and r["harness_revision"] == g_row["harness_revision"]]
        if h2h.sc_gmin_quotable():
            want_a = gmin_panel.show_calls(median(a_calls))
            want_g = gmin_panel.show_calls(median(g_calls))
            audit.record(cells[4] == want_a, f"{PAPER} tab:sc {sys_name}", "anneal median calls formatted like tab:gmin",
                         cells[4], want_a)
            audit.record(cells[5] == want_g, f"{PAPER} tab:sc {sys_name}", "GMIN median calls formatted like tab:gmin",
                         cells[5], want_g)
        else:
            want_a = h2h.show_calls_median(h2h.type7_median(a_calls))
            audit.record(cells[2] == a_row["hits_over_n"], f"{PAPER} tab:sc {sys_name}",
                         "anneal hits", cells[2] if len(cells) > 2 else "", a_row["hits_over_n"])
            audit.record(len(cells) > 3 and cells[3] == want_a, f"{PAPER} tab:sc {sys_name}",
                         "anneal median calls", cells[3] if len(cells) > 3 else "", want_a)
    sutton_chen_pvalues(audit)


def sutton_chen_pvalues(audit: Audit) -> None:
    """Each Sutton-Chen p-value row, recomputed from the anneal seeds and the corrected GMIN seeds alone.

    The results.csv rows it supersedes stay in place with their old values."""
    panel = "bench/metal-cluster-h2h/sc-n12"
    seeds = audit.rows(f"{panel}/seeds.csv")
    results = audit.rows(f"{panel}/results.csv")
    stored = audit.rows(f"{panel}/pvalues.csv")
    where = f"{panel}/pvalues.csv"
    systems = sorted({r["system"] for r in seeds})
    audit.cell(where, "one p-value row per Sutton-Chen system", str(len(systems)), float(len(stored)))
    for row in stored:
        system = row["system"]
        anneal = [r for r in seeds if r["system"] == system and r["arm"] == "anneal"]
        own = {r["harness_revision"] for r in anneal}
        gmin = [r for r in seeds if r["system"] == system and r["arm"] == "gmin"
                and r["harness_revision"] == row["harness_revision"]]
        audit.record(len(own) == 1 and row["harness_revision"] not in own and len(gmin) == 12, where,
                     f"{system} is tested against the corrected GMIN rows only", "12 corrected rows",
                     f"{len(gmin)} rows of {row['harness_revision'][:8]}")
        hits_a = sum(r["hit"] == "1" for r in anneal)
        hits_g = sum(r["hit"] == "1" for r in gmin)
        audit.cell(where, f"{system} anneal hits", row["hits_anneal"], float(hits_a))
        audit.cell(where, f"{system} GMIN hits", row["hits_gmin"], float(hits_g))
        audit.cell(where, f"{system} Fisher p", row["fisher_p"], fisher(hits_a, len(anneal), hits_g, len(gmin)))
        for column, field in (("mannwhitney_p_evals", "evals_to_hit"), ("mannwhitney_p_wall", "wall_s")):
            audit.cell(where, f"{system} {column}", row[column],
                       mann_whitney([float(r[field]) for r in anneal], [float(r[field]) for r in gmin], True))
        old = [r for r in results if r["system"] == system and r["arm"] == "anneal" and r["harness_revision"] in own]
        named = f"results.csv system={system} arm=anneal harness_revision={next(iter(own))[:8]}:"
        audit.record(len(old) == 1 and row["supersedes"].startswith(named), where,
                     f"{system} names the results.csv row it supersedes, which stays", named,
                     row["supersedes"] if old else "superseded row missing")


NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def spelled(count: int) -> str:
    """A count as ACS style writes it in running text: one through nine in words, larger counts as numerals."""
    return NUMBER_WORDS[count] if 0 < count < 10 else str(count)


LENGTH_RECEIPT = "paper/jctc/logs/pdf-length.txt"


def rendered_length(audit: Audit, receipt: str = LENGTH_RECEIPT) -> None:
    """The abstract as pdftotext reads the rendered PDF, from the receipt pdf_length.py writes for these sources."""
    length = load_script("pdf_length", "paper/jctc/scripts/pdf_length.py")
    counts, digests = length.read_receipt(ROOT / receipt)
    current = length.source_digests()
    audit.record(digests == current, receipt, "the PDF counts were measured from the committed sources",
                 " ".join(f"{path} {digest[:12]}" for path, digest in sorted(current.items())),
                 " ".join(f"{path} {digest[:12]}" for path, digest in sorted(digests.items())))
    audit.record(0 < counts["abstract_words"] <= length.ABSTRACT_WORDS, receipt,
                 f"rendered abstract words, at most {length.ABSTRACT_WORDS} for JCTC", f"<= {length.ABSTRACT_WORDS}",
                 str(counts["abstract_words"]))
    audit.record(counts["abstract_numerals"] <= length.ABSTRACT_NUMERALS, receipt,
                 f"rendered abstract numerals, at most {length.ABSTRACT_NUMERALS} for JCTC",
                 f"<= {length.ABSTRACT_NUMERALS}", str(counts["abstract_numerals"]))
    low, high = length.BODY_WORDS
    audit.record(low <= counts["body_words"] <= high, receipt, f"rendered body words, {low}-{high} for JCTC",
                 f"{low}-{high}", str(counts["body_words"]))


def cross_references(audit: Audit) -> None:
    """The label files of the two documents are current, and every reference across them names a label."""
    labels = load_script("cross_labels", "paper/jctc/scripts/cross_labels.py")
    stale = labels.stale()
    audit.record(not stale, "paper/jctc/labels-si.tex", "the cross-document labels match the sources", "current",
                 " ".join(stale) or "current")
    for path, macro, other in ((TEX, "siref", "si"), (SI, "msref", "manuscript")):
        defined = labels.labels(other)
        missing = sorted({m for m in re.findall(r"\\" + macro + r"\{([^}]*)\}", labels.uncomment(audit.text(path)))
                          if m not in defined})
        audit.record(not missing, path, f"every \\{macro} names a label of {other}.tex", "all defined",
                     " ".join(missing) or "all defined")
        own = labels.labels("manuscript" if other == "si" else "si")
        loose = sorted({m for m in re.findall(r"\\ref\{([^}]*)\}", labels.uncomment(audit.text(path))) if m not in own})
        audit.record(not loose, path, "every \\ref names a label of its own document", "all defined",
                     " ".join(loose) or "all defined")


BRITISH_STEMS = (
    "analyse", "behaviour", "cancell", "catalogue", "centre", "characteris",
    "colour", "favour", "fuell", "initialis", "labell", "licence", "maximis",
    "minimis", "modell", "neighbour", "normalis", "optimis", "organis",
    "parametris", "randomis", "recognis", "summaris", "symmetris", "travell",
)
BRITISH = re.compile(
    rf"\b\w*(?:{'|'.join(map(re.escape, BRITISH_STEMS))})\w*\b",
    re.IGNORECASE,
)


def british_words(text: str) -> list[str]:
    """British stems in text. The noun plural analyses is the American plural of analysis."""
    return [match.group(0) for match in BRITISH.finditer(text) if match.group(0).lower() != "analyses"]


def british_spellings(audit: Audit, root: Path = ROOT) -> None:
    """US spelling in every TeX source and every string a paper script can typeset, cite keys aside.

    A string that opens with "# " is the header of a committed log, which never reaches a PDF.
    """
    tracked = subprocess.run(["git", "ls-files", "paper/jctc/*.tex", "paper/jctc/tables/*.tex",
                              "paper/jctc/scripts/*.py"], cwd=root, capture_output=True, text=True, check=True)
    found = []
    for path in tracked.stdout.split():
        text = (root / path).read_text()
        if path.endswith(".py"):
            tree = ast.parse(text)
            parts = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
            strings = ["".join(p.value if isinstance(p, ast.Constant) else "{}" for p in node.values)
                       if isinstance(node, ast.JoinedStr) else node.value for node in ast.walk(tree)
                       if isinstance(node, ast.JoinedStr)
                       or (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in parts)]
            strings = [s for s in strings if not s.startswith("# ")]
        else:
            strings = [text]
        found += [f"{path}:{word}" for s in strings for word in british_words(re.sub(r"\\cite\{[^}]*\}", "", s))]
    audit.record(not found, TEX, "US spelling in everything that reaches a PDF", "none", " ".join(found) or "none")


def figure_source_scripts(audit: Audit, root: Path = ROOT) -> None:
    """Every effective receipt names the sha256 of its current source."""
    receipt = "paper/jctc/FIGURE-SOURCE.tsv"
    with (root / receipt).open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    updates_path = root / "paper/jctc/FIGURE-SOURCE-updates.tsv"
    if updates_path.is_file():
        by_key = {(row["path"], row["kind"]): row for row in rows}
        with updates_path.open(newline="") as handle:
            for update in csv.DictReader(handle, delimiter="\t"):
                key = (update["path"], update["kind"])
                if update["status"] == "superseded":
                    by_key.pop(key, None)
                elif update["status"] == "active":
                    by_key[key] = {
                        field: update[field]
                        for field in ("path", "kind", "source_sha256")
                    }
                else:
                    audit.record(False, str(updates_path.relative_to(root)),
                                 f"{key} update status", "active or superseded",
                                 update["status"])
        rows = list(by_key.values())
    for row in rows:
        path = row["path"]
        source = root / path
        digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else "missing"
        audit.record(digest == row["source_sha256"], receipt,
                     f"{path} current source sha256",
                     row["source_sha256"], digest)
    if root == ROOT:
        audit.cell(receipt, "effective figure data rows", "276",
                   float(sum(row["kind"] == "data" for row in rows)))


def jctc_requirements(audit: Audit) -> None:
    """The rendered length and the Sutton-Chen section."""
    rendered_length(audit)
    cross_references(audit)
    british_spellings(audit)
    figure_source_scripts(audit)
    block = re.search(r"\\begin\{suppinfo\}(.*?)\\end\{suppinfo\}", audit.text(TEX), re.DOTALL)
    audit.record(block is not None and block.group(1).rstrip().endswith("(PDF)."), TEX,
                 "the Supporting Information entry names its format", "(PDF).",
                 block.group(1).strip()[-6:] if block else "no suppinfo")
    sections = re.findall(r"\\subsection\{Sutton--Chen clusters at equal workload\}\s*\\label\{sec:sc\}",
                          audit.text(TEX))
    audit.record(len(sections) == 1, TEX, "the Sutton-Chen section is present", "1", str(len(sections)))
    h2h = load_script("h2h_tables_sc", "paper/jctc/scripts/h2h_tables.py")
    systems = len(h2h.sc_pairs())
    rows = re.search(r"\\label\{tab:sc\}", audit.text(TEX))
    audit.record(systems == 9 and rows is not None, TEX, "the Sutton-Chen table is present with its nine clusters",
                 "9", str(systems) if rows else "no tab:sc")
    history = (
        (PAPER, "GMIN is run here differently from two earlier campaigns"),
        (PAPER, "An earlier campaign on seeds 0--63"),
        (SI, "\\section{Earlier campaign}"),
        (SI, "Arms that were measured and not kept"),
        (PAPER, "were not kept"),
        (PAPER, "corrected GMIN"),
        (COVER, "This submission replaces the preprint"),
        (COVER, "The title and the content have changed."),
        (TEX, "reproducibility archive for this article"),
        (COVER, "reproducibility archive for this article"),
    )
    for path, fragment in history:
        present = fragment in collapse(audit.text(path))
        audit.record(not present, path, "no history narration", "absent", "present" if present else "absent")


def crest_equal(audit: Audit) -> None:
    """The committed CREST rows, and the absence of that panel from the paper."""
    h2h = load_script("h2h_tables_crest", "paper/jctc/scripts/h2h_tables.py")
    where = "bench/crest-equal/rows.csv"
    rows = audit.rows(where)
    revision = rows[-1]["code_revision"]
    rows = [r for r in rows if r["code_revision"] == revision]
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in h2h.CREST_ARMS}
    summary = h2h.crest_equal_summary()
    for arm, runs in arms.items():
        seeds = sorted(int(r["seed"]) for r in runs)
        audit.record(seeds == list(range(2000, 2012)), where, f"crest-equal {arm}: seeds 2000-2011, once each",
                     "2000-2011", " ".join(map(str, seeds)))
        audit.record(all(r["crest_exit"] in ("", "0") for r in runs), where, f"crest-equal {arm}: every run exits 0",
                     "0", " ".join(sorted({r["crest_exit"] or "0" for r in runs})))

    def calls(arm: str) -> list[float]:
        return [float(r["charged"]) for r in arms[arm]]

    def hits(arm: str) -> int:
        return sum(r["hit"] == "true" for r in arms[arm])

    target = calls("anneal")
    audit.record(len(set(target)) == 1, where, "this setting charges the same GFN2 calls on every seed", "one count",
                 " ".join(sorted({f"{v:.0f}" for v in target})))
    center = target[0]
    tolerance = h2h.CREST_CALL_TOLERANCE
    off = abs(median(calls("crest")) - center) / center
    audit.record(off <= tolerance, where,
                 f"CREST at 0.90 ps is at the calls of this setting: median within {tolerance:.0%}",
                 f"<= {tolerance:.0%}", f"{off:.2%}")
    within = sum(abs(v - center) <= tolerance * center for v in calls("crest"))
    audit.cell(where, "CREST runs within the call tolerance", summary["crest"]["within_5pct"], float(within))
    within_wall = sum(abs(v - center) <= tolerance * center for v in calls("crest-wall"))
    audit.cell(where, "CREST runs at 0.11 ps within the call tolerance", summary["crest-wall"]["within_5pct"],
               float(within_wall))
    readme = audit.text(CREST_EQUAL_README)
    for fragment in (
            "The CREST rows are charged the calls the driver served.",
            "CREST's own\n  `Total number of energy+grad calls` line was not read and no CREST log was\n  kept",
            "The driver was started with a ceiling of 50000 calls, so the 4000\n  in the `budget` column of the CREST rows is a label, not a cap.",
    ):
        audit.record(fragment in readme, CREST_EQUAL_README, "the recorded CREST protocol", fragment,
                     "present" if fragment in readme else "absent")
    stale = ("refuses calls past the 4000-call ledger", "charged the larger of the driver's count",
             "for CREST the larger of the")
    for fragment in stale:
        audit.record(fragment not in audit.text(PAPER), PAPER, "no next-run CREST protocol in the recorded panel",
                     "absent", "present" if fragment in audit.text(PAPER) else "absent")
    audit.record(float(summary["crest"]["crest_len_ps"]) == 0.90 and float(summary["crest-wall"]["crest_len_ps"]) == 0.11,
                 where, "the CREST lengths are 0.90 and 0.11 ps", "0.90 0.11",
                 f"{summary['crest']['crest_len_ps']} {summary['crest-wall']['crest_len_ps']}")
    for arm in ("crest", "crest-wall"):
        a, b = arms["anneal"], arms[arm]
        audit.cell(where, f"{arm} Fisher p", summary[arm]["fisher_p"], fisher(hits("anneal"), len(a), hits(arm), len(b)))
        for key, column in (("charged", "mannwhitney_calls_p"), ("pot_s", "mannwhitney_inside_p"),
                            ("wall_s", "mannwhitney_wall_p"), ("outside_s", "mannwhitney_outside_p")):
            p = mann_whitney([float(r[key]) for r in a], [float(r[key]) for r in b], continuity=True)
            audit.cell(where, f"{arm} Mann-Whitney p on {key}", summary[arm][column], p)
    withdrawn_counts = (121, 94, 97)
    dropped = (
        "tab:crest-equal", "sec:hexamer", "equal GFN2",
        "equal tight-binding", "served tight-binding",
        *(f"{count} of 192" for count in withdrawn_counts),
        *(f"{count}/192" for count in withdrawn_counts),
    )
    for path in (TEX, SI, COVER, "paper/jctc/FIGURES.md", "paper/jctc/CLAIMS.md"):
        text = audit.text(path)
        found = [phrase for phrase in dropped if phrase in text]
        audit.record(not found, path, "the equal-call panel and the withdrawn hexamer counts are absent",
                     "absent", " ".join(found) or "absent")
    audit.record(not (ROOT / "paper/jctc/figs/water_crest.pdf").exists(), "paper/jctc/figs/water_crest.pdf",
                 "the withdrawn hexamer figure is absent", "absent",
                 "present" if (ROOT / "paper/jctc/figs/water_crest.pdf").exists() else "absent")


TEMPLATE_SWITCHES = ("prior_reseed", "prior_growth", "prior_core", "prior_motif")


def prior_switches(audit: Audit) -> None:
    """The current method exposes only prior_soap and rejects retired controls."""
    pattern = "|".join(rf"cfg\.{name}\b" for name in TEMPLATE_SWITCHES)
    current = [str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.rs")
               if re.search(pattern, path.read_text(encoding="utf-8"))]
    audit.record(not current, "src", "no current source reads the four template switches", "none",
                 " ".join(current) or "none")
    config = audit.text("src/methods/cluster_hopping/config.rs")
    for name in TEMPLATE_SWITCHES:
        audit.record(f'"{name}"' in config, "src/methods/cluster_hopping/config.rs",
                     f"Config rejects retired {name}", name,
                     name if f'"{name}"' in config else "absent")
    audit.said(TEX, "The method exposes one structure-prior control, \\texttt{prior\\_soap}.")
    audit.said(TEX, "none is a proposal control\nof the method described here.")
    removed = git("merge-base", "--is-ancestor", "1c1b1238", "HEAD").returncode == 0
    audit.record(removed, PAPER, "the template removal is in this history", "1c1b1238", "1c1b1238" if removed else "absent")
    for revision in ("c2de652d", "e3ed7dad", "9a30f319", "beb89163"):
        found = sorted({name for name in TEMPLATE_SWITCHES
                        if git("grep", "-q", "-E", rf"cfg\.{name}\b", revision, "--", "src").returncode == 0})
        audit.record(len(found) == len(TEMPLATE_SWITCHES), PAPER, f"revision {revision} reads the template switches",
                     " ".join(sorted(TEMPLATE_SWITCHES)), " ".join(found) or "none")
    has = git("grep", "-q", "prior_core", "a24ba760", "--", "src").returncode == 0
    audit.record(not has, PAPER, "revision a24ba760 predates the switches", "none", "prior_core" if has else "none")
    audit.said("bench/NOTES-templates-removed.md", "`1ce3ab64`")


def method_definitions(audit: Audit) -> None:
    """The constants of the quench, the climb, and the descriptor moves, as the source sets them."""
    sources = {
        "src/methods/warm_lbfgs.rs": ("max_pairs: 8,", "gtol: 1e-6,", "armijo: 1e-4,", "curvature: 0.9,",
                                      "max_line_evals: 20,", "self.inner.norm = GradNorm::Infinity;"),
        "src/methods/cluster_hopping/config.rs": ("screen_steps: 25,", "cfg.relax_steps = 1000;",
                                                 "escape_lanczos_steps: 16,", "escape_overshoot: 1.5,",
                                                 "escape_max_climb: 24,", "escape_stall_patience: 5_000,"),
        "src/methods/activation.rs": ("perp_steps: 3,", "perp_rate: 0.02,", "refresh: 3,"),
        "src/methods/cluster_hopping.rs": ("const FINISH_ROUNDS: usize = 8;", "const STEEP_WELL_PER_ATOM: f64 = 64.0;",
                                           "factor.sqrt().clamp(1.0, 16.0)", "1e-10 * (1.0 + out.0.abs())"),
        "src/soap.rs": ("const FIVEFOLD_AXIS: f64 = 1.40;", "const FIVEFOLD_GATE: f64 = 1.15;",
                        "const FIVEFOLD_MIN_AXES: usize = 2;", "const PENTAGON_CAP: f64 = 0.75;",
                        "const RECON_LO: f64 = 0.89;", "const RECON_HI: f64 = 1.00;", "const NU3_DEFECT: f64 = 0.05;",
                        "concentrate_residual(&mut dr, 5);", "tikhonov_jtj(&j, dp.view(), 1e-3)",
                        "0.5 * (1.0 + libm::cos(PI * r / rcut))", "libm::exp(-0.5 * (r / (rcut / 3.0)).powi(2))"),
        "src/methods/cluster_hopping/moves.rs": ("n_max: 3,\n                    l_max: 6,",
                                                "crate::soap::step_away_fivefold(x, *rmsd, rng)",
                                                "crate::soap::step_away_fivefold_measured(x, *rmsd)"),
    }
    for path, fragments in sources.items():
        text = audit.text(path)
        for fragment in fragments:
            found = fragment.replace("\\n", "\n") in text
            audit.record(found, path, "a constant of the method definitions", fragment, fragment if found else "absent")
    for fragment in ("with 8 correction\npairs", "\\(c_1=10^{-4}\\) and \\(c_2=0.9\\) in at most 20 evaluations",
                     "largest component of the gradient is below\n\\(10^{-6}\\)", "the first 25 iterations",
                     "of 1000 iterations", "by up to 8\nmore blocks", "is below \\(10^{-3}\\)",
                     "energy per atom exceeds 64 wells", "16 Lanczos iterations", "step \\(10^{-4}\\)",
                     "takes 3 steps", "at rate \\(0.02\\)", "recomputed every 3 steps", "stops\nat 24 steps",
                     "carried \\(1.5\\) steps further", "An axis with \\(d_5<1.40\\) counts as fivefold",
                     "unless at least 2 axes do", "\\(d_5\\) is nearest \\(0.90\\) or the one nearest \\(0.99\\)",
                     "5 atoms with the largest", "root-mean-square displacement of \\(0.75\\)",
                     "if that length is at most \\(1.15\\)", "the 12 atoms farthest from it",
                     "with \\(n=0,1,2\\), \\(l\\le 6\\)", "is below \\(0.05\\) the move returns",
                     "J^{\\mathsf T}J + 10^{-3}I"):
        audit.said(PAPER, fragment)
    audit.said(PAPER, "The fivefold move and its constants are in \\path{src/soap.rs}.")
    soap = audit.text("src/soap.rs")
    far = re.search(r"for &\(_, v\) in far\.iter\(\)\.take\((\d+)\.min\(n\)\)", soap)
    audit.cell(PAPER, "farthest atoms that seed fivefold axes", "12", float(far.group(1)) if far else float("nan"))


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

    si = audit.text(SI)
    audit.record("\\section{Earlier campaign}" not in si and "200 of 200" not in si, SI,
                 "the SI carries no earlier campaign", "none", "none")

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
    audit.said(PAPER, f"\\((\\mathrm{{H_2O}})_{{10}}\\) hits on {frac(ten).replace('/', ' of ')} seeds.")
    calls = [r["calls_to_first_hit"] for r in ten]
    audit.said(PAPER, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    for n in ("15", "20"):
        group = pick(rows, n=n, budget="500000")
        audit.cell(PAPER, f"tip4p ({n}) hits at 5e5", "0/6", frac(group))
        gaps = [float(r["best_kj"]) - float(r["reference_kj"]) for r in group]
        word = {"15": "fifteen", "20": "twenty"}[n]
        audit.said(PAPER, f"The best {word}-molecule energies lie \\({min(gaps):.10f}\\) to \\({max(gaps):.10f}\\)~kJ/mol above")


def manuscript_hexamer(audit: Audit) -> None:
    dropped = (
        "tab:crest-counters",
        "tab:hexamer-gaps",
        "sec:si-hexamer",
        "142110",
        "209931",
        "32112",
        "50063",
        "544.0",
        "195.5",
        "1.3509",
        "4.35\\times 10^{-8}",
        "4836648ee77203e9462d51e25a1970b3f0df3a3bcd126f9a7454f6896ac66df8",
        "water_crest.pdf",
        "The manuscript table uses the hexamer logs",
        "Five polished gaps lie within 1e-7 Eh",
        "Open work remains on the water 16-mer",
        "six crest rows",
        "driver-count budget",
        "budget-fair polish",
        "McNemar values",
    )
    for path in (TEX, SI, COVER, "paper/jctc/FIGURES.md",
                 "paper/jctc/PROSE-RECEIPT.md", "paper/jctc/CLAIMS.md"):
        text = audit.text(path)
        found = [phrase for phrase in dropped if phrase in text]
        audit.record(
            not found,
            path,
            "the pre-fix six-seed GFN2 hexamer tables and numbers are absent",
            "absent",
            " ".join(found) or "absent",
        )

    rows = sorted({int(n) for n in re.findall(r"^Row (\d+) is ", audit.text("paper/jctc/FIGURES.md"), re.MULTILINE)})
    audit.record(rows == list(range(1, len(rows) + 1)), "paper/jctc/FIGURES.md", "the figure rows run from 1 without a gap",
                 f"1-{len(rows)}", " ".join(map(str, rows)))
    drawn = f"the {spelled(len(rows))} figures"
    for path in ("paper/jctc/REPRODUCE.md", "paper/jctc/FIGURES.md"):
        audit.said(path, f"We drew {drawn} with the committed generators.")
    audit.said("paper/jctc/REPRODUCE.md", f"The driver draws {drawn}.")
    audit.said("paper/jctc/REPRODUCE.md", f"Rows 10 to {len(rows)}:")


def manuscript_cross_field(audit: Audit) -> None:
    rows = audit.rows("results/cross-field-hits.csv")
    stalls = audit.rows("results/lj-funnel-stalls.csv")
    rates = audit.rows("bench/tip4p-water-rates/rates.csv")
    lj98 = sorted(pick(rows, system="lj", n="98", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    calls = [r["calls_to_first_hit"] for r in lj98]
    audit.said(PAPER, f"LJ\\(_{{98}}\\) hit on {frac(lj98).replace('/', ' of ')}.")
    audit.said(PAPER, "In seed order the calls to the first hit are " + ", ".join(calls[:-1]) + f", and {calls[-1]}.")
    audit.said(PAPER, f"The median is {median([float(c) for c in calls]):.0f}.")
    retry98 = pick(stalls, phase="retry", n="98")
    lj104 = sorted(pick(rows, system="lj", n="104", arm="bipop-leave"), key=lambda r: int(r["seed"]))
    hit104 = [r for r in lj104 if r["hit"] == "1"]
    retry104 = sorted((r["seed"] for r in pick(stalls, phase="retry", n="104") if r["hit"] == "1"), key=int)
    retry_median = median([float(r["calls_to_hit"]) for r in retry98])
    arm_median = median([float(c) for c in calls])
    audit.record(retry_median < arm_median, PAPER, "the retry median on LJ98 is lower than the arm median",
                 "lower", "lower" if retry_median < arm_median else "not lower")
    audit.said(PAPER, f"By comparison, the pentagonal-retry baseline also hits {frac(retry98).replace('/', ' of ')} on "
                    f"LJ\\(_{{98}}\\) with a lower median of {retry_median:.0f} calls, and on LJ\\(_{{104}}\\) reaches "
                    f"{len(retry104)} of 6 seeds ({retry104[0]}, {retry104[1]}, and {retry104[2]}) versus "
                    f"{len(hit104)} of {len(lj104)} ({hit104[0]['seed']} and {hit104[1]['seed']}, at "
                    f"{hit104[0]['calls_to_first_hit']} and {hit104[1]['calls_to_first_hit']} calls) for the BIPOP variant.")
    only_arm = sorted({r["seed"] for r in hit104} - set(retry104))
    only_retry = sorted(set(retry104) - {r["seed"] for r in hit104})
    audit.cell(PAPER, "the hit the retry rows miss", "6401", " ".join(only_arm))
    audit.cell(PAPER, "hits the retry rows record and this arm misses", "6402 6404", " ".join(only_retry))
    for n, budget in (("15", "500000"), ("20", "2000000")):
        trust = pick(rows, system="tip4p", n=n, arm="cage-trust")
        audit.cell(PAPER, f"cage-trust ({n}) hits", "0/6", frac(trust))
        audit.cell(PAPER, f"cage-trust ({n}) budget", budget, float(trust[0]["budget"]))
        rate = pick(rates, n=n, budget=budget)
        audit.cell(PAPER, f"rate file ({n}) at {budget}", "1/6", frac(rate))
        audit.cell(PAPER, f"rate file ({n}) hit seed", {"15": "3", "20": "2"}[n], " ".join(r["seed"] for r in rate if r["hit"] == "1"))
    best = min(pick(rows, system="tip4p", n="20", arm="cage-trust"), key=lambda r: float(r["best"]))
    audit.said(PAPER, f"\\({float(best['best']):.6f}\\)~kJ/mol, which is \\({float(best['best']) - float(best['reference']):.3f}\\)~kJ/mol above")


def supporting_information(audit: Audit) -> None:
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


CREST_EQUAL_README = "bench/crest-equal/README.md"
CREST_EQUAL_RGPOT_SIDECAR = "bench/crest-equal/rgpot-revisions.csv"
GFN2_RGPOT_SIDECAR = "bench/covalent-h2h/gfn2/rgpot-revisions.csv"


def sidecar_rows(audit: Audit, path: str) -> list[dict[str, str]]:
    """Read a sidecar CSV through audit.text without registering it as a benchmark result table."""
    lines = [line for line in audit.text(path).splitlines(keepends=True) if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(lines))


def crest_equal_bench(audit: Audit) -> None:
    """The equal-call CREST panel against its README: seeds, the budget, the ledger, and the summary."""
    panel = load_script("crest_equal_panel", "scripts/crest_equal/run_panel.py")
    where = "bench/crest-equal/rows.csv"
    rows = audit.rows(where)
    audit.record("rgpot_revision" not in rows[0], where,
                 "rows.csv keeps its original header without rgpot_revision",
                 "absent", "present" if "rgpot_revision" in rows[0] else "absent")
    sidecar = sidecar_rows(audit, CREST_EQUAL_RGPOT_SIDECAR)
    row_keys = [(r["system"], r["arm"], r["seed"], r["code_revision"], r["host"]) for r in rows]
    side_keys = [(r["system"], r["arm"], r["seed"], r["code_revision"], r["host"]) for r in sidecar]
    audit.record(row_keys == side_keys, CREST_EQUAL_RGPOT_SIDECAR,
                 "each crest-equal row has one matching rgpot-revisions.csv sidecar row",
                 f"{len(row_keys)} aligned rows",
                 f"{len(side_keys)} rows, {'aligned' if row_keys == side_keys else 'mismatched'}")
    for s_row in sidecar:
        rev = s_row["code_revision"]
        rgpot = s_row.get("rgpot_revision", "")
        if rev == "723b7c23bbb31a076a559e3ac7f4f91be9e36e1e":
            ok, want = rgpot == "", "empty"
        else:
            ok, want = bool(re.fullmatch(r"[0-9a-f]{40}", rgpot)), "40-hex sha"
        audit.record(ok, CREST_EQUAL_RGPOT_SIDECAR,
                     f"{s_row['system']} {s_row['arm']} seed {s_row['seed']} rgpot_revision",
                     want, rgpot or "empty")
    revisions = sorted({r["code_revision"] for r in rows})
    for revision in revisions:
        mine = [r for r in rows if r["code_revision"] == revision]
        for arm in panel.ARMS:
            seeds = sorted(int(r["seed"]) for r in mine if r["arm"] == arm)
            audit.record(seeds == list(panel.SEEDS), where, f"{revision[:8]} {arm} seeds",
                         " ".join(map(str, seeds)), " ".join(map(str, panel.SEEDS)))
    audit.record({r["budget"] for r in rows} == {str(panel.BUDGET)}, where, "budget",
                 " ".join(sorted({r["budget"] for r in rows})), panel.BUDGET)
    over = sorted(int(r["charged"]) for r in rows if int(r["charged"]) > panel.BUDGET)
    audit.record(not over, where, "no row is charged above the budget", " ".join(map(str, over)) or "none", "none")
    counts_path = "bench/crest-equal/call-counts.csv"
    if counts_path in audit.read or (ROOT / counts_path).is_file():
        for index, row in enumerate(audit.rows(counts_path), start=1):
            label = f"row {index} {row.get('arm', '')} seed {row.get('seed', '')} charged"
            try:
                driver_calls = int(row["driver_calls"])
                charged = int(row["charged"])
                if row["arm"] == "anneal" and not row.get("crest_log_calls"):
                    expected: object = driver_calls
                else:
                    expected = max(driver_calls, int(row["crest_log_calls"]))
                ok = charged == expected
            except (KeyError, ValueError):
                ok = False
                expected = "valid driver_calls and crest_log_calls"
            audit.record(ok, counts_path, label, row.get("charged", ""), expected)

    def mw(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return float("nan")
        result = mann_whitney(left, right, continuity=True, degenerate=1.0)
        return float("nan") if result is None else result

    summary = panel.summarise(rows, fisher_fn=fisher, mann_whitney_fn=mw)
    committed = audit.rows("bench/crest-equal/summary.csv")
    audit.record(summary == committed, "bench/crest-equal/summary.csv", "recomputed from rows.csv",
                 "equal" if summary == committed else "differs", "equal")
    readme_text = audit.text(CREST_EQUAL_README)
    forbidden_claim = (panel.claim_sentence(summary), "hits 1 of 12", "hits 4 of 12", "1/12", "4/12")
    found_claim = [item for item in forbidden_claim if item in readme_text]
    audit.record(not found_claim, CREST_EQUAL_README,
                 "sign-affected hit-rate claim is absent from crest-equal README",
                 "absent", " | ".join(found_claim) or "absent")
    anneal = [int(r["charged"]) for r in rows if r["arm"] == "anneal"]
    centre = sorted(anneal)[len(anneal) // 2 - 1: len(anneal) // 2 + 1]
    centre = sum(centre) / len(centre)
    for arm in ("crest", "crest-wall"):
        charged = sorted(int(r["charged"]) for r in rows if r["arm"] == arm)
        within = sum(abs(value - centre) <= 0.05 * centre for value in charged)
        row = next(r for r in summary if r["arm"] == arm)
        audit.cell(where, f"{arm} within 5 percent", row["within_5pct"], within)
        audit.said(CREST_EQUAL_README, f"{within} of {len(charged)}")
        audit.said(CREST_EQUAL_README, f"{charged[0]} to {charged[-1]} calls")
        if arm == "crest":
            audit.said(CREST_EQUAL_README, f"{charged[0] / centre:.2f} to {charged[-1] / centre:.2f} of anneal's {centre:g}")
    audit.said(CREST_EQUAL_README, f"The most any CREST row was charged is {max(int(r['charged']) for r in rows if r['arm'] == 'crest')}")
    walls = {arm: sorted(float(r["wall_s"]) for r in rows if r["arm"] == arm) for arm in panel.ARMS}
    middle = {arm: (w[len(w) // 2 - 1] + w[len(w) // 2]) / 2 for arm, w in walls.items()}
    audit.said(CREST_EQUAL_README, f"Median wall is {middle['anneal']:.2f} s for anneal, {middle['crest']:.2f} s at 0.90 ps and "
                                   f"{middle['crest-wall']:.2f} s at\n0.11 ps.")
    unconverged = sorted(r["seed"] for r in rows if r["arm"] == "anneal" and not r["gap_Eh"])
    audit.said(CREST_EQUAL_README, f"Anneal seed {unconverged[0]} polished without converging")
    audit.record(len(unconverged) == 1, where, "unconverged anneal polishes", " ".join(unconverged), "one")


def gad_published(audit: Audit) -> None:
    path = "bench/gad-published/README.md"
    rows = audit.rows("bench/gad-published/rows.csv")
    seeds = audit.rows("bench/gad-published/seed-reruns.csv")
    columns = {"Median calls to hit": "evals_median", "IQR": "evals_iqr", "Best (median)": "best_median",
               "Wall (s)": "wall_median_s", "Pot (s)": "pot_median_s", "Outside (s)": "outside_median_s",
               "Inside µs/call": "inside_us", "Outside µs/call": "outside_us"}
    for line, row in md_table(audit, path, "Size | Arm | Hits"):
        system = row["Size"].lower()
        line_rows = pick(rows, system=system, arm=row["Arm"])
        base = line_rows[-1]
        where = f"{path}:{line} {system} {row['Arm']}"
        audit.cell(where, "hits", row["Hits"], f"{base['hits']}/{base['n']}")
        for label, key in columns.items():
            audit.cell(where, label, row[label], float(base[key]))
        if row["Arm"] == "anneal":
            audit.cell(where, "Fisher p", row["Fisher p"], None)
            continue
        anneal = pick(rows, system=system, arm="anneal")[-1]
        inferential = next(candidate for candidate in reversed(line_rows)
                           if candidate["fisher_p"] and candidate["mannwhitney_p"])
        p = fisher(int(base["hits"]), int(base["n"]), int(anneal["hits"]), int(anneal["n"]))
        audit.cell(where, "Fisher p (recomputed)", row["Fisher p"], p)
        audit.cell(where, "Fisher p (rows.csv)", row["Fisher p"], float(inferential["fisher_p"]))
        audit.cell(where, "Mann-Whitney p (rows.csv)", row["Mann–Whitney p"],
                   float(inferential["mannwhitney_p"]))
    for row in pick(rows, arm="gad_deposit"):
        gen, lo, hi = row["hit_generation"], row["evals_lower"], row["evals_upper"]
        audit.cell(path, f"{row['system']} lower bound, generations 0..{gen} x 64", lo, float((int(gen) + 1) * 64))
        audit.cell(path, f"{row['system']} upper bound, x 1000 FIRE steps", hi, float((int(gen) + 1) * 64 * 1000))
        size = row["system"].removeprefix("lj")
        audit.said(path, f"generation {gen} for LJ{size} (bounds {lo} and {hi} calls)")
    expected = {("lj31", "anneal"): 12, ("lj31", "gmin"): 12, ("lj31", "gad"): 12, ("lj38", "gad"): 12}
    for group, count in expected.items():
        found = pick(seeds, system=group[0], arm=group[1])
        audit.cell("bench/gad-published/seed-reruns.csv", f"{group[0]} {group[1]} seed rows", str(count), float(len(found)))
        audit.record({row["host"] for row in found} == {"bench"}, "bench/gad-published/seed-reruns.csv",
                     f"{group[0]} {group[1]} rerun host", "bench", " ".join(sorted({row["host"] for row in found})))
    hits = {row["seed"]: row["evals_to_hit"] for row in pick(seeds, system="lj31", arm="gad") if row["hit"] == "1"}
    audit.said(path, f"seed 2000 at {hits['2000']} calls and seed 2006 at {hits['2006']} calls")
    generations = {}
    for system in ("lj31", "lj38"):
        values = [float(row["generations"]) for row in pick(seeds, system=system, arm="gad")]
        generations[system] = (median(values), min(values), max(values))
    audit.said(path, "a median of "
               f"{generations['lj31'][0]:g} generations on LJ31 "
               f"({generations['lj31'][1]:g}–{generations['lj31'][2]:g}) and "
               f"{generations['lj38'][0]:g} on LJ38 "
               f"({generations['lj38'][1]:g}–{generations['lj38'][2]:g})")
    anneal_calls = [float(row["evals_to_hit"]) for row in pick(seeds, system="lj31", arm="anneal")]
    gmin_calls = [float(row["evals_to_hit"]) for row in pick(seeds, system="lj31", arm="gmin")]
    audit.cell(path, "LJ31 GMIN median calls from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["evals_median"], median(gmin_calls))
    audit.cell(path, "LJ31 GMIN IQR from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["evals_iqr"], iqr(gmin_calls))
    audit.cell(path, "LJ31 GMIN Mann-Whitney from per-seed reruns",
               pick(rows, system="lj31", arm="gmin")[-1]["mannwhitney_p"],
               mann_whitney_exact_no_ties(gmin_calls, anneal_calls))


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
    constants = load_script("paper_constants", "paper/jctc/scripts/paper_constants.py")
    proven_rgpot = constants.values()["RgpotRevision"]
    proven_engine = "4dac51020e4f1419fa930d95894b581012dabf47681b3a883cba21063f7e8672"
    audit.said("bench/covalent-h2h/gfn2/scc-false-minimum/README.md", proven_rgpot)
    audit.said("bench/covalent-h2h/gfn2/scc-false-minimum/README.md", proven_engine)
    audit.record("rgpot_revision" not in rows[0], "bench/covalent-h2h/gfn2/rows.csv",
                 "rows.csv keeps its original header without rgpot_revision",
                 "absent", "present" if "rgpot_revision" in rows[0] else "absent")
    sidecar = sidecar_rows(audit, GFN2_RGPOT_SIDECAR)
    row_keys = [(r["system"], r["code"], r["seed"], r["code_revision"], r["engine_sha256"], r["host"]) for r in rows]
    side_keys = [(r["system"], r["code"], r["seed"], r["code_revision"], r["engine_sha256"], r["host"]) for r in sidecar]
    audit.record(row_keys == side_keys, GFN2_RGPOT_SIDECAR,
                 "each covalent-h2h gfn2 row has one matching rgpot-revisions.csv sidecar row",
                 f"{len(row_keys)} aligned rows",
                 f"{len(side_keys)} rows, {'aligned' if row_keys == side_keys else 'mismatched'}")
    for s_row in sidecar:
        want_rgpot = proven_rgpot if s_row["engine_sha256"] == proven_engine else ""
        audit.record(s_row.get("rgpot_revision") == want_rgpot, GFN2_RGPOT_SIDECAR,
                     f"{s_row['system']} {s_row['code']} seed {s_row['seed']} rgpot_revision",
                     want_rgpot or "empty", s_row.get("rgpot_revision", "") or "empty")
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
    audit.said(PAPER, f"The newest released tag is v{version}, which is the crate version.")
    tagged = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/v{version}"], cwd=ROOT,
                            capture_output=True, text=True).returncode == 0
    audit.record(tagged, PAPER, "release tag exists", f"v{version}", "tagged" if tagged else "no tag")


def lj_large(audit: Audit) -> None:
    audit.rows("bench/ase-gofee-h2h/lj-large.csv")
    seeds = "bench/ase-gofee-h2h/lj-large-seeds.csv"
    if not (ROOT / seeds).is_file():
        return
    audit.rows(seeds)
    h2h = load_script("h2h_large_gofee", "paper/jctc/scripts/h2h_tables.py")
    audit.said(SI, h2h.gofee_hits_sentence())
    audit.said(SI, h2h.gofee_wall_sentence())


def ase_gofee(audit: Audit) -> None:
    rows = audit.rows("bench/ase-gofee-h2h/results.csv")
    row = pick(rows, system="lj75", arm="agox_gofee", seed="2000")[0]
    audit.cell("bench/ase-gofee-h2h/README.md", "GOFEE LJ75 seed 2000 stop", "budget", row["stop"])
    audit.said("bench/ase-gofee-h2h/README.md",
               f"it spent the full {row['charged']} calls in {float(row['wall_s']):.0f} s, best energy {float(row['best']):.6f}")


def tip4p_panel(audit: Audit) -> None:
    """The TIP4P panel is derived directly from its committed per-seed rows."""
    path = "bench/tip4p-h2h/README-h2h.md"
    rows = audit.rows("bench/tip4p-h2h/rows-h2h-r2.csv")
    sizes = {"(H₂O)₈": "tip4p8", "(H₂O)₁₅": "tip4p15", "(H₂O)₂₀": "tip4p20", "(H₂O)₂₁": "tip4p21", "(H₂O)₂₅": "tip4p25"}
    for line, row in md_table(audit, path, "Size | anneal | GMIN | calls (a/g)"):
        size = sizes[row["Size"]].removeprefix("tip4p")
        arms = {
            arm: pick(rows, molecules=size, arm=arm)
            for arm in ("anneal", "gmin")
        }
        where = f"{path}:{line} tip4p{size}"
        for arm, label in (("anneal", "anneal"), ("gmin", "GMIN")):
            group = arms[arm]
            audit.cell(where, f"{arm} row count", "12", float(len(group)))
            expected_hits = (
                "unscored" if size == "25"
                else f"{sum(int(item['hit']) for item in group)}/{len(group)}"
            )
            audit.cell(where, f"{arm} hits", row[label], expected_hits)
            best, median_best = row[f"{label} best / median"].split(" / ")
            energies = [float(item["best_energy"]) for item in group]
            audit.cell(where, f"{arm} best", best, min(energies))
            audit.cell(where, f"{arm} median best", median_best, median(energies))
        calls_a, calls_g = row["calls (a/g)"].split(" / ")
        audit.cell(where, "anneal median calls", calls_a,
                   median([float(item["calls"]) for item in arms["anneal"]]))
        audit.cell(where, "GMIN median calls", calls_g,
                   median([float(item["calls"]) for item in arms["gmin"]]))
        audit.cell(where, "p(U), no continuity correction", row["p(U)"],
                   mann_whitney(
                       [float(item["best_energy"]) for item in arms["anneal"]],
                       [float(item["best_energy"]) for item in arms["gmin"]],
                       continuity=False,
                   ))


def pinned_hf_revision(audit: Audit, path: str) -> str:
    """Read the literal dataset revision from a fetcher without importing it."""
    for node in ast.parse(audit.text(path)).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if (
            any(isinstance(target, ast.Name) and target.id == "HF_REVISION_DEFAULT"
                for target in targets)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            return value.value
    raise ValueError(f"{path}: no literal HF_REVISION_DEFAULT")


def reproduction_records(audit: Audit) -> None:
    """Every requested panel has one forced command and records its execution context."""
    reproduce = "paper/jctc/REPRODUCE.md"
    reproduce_rules = {"sc_h2h_n12", "tip4p_h2h", "agox_gofee_h2h"}
    campaigns = (
        ("sc_h2h_n12", "scripts/metal_h2h/enter_sc_n12.sh", "sutton-chen-n12", 1, ""),
        ("tip4p_h2h", "scripts/tip4p_h2h/enter_h2h.sh", "tip4p-gmin", 1, ""),
        ("agox_gofee_h2h", "scripts/ase_gofee_h2h/enter_bench.sh", "agox-gofee", 1, ""),
        (
            "water_crest_fair",
            "scripts/water_crest_fair.sh",
            "water-crest-fair-n192",
            8,
            "",
        ),
        ("crest_equal", ("scripts/crest_equal/enter_bench.sh", "scripts/crest_equal/run_panel.py"), "crest-equal-h2o6", 1, ""),
    )
    snake = audit.text("workflow/Snakefile")
    for rule, runner, campaign, cores, suffix in campaigns:
        command = f"snakemake -s workflow/Snakefile {rule} --cores {cores} --force{suffix}"
        if rule in reproduce_rules:
            audit.said(reproduce, command)
        audit.record(command in snake, "workflow/Snakefile", f"{rule} records the exact forced command",
                     command, command if command in snake else "missing")
        runners = (runner,) if isinstance(runner, str) else runner
        text = "\n".join(audit.text(path) for path in runners)
        runner = " and ".join(runners)
        for field in ("sessions.jsonl", "session_metadata.py", campaign) + (("--anneal-repro",) if rule == "water_crest_fair" else ()):
            audit.record(field in text, runner, f"{rule} session metadata names {field}",
                         field, field if field in text else "missing")
    for path in (reproduce, "scripts/water_crest_fair.sh", "workflow/Snakefile", PAPER):
        fixed = "/work/anneal_repro" in audit.text(path)
        audit.record(not fixed, path, "the anneal_repro checkout is a parameter, not a fixed path", "none",
                     "/work/anneal_repro" if fixed else "none")
    audit.said("scripts/water_crest_fair.sh", "REPRO=${ANNEAL_REPRO_ROOT:-$ROOT/../anneal_repro}")
    audit.said("workflow/Snakefile", 'repro=config.get("anneal_repro", ""),')
    text = audit.text(reproduce)
    audit.record("crest_equal" not in text and "water_crest_fair" not in text, reproduce,
                 "withdrawn CREST and hexamer commands are absent from REPRODUCE.md", "absent",
                 "present" if ("crest_equal" in text or "water_crest_fair" in text) else "absent")
    audit.said(reproduce, "Run every command of this file from the repository root.")
    relative = [line.strip() for line in text.splitlines()
                if line.startswith("    python3 scripts/") or line.startswith("    python3 scripts\\")]
    audit.record("Run these from paper/jctc" not in text and not relative, reproduce,
                 "every command runs from the repository root", "none from paper/jctc",
                 " | ".join(relative) or ("paper/jctc" if "Run these from paper/jctc" in text else "none from paper/jctc"))
    for rule in ("crest_equal", "cuh2_fcc_rec"):
        command = f"snakemake -s workflow/Snakefile {rule} --cores 1 --force"
        if rule == "cuh2_fcc_rec":
            audit.said(reproduce, command)
        audit.record(f"rule {rule}:" in snake, "workflow/Snakefile", f"the Snakefile has the rule {rule}", rule,
                     rule if f"rule {rule}:" in snake else "missing")
    output_roots = {
        "SC_H2H_N12_OUTPUT": "/work/metal-sc/output",
        "TIP4P_H2H_OUTPUT": "/work/tip4p-h2h/output",
        "AGOX_GOFEE_OUTPUT": "/work/ase-gofee/output",
        "CUH2_FCC_REC_OUTPUT": "/work/cuh2-fcc-rec",
    }
    for variable, output in output_roots.items():
        audit.said(reproduce, f"{variable}={output}")
    output_flags = (
        "output", "rows-out", "sessions-out", "hosts-out", "check-out",
        "wall-out", "verify-out", "summary", "seeds-out",
    )
    output_pattern = (
        r"--(?:" + "|".join(output_flags) + r")\s+([^\s\\]+)"
    )
    output_paths = re.findall(output_pattern, text)
    tracked = set(git("ls-files").stdout.splitlines())
    unsafe_outputs = [
        path for path in output_paths
        if path in tracked and (
            path.endswith((".csv", ".tsv", ".jsonl"))
            or path.startswith("paper/jctc/logs/")
        )
    ]
    unsafe_outputs.extend(
        path for path in re.findall(r">\s*([^\s\\]+)", text)
        if path in tracked and (
            path.endswith((".csv", ".tsv", ".jsonl"))
            or path.startswith("paper/jctc/logs/")
        )
    )
    audit.record(
        not unsafe_outputs and "--write" not in text,
        reproduce,
        "REPRODUCE commands never write tracked records",
        "fresh outputs only",
        " ".join(unsafe_outputs) or (
            "--write" if "--write" in text else "fresh outputs only"
        ),
    )
    campaign_outputs = {
        "scripts/tip4p_h2h/enter_h2h.sh": (
            "$OUTPUT/sessions.jsonl", "$OUTPUT/rows-h2h.csv",
            "$OUTPUT/h2h-summary.csv",
        ),
        "scripts/metal_h2h/enter_sc_n12.sh": (
            "$OUTPUT/sessions.jsonl", "$OUTPUT/results.csv",
            "$OUTPUT/seeds.csv",
        ),
        "scripts/ase_gofee_h2h/enter_bench.sh": (
            '"\\$OUTPUT/sessions.jsonl"', '--output-dir "\\$OUTPUT"',
        ),
        "scripts/run_cuh2_fcc_rec.sh": (
            "${CUH2_FCC_REC_OUTPUT:-/work/cuh2-fcc-rec}",
        ),
    }
    for path, required in campaign_outputs.items():
        source = audit.text(path)
        missing = [token for token in required if token not in source]
        audit.record(
            not missing,
            path,
            "the reproduction campaign writes records below /work",
            "all /work outputs",
            " ".join(missing) or "all /work outputs",
        )
    recorder = audit.text("scripts/session_metadata.py")
    for field in ('"compilers": compilers()', '"python": {"version": platform.python_version()', '"openmp": openmp_runtime()',
                  '"anneal_repro": repro_checkout(anneal_repro)', '"omp_num_threads": os.environ.get("OMP_NUM_THREADS")'):
        audit.record(field in recorder, "scripts/session_metadata.py", "the session receipt records the build context",
                     field, field if field in recorder else "missing")
    audit.said(reproduce, "the versions\nof rustc, cargo, gfortran and cc (null when one is not installed), the Python\n"
                          "version, and the OpenMP libraries the dynamic loader resolves with\n`OMP_NUM_THREADS`")
    audit.record("scripts/crest_equal/run_panel.py" not in audit.text(PAPER), PAPER,
                 "the paper does not name the crest_equal runner", "absent",
                 "present" if "scripts/crest_equal/run_panel.py" in audit.text(PAPER) else "absent")
    dataset = "71185263eb6a8130d1262fd76d90109f5fabdcf5"
    data_fetcher = "scripts/data/fetch_data.py"
    bench_fetcher = "paper/jctc/scripts/fetch_bench_data.py"
    audit.said(data_fetcher, 'HF_REPO = os.environ.get("HF_REPO", "HaoZeke/anneal-paper-data")')
    audit.said(bench_fetcher, '"HF_DATASET", "HaoZeke/anneal-paper-data"')
    data_revision = pinned_hf_revision(audit, data_fetcher)
    bench_revision = pinned_hf_revision(audit, bench_fetcher)
    audit.record(data_revision == bench_revision, bench_fetcher,
                 "both data fetchers pin the same dataset revision",
                 data_revision, bench_revision)
    audit.cell(data_fetcher, "the pinned paper dataset revision", dataset, data_revision)
    audit.said(reproduce, f"`HaoZeke/anneal-paper-data` at\n`{dataset}`")
    audit.said(reproduce,
               "`python3 scripts/data/fetch_data.py bench/tip4p-h2h/logs/`")
    hf_manifest = list(csv.DictReader(
        (ROOT / "data/hf-files.csv").open(newline="")
    ))
    tip4p_logs = [
        row for row in hf_manifest
        if row["path"].startswith("bench/tip4p-h2h/logs/")
    ]
    audit.record(len(tip4p_logs) == 120, "data/hf-files.csv",
                 "the HF manifest records all TIP4P head-to-head logs",
                 "120", str(len(tip4p_logs)))
    lj_csm = audit.text("paper/jctc/data/lj_csm_sizes/provenance.txt")
    sums = hashlib.sha256((ROOT / "paper/jctc/data/lj_csm_sizes/audit/SHA256SUMS").read_bytes()).hexdigest()
    lj_csm_tokens = lj_csm.replace(".", "").split()
    has_sums = sums in lj_csm_tokens and "audit/SHA256SUMS" in lj_csm_tokens
    audit.record(has_sums, "paper/jctc/data/lj_csm_sizes/provenance.txt",
                 "provenance.txt records the digest of audit/SHA256SUMS", sums,
                 sums if has_sums else "absent")
    nested = "pixi run --locked --frozen snakemake water_crest_fair --cores 8 --force"
    pinned = all(token in audit.text("scripts/water_crest_fair.sh")
                 for token in ("pixi run", "--locked", "--frozen", "water_crest_fair"))
    audit.record(
        pinned,
        "scripts/water_crest_fair.sh",
        "full CREST campaign uses the pinned Pixi workflow",
        nested,
        nested if pinned else "missing",
    )
    audit.said("scripts/tip4p_h2h/enter_h2h.sh", "--rows $OUTPUT/rows-h2h.csv")
    runner = audit.text("scripts/tip4p_h2h/run_h2h.py")
    audit.record("unlink" not in runner, "scripts/tip4p_h2h/run_h2h.py", "the TIP4P-vs-GMIN rows file is append-only",
                 "no unlink", "no unlink" if "unlink" not in runner else "unlink")


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
    polish_path = "bench/metal-cluster-h2h/ptcu-b3d03061/polish-reruns.csv"
    polish = pick(audit.rows(polish_path), system="pt6cu17", host="bench")[0]
    audit.said("bench/metal-cluster-h2h/ptcu-b3d03061/README.md",
               f"gives −{-float(polish['energy_eV']):.8f} eV in `polish-reruns.csv`")
    audit.record(polish["code_revision"] == "be3849e7811634aca67688da70f5413f9d860ec9", polish_path,
                 "Pt6Cu17 polish rerun revision", "be3849e7", polish["code_revision"][:8])
    audit.record(polish["code_tree"] == "93b2b874d245fed5d538386e590c415029243ae5", polish_path,
                 "Pt6Cu17 polish rerun tree", "93b2b874", polish["code_tree"][:8])

    tip = "bench/tip4p-h2h/README.md"
    audit.said(tip, "18.4 µs per GMIN call. The adaptive arm takes 18.3 µs")
    audit.said(tip, "Seed 2004 reaches −623.877 against −623.336 and misses on both.")
    audit.said(tip, "Seeds 2002 and 2004, the two one-contact misses, reach −628.371.")
    audit.cell(tip, "cage wall ratio at 15 molecules", "5.237", 40.21225 / 7.6787855)


def unrecorded_reasons() -> dict[tuple[str, str], str]:
    """Why the revision of each listed row file and column cannot be recovered, from ``UNRECORDED_FILE``."""
    with (ROOT / UNRECORDED_FILE).open(newline="") as handle:
        return {(r["row_file"], r["revision_column"]): r["reason"]
                for r in csv.DictReader(line for line in handle if not line.startswith("#"))}


def unpushed_reasons() -> dict[str, str]:
    """Why each listed revision stays unresolved, from ``UNPUSHED_FILE``."""
    with (ROOT / UNPUSHED_FILE).open(newline="") as handle:
        return {r["revision"]: r["reason"] for r in csv.DictReader(line for line in handle if not line.startswith("#"))}


class Revisions:
    """What the rows cite, and how each cited revision is reached from a pushed ref.

    ``cited`` maps a revision to the row files that cite it and ``gaps`` holds
    the column gaps of each file. A cited revision is ``pushed`` when a remote
    ref contains it, ``aliased`` when a verified alias in ``ALIASES`` stands for
    it, and ``same_tree`` when every ``code_tree`` its rows record is the tree
    of one commit on the remote. ``unresolved`` is what none of these reach.
    """

    def __init__(self, cited: dict[str, set[str]], gaps: dict[str, str], index: RevisionIndex, pushed: set[str],
                 aliased: dict[str, str], same_tree: dict[str, list[str]], unresolved: set[str]) -> None:
        self.cited, self.gaps, self.index = cited, gaps, index
        self.pushed, self.aliased, self.same_tree, self.unresolved = pushed, aliased, same_tree, unresolved

    def how(self, name: str) -> str:
        """Where the resolved revision ``name`` is reached, for the audit's messages."""
        if name in self.pushed:
            where = git("for-each-ref", "--format=%(refname:short)", "--contains", self.index.commit[name], "refs/remotes").stdout.split()
            return "on " + ", ".join(where[:3])
        if name in self.aliased:
            return f"as alias of {self.aliased[name][:8]}"
        if name in self.same_tree:
            return "with the tree of " + ", ".join(c[:8] for c in self.same_tree[name][:3])
        return "unresolved"


def resolve_revisions(audit: Audit) -> Revisions:
    """Reach every revision the files read cite from a pushed ref; the checks are recorded once per set of files read."""
    key = tuple(sorted(audit.read))
    if audit.revisions is not None and audit.revisions_key == key:
        return audit.revisions
    gaps: dict[str, str] = {}
    revisions: dict[str, set[str]] = defaultdict(set)
    trees: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
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
            tree_column = TREE_COLUMNS.get(column)
            for r in rows:
                value = r[column].strip()
                if re.fullmatch(r"[0-9a-f]{7,40}", value) and value not in EXTERNAL_REVISIONS:
                    revisions[value].add(path)
                    if tree_column in r and r[tree_column].strip():
                        trees[value][r[tree_column].strip()].add(path)
        if problems:
            gaps[path] = "; ".join(problems)
    names = sorted(revisions)
    table = alias_table()
    index = alias_index(names, table)
    aliases = verified_aliases(audit, table, index)
    pushed = {n for n in names if index.on_remote(n)}
    aliased = {n: aliases[local] for n in names if n not in pushed for local in aliases if local.startswith(n)}
    same_tree: dict[str, list[str]] = {}
    for name in names:
        recorded = trees.get(name, {})
        if not recorded:
            continue
        where = ", ".join(sorted({p for paths in recorded.values() for p in paths}))
        if len(recorded) > 1:
            audit.record(False, where, f"rows cite revision {name[:8]} with one code_tree",
                         "one tree", ", ".join(sorted(t[:8] for t in recorded)))
            continue
        (tree,) = recorded
        if name in pushed:
            audit.record(index.tree_of(name) == tree, where, f"code_tree of revision {name[:8]} is its tree",
                         tree[:8], (index.tree_of(name) or "no such commit")[:8])
        elif name not in aliased and index.by_tree.get(tree):
            same_tree[name] = index.by_tree[tree]
            audit.record(True, where, f"revision {name[:8]} has the tree of a pushed commit", tree[:8],
                         ", ".join(c[:8] for c in same_tree[name][:3]))
    unresolved = {n for n in names if n not in pushed and n not in aliased and n not in same_tree}
    for name in sorted(unresolved):
        for path in revisions[name]:
            note = f"revision {name[:8]} is not in this repository" if not index.exists(name) \
                else f"revision {name[:8]} is not on a pushed branch"
            gaps[path] = f"{gaps[path]}; {note}" if path in gaps else note
    audit.revisions = Revisions(dict(revisions), gaps, index, pushed, aliased, same_tree, unresolved)
    audit.revisions_key = key
    return audit.revisions


def provenance(audit: Audit) -> dict[str, str]:
    found = resolve_revisions(audit)
    reasons = unpushed_reasons()
    for name in sorted(found.unresolved | (set(reasons) & set(found.cited))):
        unresolved = name in found.unresolved
        audit.record(unresolved and name in reasons, ", ".join(sorted(found.cited[name])), f"unpushed revision {name[:8]}",
                     reasons.get(name, "not listed"), "no pushed equivalent" if unresolved else f"resolved {found.how(name)}")
    return found.gaps


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
        self.by_tree: dict[str, list[str]] = defaultdict(list)
        for line in git("log", "--remotes", "--format=%T %H").stdout.splitlines():
            tree, commit = line.split()
            self.by_tree[tree].append(commit)
        self.trees: dict[str, str] = {}
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

    def tree_of(self, revision: str) -> str | None:
        """The tree of the commit ``revision`` names, or None when this repository has no such commit."""
        if revision not in self.trees and revision in self.commit:
            self.trees[revision] = git("rev-parse", f"{self.commit[revision]}^{{tree}}").stdout.strip()
        return self.trees.get(revision)

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


COVER = "paper/jctc/cover-letter.tex"
COVER_TOC = "paper/jctc/toc-graphic.tex"
TEX_SOURCES = (
    TEX,
    SI,
    COVER,
    COVER_TOC,
    "paper/jctc/macros.tex",
    "paper/jctc/labels-manuscript.tex",
    "paper/jctc/labels-si.tex",
)
# Sentences of the cover letter that carry a digit and are not abstract sentences.
COVER_OTHER = ("This submission replaces the preprint arXiv:2302.02811, deposited in 2023 under the title "
               "``Unified Software Design Patterns for Simulated Annealing.",
               "A table of contents graphic at 3.25~in by 1.75~in accompanies the submission.")


def collapsed(text: str) -> str:
    """Whitespace collapsed so a sentence compares by its words."""
    return " ".join(text.split())


def gmin_body_copies(audit: Audit, panel: object, groups: dict) -> None:
    """Every copy of a GMIN-panel count or wall in the three documents, each against the rows.

    A pinned sentence only proves one copy is right; a stale second copy of the same number elsewhere
    would pass it. Each pattern must match at least once in the manuscript.
    """
    systems = quoted_systems(panel, groups)

    def col(system: str, arm: str, key: str) -> float:
        return median([float(r[key]) for r in groups[(system, arm)]])

    def hits(system: str, arm: str) -> int:
        return hits_of(groups[(system, arm)])

    ahead = sum(hits(s, "anneal") > hits(s, "gmin") for s in systems)
    behind = sum(hits(s, "anneal") < hits(s, "gmin") for s in systems)
    lower = sum(col(s, "anneal", "wall_s") < col(s, "gmin", "wall_s") for s in systems)
    total = str(len(systems))
    of = r"on (\d+) of (?:the )?(\d+)"
    patterns = (
        ("this setting hits more seeds", rf"(?<!GMIN )hits more seeds (?:than GMIN )?{of}", (str(ahead), total)),
        ("GMIN hits more seeds", rf"GMIN hits more seeds (?:than this setting )?{of}", (str(behind), total)),
        ("ties", r"the two tie on (\d+)", (str(len(systems) - ahead - behind),)),
        ("GMIN hits more, short form", r"and GMIN hits more on (\w+)\.", (spelled(behind),)),
        ("lower median wall", rf"lower (?:for this setting )?{of} clusters", (str(lower), total)),
    )
    if "m80r6" in systems:
        patterns = patterns + (
            ("M80 rho=6 wall of this setting", r"calls in a median of (\d+(?:\.\d+)?)~s",
             (panel.show_s(col("m80r6", "anneal", "wall_s")),)),
            ("M80 rho=6 calls and wall of GMIN", r"minimum in a median of (\d+) calls and (\d+(?:\.\d+)?)~s",
             (panel.show_calls(col("m80r6", "gmin", "evals_to_hit")), panel.show_s(col("m80r6", "gmin", "wall_s")))),
        )
    other_panel = re.compile(r"Sutton|TIP4P|Atomic Simulation|\bASE\b")

    def gmin_sentences(path: str) -> str:
        text = audit.text(path)
        for label in ("sec:sc", "sec:tip4p-gmin"):
            text = re.sub(rf"\\label\{{{label}\}}.*?(?=\\(?:sub)?section\{{)", "", text, flags=re.S)
        sentences = re.split(r"(?<=\.)\s+(?=[A-Z\\])", collapsed(text))
        return " ".join(s for s in sentences if not other_panel.search(s))

    documents = (TEX, "paper/jctc/si.tex", "paper/jctc/cover-letter.tex")
    for label, pattern, expected in patterns:
        seen = 0
        for path in documents:
            for match in re.finditer(pattern, gmin_sentences(path)):
                seen += path == TEX
                audit.record(match.groups() == expected, path, f"every copy: {label}", " ".join(expected),
                             " ".join(match.groups()))
        if label != "ties":
            audit.record(seen > 0, PAPER, f"every copy: {label} is stated", "at least 1", str(seen))


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
    audit.record(checked >= 4, COVER, "the cover letter carries the abstract numbers", ">= 4", str(checked))
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

    systems = quoted_systems(panel, groups)
    ahead = [s for s in systems if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")])]
    ours = [s for s in ahead if p_of(s) < 0.05]
    rest = [s for s in ahead if s not in ours]
    behind = [s for s in systems if hits_of(groups[(s, "anneal")]) < hits_of(groups[(s, "gmin")])]
    audit.cell(PAPER, "the significant wins of this setting", "lj38 lj75 lj98", " ".join(ours))
    audit.cell(PAPER, "the wins that are not significant", "lj102 lj103 lj104", " ".join(rest))
    audit.said(PAPER, f"The {len(ours)} are {series(name(s) for s in ours)}.")
    audit.said(PAPER, f"The other {len(rest)} wins of this setting are {series(name(s) for s in rest)}, "
                    f"at \\(p\\) of {series(panel.show_p(p_of(s), tex=False) for s in rest)}, and the GMIN win on "
                    f"{series(name(s) for s in behind)} is not significant either.")
    lower = [s for s in systems if wall(s, "anneal") < wall(s, "gmin")]
    slower = [s for s in systems if s not in lower]
    audit.cell(PAPER, "the two clusters where this setting is slower", "none", " ".join(slower) or "none")
    wall_claim = f"The median wall of a run is lower for this setting on {len(lower)} of the {len(systems)} clusters"
    if slower:
        wall_claim += f" and higher on {series(name(s) for s in slower)}"
    audit.said(PAPER, wall_claim + ".")
    stated = collapse(audit.text(PAPER)).count("The median wall of a run is lower for this setting")
    audit.record(stated == 1, PAPER, "the GMIN wall conclusion is stated once", "1", str(stated))
    audit.cell(PAPER, "GMIN hits 12 of 12 where it is faster", "12", float(hits_of(groups[("m80r6", "gmin")])))
    gmin_body_copies(audit, panel, groups)
    s_ = panel.show_s

    lj38 = ase[("lj38", "recommended")]
    cap = ase_cap(audit, ase, "lj38")
    capped = all(r["stop"] == "wall" for a in ASE_ARMS for r in ase[("lj38", a)])
    audit.record(capped, PAPER, "every ASE run on LJ38 stops at the cap", "every", "every" if capped else "not every")
    audit.cell(PAPER, "LJ38 median wall of this setting in the ASE panel", "0.067", median([float(r["wall_s"]) for r in lj38]))
    audit.cell(PAPER, "LJ38 ASE cap", "4.654", cap)

    gofee_dir = "bench/ase-gofee-h2h"
    summary = {(r["system"], r["arm"]): r for r in audit.rows(f"{gofee_dir}/summary.csv")}
    lj = ("lj38", "lj55", "lj75")
    audit.cell(PAPER, "this setting hits the three clusters on 6 of 6", "6/6 6/6 6/6",
               " ".join(f"{summary[(s, 'paper')]['hits']}/{summary[(s, 'paper')]['n']}" for s in lj))
    audit.cell(PAPER, "ASE basin hopping and GOFEE hit none of the three", "0 0 0 0 0 0",
               " ".join(summary[(s, a)]["hits"] for a in ("ase_bh", "agox_gofee") for s in lj))
    mh = [summary[(s, "ase_mh")]["hits"] for s in lj]
    audit.cell(PAPER, "ASE minima hopping hits none of LJ75", "0", float(mh[2]))
    g = [float(summary[(s, "agox_gofee")]["wall_median_s"]) for s in lj]
    a = [float(summary[(s, "paper")]["wall_median_s"]) for s in lj]
    audit.record(all(x > y for x, y in zip(g, a)), PAPER, "the GOFEE median wall is higher on all three",
                 "higher", "higher" if all(x > y for x, y in zip(g, a)) else "not")
    gofee_rows = audit.rows(f"{gofee_dir}/results.csv")
    gofee_held = bool(gofee_rows) and all(commit_held(r["head"]) for r in gofee_rows)
    if gofee_held:
        audit.said(PAPER, "ASE basin hopping hits none of them, and GOFEE hits none.")
        audit.said(PAPER, f"ASE minima hopping hits LJ\\(_{{38}}\\) on {mh[0]} of 6, LJ\\(_{{55}}\\) on {mh[1]} of 6, and "
                        "LJ\\(_{75}\\) on none.")
        audit.said(PAPER, "This setting hits all three clusters on 6 of 6 seeds.")
        audit.said(PAPER, f"The median wall of a GOFEE run is {s_(g[0])}~s on LJ\\(_{{38}}\\), {s_(g[1])}~s on "
                        f"LJ\\(_{{55}}\\), and {s_(g[2])}~s on LJ\\(_{{75}}\\), against {s_(a[0])}~s, {s_(a[1])}~s, "
                        f"and {s_(a[2])}~s for this setting.")
        audit.said(PAPER, "its median wall is higher than that of this setting on all three")
    rows = gofee_rows
    heads = {r["head"][:8] for r in rows if r["head"] != "unknown"}
    unknown = sum(r["head"] == "unknown" for r in rows)
    if gofee_held:
        audit.said(PAPER, f"The rows of Table~\\ref{{tab:gofee}} span {len(heads)} repository heads.")
        audit.said(
            PAPER,
            f"\\texttt{{results.csv}} records those heads on {len(rows) - unknown} rows.\n"
            f"\\path{{bench/unreachable-revisions.csv}} indexes the other {unknown} rows.",
        )
    mapped = sum(int(r["rows"]) for r in committed_revision_map()
                 if r["row_file"] == f"{gofee_dir}/results.csv" and r["revision"] == UNRECORDED)
    audit.cell(PAPER, "the GOFEE rows with an unrecorded head in the revision map", str(unknown), float(mapped))
    versions = dict(line.split("=", 1) for line in audit.text(f"{gofee_dir}/versions.txt").split() if "=" in line)
    audit.said(PAPER, f"The versions are ASE {versions['ase']} and AGOX {versions['agox']}.")
    audit.cell(PAPER, "the GOFEE and minima-hopping wall cap on LJ75 is 2 hours", "2", float(versions["lj75_mh_gofee_wall_cap_s"]) / 3600)
    if gofee_held:
        audit.said(PAPER, "ASE minima hopping and GOFEE on LJ\\(_{75}\\) also stop at a wall cap of 2 hours")
    other = [r for r in audit.rows(f"{gofee_dir}/asg2-summary.csv") if r["source"] == "measured"]
    audit.cell(SI, "the other panel systems of the GOFEE table", "au55 cu38 cu12au26 sw10 tc10 tip4p8",
               " ".join(dict.fromkeys(r["system"] for r in other)))

    tip_rows_path = "bench/tip4p-h2h/rows-h2h-r2.csv"
    tip_rows = audit.rows(tip_rows_path)
    sizes = ("8", "15", "20", "21", "25")
    groups = {
        (size, arm): pick(tip_rows, molecules=size, arm=arm)
        for size in sizes for arm in ("anneal", "gmin")
    }
    audit.cell(PAPER, "TIP4P rows per size and arm", "12",
               float(min(len(group) for group in groups.values())))
    audit.cell(PAPER, "TIP4P seed range", "2000 2011",
               f"{min(int(row['seed']) for row in tip_rows)} {max(int(row['seed']) for row in tip_rows)}")
    same_starts = all(
        {row["seed"]: row["start_id"] for row in groups[(size, "anneal")]}
        == {row["seed"]: row["start_id"] for row in groups[(size, "gmin")]}
        for size in sizes
    )
    audit.record(same_starts, tip_rows_path,
                 "both TIP4P arms use the same start identifier at each seed",
                 "same", "same" if same_starts else "different")
    machine_ids = {row["machine_id"] for row in tip_rows}
    cores = {row["core"] for row in tip_rows}
    audit.record(len(machine_ids) == 1 and cores == {"0"}, tip_rows_path,
                 "TIP4P r2 uses one machine pinned to core 0",
                 "one machine core 0",
                 f"{len(machine_ids)} machines cores {' '.join(sorted(cores))}")
    abba = all(
        [
            next(row["arm_order"] for row in groups[(size, "anneal")]
                 if row["seed"] == seed)
            for seed in sorted(row["seed"] for row in groups[(size, "anneal")])
        ] == ["anneal,gmin" if index % 2 == 0 else "gmin,anneal"
              for index in range(12)]
        for size in sizes
    )
    audit.record(abba, tip_rows_path, "TIP4P r2 uses ABBA arm order",
                 "ABBA", "ABBA" if abba else "other")
    hit_counts = {
        key: sum(int(row["hit"]) for row in group)
        for key, group in groups.items()
    }
    manuscript_tip = collapse(audit.text(TEX)).split(
        "against GMIN on TIP4P this setting hits", 1
    )[1].split(".", 1)[0]
    manuscript_hits = [int(value) for value in re.findall(
        r"on (\d+) of 12", manuscript_tip
    )]
    audit.cell(TEX, "manuscript TIP4P 15 this-setting hits",
               str(manuscript_hits[0]), hit_counts[("15", "anneal")])
    audit.cell(TEX, "manuscript TIP4P 15 GMIN hits",
               str(manuscript_hits[1]), hit_counts[("15", "gmin")])
    tip_section = audit.text(SI).split("\\section{TIP4P against GMIN}", 1)[1].split(
        "\\section{Alternative stall-exit arms}", 1
    )[0]
    for size in ("8", "15", "20"):
        needle = f"\\((\\mathrm{{H_2O}})_{{{size}}}\\) on "
        claim = tip_section.split(needle, 1)[1]
        match = re.match(
            r"(\d+) of 12 against(?: GMIN on)? (\d+) of 12", claim
        )
        if match is None:
            raise ValueError(f"cannot parse TIP4P prose for {size} molecules")
        audit.cell(SI, f"TIP4P {size} prose this-setting hits",
                   match.group(1), hit_counts[(size, "anneal")])
        audit.cell(SI, f"TIP4P {size} prose GMIN hits",
                   match.group(2), hit_counts[(size, "gmin")])
    audit.said(PAPER, "Against GMIN on TIP4P, this setting hits more seeds at 8, 15, and 20\n"
                     "molecules, and neither arm hits at 21 molecules")
    audit.said(PAPER, "Size 25 has no independent reference and is unscored")
    audit.record(
        all(hit_counts[(size, "anneal")] > hit_counts[(size, "gmin")]
            for size in ("8", "15", "20"))
        and all(hit_counts[(size, arm)] == 0
                for size in ("21",) for arm in ("anneal", "gmin")),
        tip_rows_path,
        "the TIP4P discussion matches the per-seed hit rows",
        "anneal leads at 8, 15, 20; both miss at 21; 25 is unscored",
        "matches",
    )
    wall_path = "bench/tip4p-h2h/wall-r2.csv"
    with (ROOT / wall_path).open(newline="") as handle:
        wall_rows = {
            row["molecules"]: row for row in csv.DictReader(handle)
        }
    r2_runner = load_script(
        "tip4p_h2h_r2_audit", "bench/tip4p-h2h/run_h2h_r2.py"
    )
    for size in sizes:
        summary = wall_rows[size]
        anneal_walls = [
            float(row["wall_s"]) for row in groups[(size, "anneal")]
        ]
        gmin_walls = [
            float(row["wall_s"]) for row in groups[(size, "gmin")]
        ]
        exact_p, method = r2_runner.mann_whitney_u_two_sided_exact(
            anneal_walls, gmin_walls
        )
        audit.cell(wall_path, f"TIP4P {size} r2 anneal n",
                   summary["anneal_n"], len(anneal_walls))
        audit.cell(wall_path, f"TIP4P {size} r2 GMIN n",
                   summary["gmin_n"], len(gmin_walls))
        expected_hits = "unscored" if size == "25" else None
        audit.cell(wall_path, f"TIP4P {size} r2 anneal hits",
                   summary["anneal_hits_n"],
                   expected_hits or f"{hit_counts[(size, 'anneal')]}/12")
        audit.cell(wall_path, f"TIP4P {size} r2 GMIN hits",
                   summary["gmin_hits_n"],
                   expected_hits or f"{hit_counts[(size, 'gmin')]}/12")
        for arm, walls in (("anneal", anneal_walls), ("gmin", gmin_walls)):
            calls = [
                float(row["calls"]) for row in groups[(size, arm)]
            ]
            energies = [
                float(row["best_energy"]) for row in groups[(size, arm)]
            ]
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} calls median",
                       summary[f"{arm}_calls_median"], median(calls))
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} energy best",
                       summary[f"{arm}_energy_best_kj_mol"], min(energies))
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} energy median",
                       summary[f"{arm}_energy_median_kj_mol"], median(energies))
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} wall median",
                       summary[f"{arm}_wall_median_s"], median(walls))
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} wall IQR",
                       summary[f"{arm}_wall_iqr_s"],
                       type7(walls, 0.75) - type7(walls, 0.25))
            audit.cell(wall_path, f"TIP4P {size} r2 {arm} wall maximum",
                       summary[f"{arm}_wall_max_s"], max(walls))
        audit.record(summary["mwu_method"] == method, wall_path,
                     f"TIP4P {size} exact wall test method",
                     method, summary["mwu_method"])
        audit.cell(wall_path, f"TIP4P {size} exact wall p",
                   summary["wall_mwu_p"], exact_p)
    audit.said(TEX, "In the 120-run ABBA comparison on one machine pinned to core 0")
    audit.said(SI, "The 120-run ABBA comparison uses seeds 2000--2011 on one machine pinned to core 0")
    audit.said(SI, "with the arm order alternating across seeds")
    audit.cell(tip_rows_path, "TIP4P ABBA comparison runs", "120", float(len(tip_rows)))
    paper_tip4p = collapse(audit.text(TEX) + "\n" + tip_section)
    audit.record("separate cores" not in paper_tip4p.lower(), PAPER,
                 "TIP4P r2 does not claim separate cores",
                 "absent", "present" if "separate cores" in
                 paper_tip4p.lower() else "absent")
    audit.said(TEX, "its median walls are 0.022 versus 1.952~s at 8 molecules, "
               "11.667 versus\n48.016~s at 15, and 21.541 versus 83.630~s at 20.")
    audit.said(TEX, "The exact rank-enumeration Mann--Whitney wall-time \\(p\\) is\n"
               "\\(7.4\\times10^{-7}\\) at sizes 8, 15, 20, and 21.")
    audit.said(TEX, "their median costs\n"
               "are 13.156 versus 40.548~\\(\\mu\\)s per call.")
    h2h_runner = audit.text("scripts/tip4p_h2h/run_h2h.py")
    shared_hit_tol = (
        "from target_tol import DEFAULT_HIT_TOL, energy_hits" in h2h_runner
        and "HIT_TOL = DEFAULT_HIT_TOL\n" in h2h_runner
    )
    audit.record(shared_hit_tol, "scripts/tip4p_h2h/run_h2h.py", "the TIP4P-vs-GMIN hit tolerance",
                 "shared 0.001 kJ/mol", "shared 0.001 kJ/mol" if shared_hit_tol else "other")
    sessions_path = "bench/tip4p-h2h/sessions-r2.jsonl"
    sessions = [json.loads(line) for line in audit.text(sessions_path).splitlines()]
    session_tols = {(row.get("hit_tol"), row.get("hit_tol_unit")) for row in sessions}
    audit.record(session_tols == {(0.001, "kJ/mol")}, sessions_path,
                 "every ABBA session records its hit tolerance, 0.001 kJ/mol",
                 "0.001 kJ/mol", sorted(map(str, session_tols)))
    macro_tol = re.search(r"\\newcommand\{\\TIPFourPHitToleranceKJ\}\{([0-9.]+)\}",
                          audit.text("paper/jctc/macros.tex"))
    audit.record(macro_tol is not None and float(macro_tol.group(1)) == 0.001, "paper/jctc/macros.tex",
                 "the manuscript hit tolerance equals the recorded session tolerance, 0.001 kJ/mol",
                 "0.001", macro_tol.group(1) if macro_tol else "missing")
    audit.said("bench/tip4p-h2h/README-h2h.md", "reference plus 0.001 kJ/mol")
    audit.said("paper/jctc/REPRODUCE.md", "python3 bench/tip4p-h2h/run_h2h_r2.py run")
    audit.said("paper/jctc/REPRODUCE.md", "--hit-tol 0.001")
    exact_wall_source = "mann_whitney_u_two_sided_exact" in audit.text(
        "bench/tip4p-h2h/run_h2h_r2.py"
    )
    audit.record(exact_wall_source,
                 "bench/tip4p-h2h/run_h2h_r2.py",
                 "the TIP4P r2 runner implements the exact wall test",
                 "exact", "exact" if exact_wall_source else "missing")
    tblite_receipt_path = (
        "paper/jctc/data/water_v45b/rerun-9a30f319/toolchain.json"
    )
    tblite_receipt = json.loads(audit.text(tblite_receipt_path))
    expected_tblite = {
        "package": "tblite",
        "version": "0.7.0",
        "source_commit": "fa8a4416e8fe093d0075bc10ac875494c2a449a9",
        "metadata": "fpm.toml",
    }
    shown_tblite = {
        key: tblite_receipt.get(key)
        for key in expected_tblite
    }
    audit.record(shown_tblite == expected_tblite, tblite_receipt_path,
                 "the water rerun records tblite package metadata",
                 json.dumps(expected_tblite, sort_keys=True),
                 json.dumps(shown_tblite, sort_keys=True))
    audit.said("paper/jctc/data/water_v45b/rerun-9a30f319/README.md",
               "through tblite\n0.7.0")
    conversion_paths = (
        "scripts/covalent_h2h/gfn2_crest.py",
        "scripts/h2o16_crest/run_capped.py",
        "scripts/tric_quench/quench_compare.py",
        "scripts/h2o20_goat_campaign.sh",
        "scripts/h2o20_goat_reference.sh",
        "paper/jctc/scripts/water_census.py",
        "examples/ase_objective.py",
    )
    for path in conversion_paths:
        source = audit.text(path)
        uses_constant = "from codata2018 import" in source and "HARTREE_EV" in source
        hard_coded = re.search(r"\b27\.21138\d*\b", source)
        audit.record(uses_constant and hard_coded is None, path,
                     "Hartree-to-eV conversion comes from codata2018",
                     "codata2018", "codata2018" if uses_constant
                     and hard_coded is None else "hard-coded")
    audit.said("paper/jctc/logs/water/VERSIONS", "xtb version 6.7.1")
    for hex_note in ("paper/jctc/data/hq_sci_h2o6/SOFTWARE_TAG", "paper/jctc/data/w6gofee4k/README"):
        audit.record("6.7.1" not in audit.text(hex_note), hex_note,
                     "the hexamer geometry file keeps its protocol text",
                     "protocol text", "xtb banner" if "6.7.1" in audit.text(hex_note) else "protocol text")
    audit.said(TEX, "\\texttt{data/hq\\_sci\\_h2o6/}\nrecords the recommended GFN2-xTB protocol and the eight seed energies,\n"
                    "\\texttt{data/w6gofee4k/} records the GOFEE and recommended hexamer\n"
                    "geometries at 4000 force evaluations, and \\texttt{logs/water/VERSIONS}\n"
                    "records xtb 6.7.1")
    audit.said("bench/tip4p-h2h/README-h2h.md", "The timed anneal binary was built from `7da10659`")
    rates = audit.rows("bench/tip4p-water-rates/rates.csv")
    for n, budget, want in (("10", "500000", "6/6"), ("15", "500000", "1/6"), ("20", "2000000", "1/6")):
        audit.cell(PAPER, f"TIP4P {n} at b47d442a", want, frac(pick(rates, n=n, budget=budget)))
    audit.said("bench/tip4p-water-rates/README.md", "`rates.csv`: the rows of b47d442a")
    audit.said("bench/tip4p-water-rates/README.md", "`rates-a24ba760.csv`: the rows of a24ba760")
    audit.said(PAPER, "against GMIN on TIP4P this setting hits \\((\\mathrm{H_2O})_{15}\\) on 7 of 12\nand GMIN on 0 of 12.")
    audit.said("bench/tip4p-h2h/README-h2h.md",
               "Size 25 has no independent reference and is\nunscored")

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
    audit.said("bench/gmin-panel/gmin-source.txt", "gfortran=13.3.0")
    audit.said("bench/gmin-panel/gmin-source.txt", "/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.12.0")
    audit.said("bench/gmin-panel/README.md", "(`taskset -c 0`, one thread, every other process moved\nto cores 1-7)")
    audit.said("paper/jctc/logs/water/HOST-TOPOLOGY", "cpu=Intel Xeon Processor, family 6, model 207")
    text = audit.text(PAPER)
    for needle in ("host name\n\\texttt{bench}", "8 cores of an Intel Xeon Processor, family 6, model 207",
                   "2400~MHz under KVM", "Ubuntu 24.04", "Rust 1.88.0 and gfortran 13.3.0",
                   "reference BLAS and LAPACK 3.12.0",
                   "which a copper-slab run on\nthat host wrote"):
        audit.record(needle in text, PAPER, "the host paragraph says", needle.replace("\n", " "), "present")
    commands = {
        "bash scripts/gmin_panel/build_gmin.sh": "scripts/gmin_panel/build_gmin.sh",
        "GMIN_PANEL_OUT=/work/gmin-panel/output \\\n  snakemake -s workflow/Snakefile gmin_panel": "workflow/Snakefile",
        "GMIN_PANEL_OUT=/work/gmin-panel/output \\\n  bash scripts/gmin_panel/enter_bench.sh --summarize --tex \\\n"
        "  --summary /work/gmin-panel/output/summary.csv \\\n  --readme /work/gmin-panel/output/README.md \\\n"
        "  --manuscript /work/gmin-panel/output/manuscript.tex": "scripts/gmin_panel/enter_bench.sh",
        "ASE_PANEL_OUT=/work/ase-panel/output \\\n  bash scripts/ase_panel/enter_bench.sh --system lj38\n":
            "scripts/ase_panel/enter_bench.sh",
        "ASE_NONLJ_OUT=/work/ase-nonlj/output \\\n  pixi run --locked --frozen bash scripts/ase_nonlj/enter_bench.sh\n":
            "scripts/ase_nonlj/enter_bench.sh",
        "H2H_TABLES_OUTPUT=/work/h2h-tables \\\n  python3 paper/jctc/scripts/h2h_tables.py":
            "paper/jctc/scripts/h2h_tables.py",
        "SC_H2H_N12_OUTPUT=/work/metal-sc/output \\\n  snakemake -s workflow/Snakefile sc_h2h_n12 --cores 1 --force":
            "workflow/Snakefile",
        "TIP4P_H2H_OUTPUT=/work/tip4p-h2h/output \\\n  snakemake -s workflow/Snakefile tip4p_h2h --cores 1 --force":
            "workflow/Snakefile",
        "AGOX_GOFEE_OUTPUT=/work/ase-gofee/output \\\n  snakemake -s workflow/Snakefile agox_gofee_h2h --cores 1 --force":
            "workflow/Snakefile",
    }
    for command, path in commands.items():
        audit.record(command in text and (ROOT / path).exists(), PAPER, "the command names a script that exists",
                     command, path)
    audit.said("workflow/Snakefile", "rule gmin_panel:")
    audit.said("scripts/gmin_panel/enter_bench.sh", "--summarize")
    audit.said("scripts/ase_panel/enter_bench.sh", "lj38|lj55|lj98|lj104|lj150|au55")
    audit.said("scripts/ase_nonlj/enter_bench.sh", "m38r6")
    audit.said("scripts/ase_panel/enter_bench.sh", ': "${ASE_PANEL_REVISION:?')
    audit.said("scripts/ase_panel/enter_bench.sh", "ROWS=${ASE_PANEL_ROWS:-$DEFAULT_ROWS}")
    audit.said("scripts/ase_nonlj/enter_bench.sh", 'echo "keep existing $SYSTEM $ARM seed $SEED"')
    audit.said(PAPER, "Both ASE panels can be searched again from this repository")
    audit.said(PAPER, "\\path{bench/unreachable-revisions.csv} lists each revision with the rows that cite it.")
    morse_revs = {r["code_revision"] for r in audit.rows("bench/ase-nonlj/rows.csv")}
    newest = "9a30f3190b3bb7040c675a904f8388eb68224151"
    audit.record(newest in morse_revs, PAPER, "the Morse ASE rows carry the revision the commands build", newest[:8],
                 newest[:8] if newest in morse_revs else " ".join(sorted(r[:8] for r in morse_revs)))
    held = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{newest}^{{commit}}"], check=False).returncode == 0
    audit.record(held, PAPER, "the Morse ASE revision is in this checkout", "held", "held" if held else "missing")
    audit.said("scripts/ase_nonlj/enter_bench.sh", 'ROWS=${ASE_NONLJ_ROWS:-$ROOT/bench/ase-nonlj/rows.csv}')
    audit.said("scripts/ase_nonlj/enter_bench.sh", "set ASE_NONLJ_REVISION to a reachable commit SHA")
    audit.said(PAPER, "git worktree add /work/ase-9a30f319 9a30f3190b3bb7040c675a904f8388eb68224151")
    audit.said("paper/jctc/scripts/water_h2h.py", 'command = f"gfn2_water_h2h {MOLECULES} {budget} {seed} {outdir.name}"')
    audit.said("paper/jctc/scripts/water_h2h.py", 'paper_dir = LOGS / f"anneal-paper4000-seed{seed}"')
    audit.said("paper/jctc/logs/water/crest-hexamer-seed0.log", "$ path start.xyz -T 1")
    audit.said("paper/jctc/logs/water/VERSIONS", "polish=xtb --gfn 2 --opt vtight --acc 0.01 --iterations 500")
    for recorder_path in (
        "paper/jctc/scripts/water_h2h.py",
        "paper/jctc/scripts/hexamer_xtb680_repolish.py",
        "scripts/crest_equal/run_panel.py",
        "scripts/covalent_h2h/gfn2_crest.py",
        "scripts/h2o16_crest/run_capped.py",
    ):
        audit.record("rgpot_revision" in audit.text(recorder_path), recorder_path,
                     "GFN2 recorder includes rgpot_revision", "rgpot_revision",
                     "present" if "rgpot_revision" in audit.text(recorder_path) else "absent")
    sidecar_order = (
        ("scripts/crest_equal/run_panel.py",
         "append_rows(sidecar_path, RGPOT_SIDECAR_FIELDS",
         "append_rows(rows_path, ROW_FIELDS"),
        ("scripts/covalent_h2h/gfn2_crest.py",
         "append_rows(\n                sidecar,",
         "append_rows(args.rows, ROW_FIELDS + PROVENANCE_FIELDS"),
    )
    for recorder_path, sidecar_write, row_write in sidecar_order:
        recorder = audit.text(recorder_path)
        ordered = sidecar_write in recorder and row_write in recorder and recorder.index(sidecar_write) < recorder.index(row_write)
        audit.record(ordered, recorder_path, "rgpot sidecar is written before rows.csv",
                     "sidecar then rows", "sidecar then rows" if ordered else "wrong order")
    audit.said(TEX, "\\path{bench/tip4p-h2h/rows-h2h-r2.csv}")
    audit.said(TEX, "\\path{bench/tip4p-h2h/wall-r2.csv}")
    audit.said(COVER, "the TIP4P table from its committed per-seed rows.")
    audit.said(PAPER, "\\texttt{paper/jctc/scripts/ablation\\_table.py} writes Table~\\msref{tab:ablation}")
    audit.record((ROOT / "paper/jctc/scripts/ablation_table.py").exists(), PAPER, "ablation_table.py exists", "exists", "exists")

    # Title, affiliation, spelling, and the reproducibility archive DOI.
    cover = "paper/jctc/cover-letter.tex"
    title = "One basin-hopping setting for Lennard-Jones,\nMorse, gold, and water clusters"
    audit.said(PAPER, "\\title{" + title + "}")
    audit.said(SI, "One basin-hopping setting for Lennard-Jones,\nMorse, gold, and water clusters")
    rohit_affiliations = (
        "\\author{Rohit Goswami}\n"
        "\\affiliation{SURF B.V., Amsterdam}\n"
        "\\alsoaffiliation{TurtleTech ehf., Reykjavik, Iceland}\n"
    )
    for path in (TEX, SI):
        rohit_block = audit.text(path).split("\\author{Rohit Goswami}", 1)[1].split("\\author{Ruhila Goswami}", 1)[0]
        shown = "\\author{Rohit Goswami}" + rohit_block
        audit.record(shown == rohit_affiliations, path,
                     "Rohit's affiliations are SURF first and TurtleTech second",
                     rohit_affiliations.strip(), shown.strip())
        audit.record("SURF BV" not in audit.text(path), path, "the affiliation is spelled one way", "SURF B.V.", "SURF B.V.")
    cover_address = (
        "\\noindent Rohit Goswami\\\\\n"
        "SURF B.V., Amsterdam, The Netherlands\\\\\n"
        "TurtleTech ehf., Reykjavik, Iceland\\\\\n"
        "Email: \\url{rgoswami@ieee.org}"
    )
    audit.said(COVER, cover_address)
    citation_rohit = audit.text("CITATION.cff").split('given-names: "Rohit"', 1)[1].split("- family-names:", 1)[0]
    audit.record('affiliation: "SURF B.V.; TurtleTech ehf"' in citation_rohit,
                 "CITATION.cff", "Rohit's citation affiliations are SURF first and TurtleTech second",
                 "SURF B.V.; TurtleTech ehf",
                 citation_rohit.strip())
    zenodo_creators = json.loads(audit.text(".zenodo.json"))["creators"]
    zenodo_rohit = next(creator for creator in zenodo_creators
                        if creator["name"] == "Goswami, Rohit")
    audit.record(zenodo_rohit.get("affiliation") == "SURF B.V.; TurtleTech ehf",
                 ".zenodo.json",
                 "Rohit's Zenodo affiliations are SURF first and TurtleTech second",
                 "SURF B.V.; TurtleTech ehf",
                 zenodo_rohit.get("affiliation", "missing"))
    audit.record("\\todo{" not in audit.text(PAPER), PAPER, "no \\todo remains in the manuscript", "none", "none")
    notes = audit.text(TEX).split("\\section*{Notes}", 1)[-1].split("\\begin{suppinfo}", 1)[0]
    interest = "The authors declare no competing financial interest."
    decl_len = 345
    decl_sha256 = "c187850a0ee54424df627c2b652f6a8c956865e13698f9a6308a506917c5fbfa"
    notes_flat = collapse(notes)
    audit.said(TEX, interest)
    decl_present = any(
        hashlib.sha256(notes_flat[i:i + decl_len].encode("utf-8")).hexdigest() == decl_sha256
        for i in range(max(0, len(notes_flat) - decl_len + 1))
    )
    audit.record(
        decl_present,
        TEX,
        "the Notes declaration matches the sealed text",
        decl_sha256[:12],
        decl_sha256[:12] if decl_present else "changed",
    )
    after = notes_flat[len(interest):].lstrip() if notes_flat.startswith(interest) else ""
    placed = (
        len(after) >= decl_len
        and hashlib.sha256(after[:decl_len].encode("utf-8")).hexdigest() == decl_sha256
    )
    audit.record(
        placed,
        TEX,
        "the declaration follows the competing-interest sentence",
        "immediately after",
        "immediately after" if placed else "elsewhere",
    )
    for path in (TEX, COVER):
        audit.said(path, "The concept DOI of the \\texttt{anneal\\_repro} reproducibility archive is \\DOI{}.")
    avail = audit.text(TEX).split("\\section*{Data and software availability}", 1)[-1]
    for script in re.findall(r"\\path\{([^}]+)\}", avail):
        audit.record((ROOT / script).is_file(), PAPER, f"availability names an existing script {script}", "present", "present" if (ROOT / script).is_file() else "missing")
    for path in (PAPER, SI, cover):
        found = sorted(set(british_words(audit.text(path))))
        audit.record(not found, path, "American spelling", "none", " ".join(found) or "none")
    lf = load_script("landfold_panels_tex", "paper/jctc/scripts/landfold_panels.py")
    audit.said(PAPER, lf.caption_tex())
    raster = lf.raster_letters()
    audit.said(PAPER, f"Panels (a) to ({chr(ord(raster[0]) - 1)}) are drawn as vectors from the landscape tables.")
    sides = set()
    for png in (ROOT / "paper/jctc/data/landfold_maps_v7").rglob("landscape.png"):
        width, height = struct.unpack(">II", png.read_bytes()[16:24])
        sides.add(width if width == height else -1)
    side = sides.pop() if len(sides) == 1 else float("nan")
    audit.said(PAPER, f"Panels ({raster[0]}) to ({raster[-1]}) redraw the sealed {side}-pixel maps of the campaigns as "
                    "vector rectangles, one per run of equal pixels, so they keep the resolution of those maps.")
    landfold_caption = audit.text(SI).split(
        "\\includegraphics[width=\\textwidth,height=0.58\\textheight", 1
    )[1].split("\\label{fig:landfold-maps}", 1)[0]
    forbidden_landfold = [
        phrase for phrase in ("incomplete", "not promoted", "1{,}184{,}080{,}000")
        if phrase in landfold_caption
    ]
    audit.record(
        not forbidden_landfold,
        SI,
        "the landfold caption describes only panels that are drawn",
        "drawn panels only",
        " ".join(forbidden_landfold) or "drawn panels only",
    )
    figure = (ROOT / "paper/jctc/figs/landfold_maps.pdf").read_bytes()
    audit.record(len(raster) == 6, "paper/jctc/figs/landfold_maps.pdf",
                 "six source-raster maps are redrawn with vector primitives", "6", str(len(raster)))
    audit.record(b"/Subtype /Image" not in figure, "paper/jctc/figs/landfold_maps.pdf",
                 "the landfold PDF embeds no raster image", "none",
                 "none" if b"/Subtype /Image" not in figure else "image XObject")
    audit.record(raster == "".join(chr(c) for c in range(ord(raster[0]), ord(raster[-1]) + 1)), PAPER,
                 "the raster panels of the landfold figure are one run of letters", "contiguous", raster)


SENTENCE_END = re.compile(r"[.?!](?![\d])")
FORBIDDEN_RERUN = re.compile(r"\\rerun\b|\[rerun", re.IGNORECASE)
FORBIDDEN_PENDING = re.compile(r"\bpending\b|to be assigned|\btbd\b", re.IGNORECASE)
DOI_DEF = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\{\\DOI\}|\\def\\DOI\b")


def rerun_check(audit: Audit, pre_submission: bool = False) -> None:
    """Forbid \\rerun, [RERUN, pending, to be assigned, and TBD in TeX sources, and check \\DOI."""
    _ = pre_submission
    doi_defs = [path for path in TEX_SOURCES if DOI_DEF.search(audit.text(path))]
    audit.record(doi_defs == ["paper/jctc/macros.tex"], "paper/jctc/macros.tex",
                 "\\DOI macro is defined in one place", "paper/jctc/macros.tex", " ".join(doi_defs) or "none")
    doi_value = re.search(r"\\newcommand\{\\DOI\}\{([^\n]+)\}", audit.text("paper/jctc/macros.tex"))
    audit.record(bool(doi_value and doi_value.group(1).strip()), "paper/jctc/macros.tex",
                 "\\DOI macro has a non-empty value", "non-empty",
                 doi_value.group(1).strip() if doi_value else "empty")
    citation_doi = re.search(
        r"reproducibility archive concept DOI ([0-9]+\.[0-9]+/[A-Za-z0-9.]+)",
        audit.text("CITATION.cff"),
    )
    macro_doi = re.search(
        r"doi\.org/([0-9]+\.[0-9]+/[A-Za-z0-9.]+)",
        doi_value.group(1) if doi_value else "",
    )
    citation_value = citation_doi.group(1) if citation_doi else "missing"
    macro_value = macro_doi.group(1) if macro_doi else "missing"
    audit.record(citation_value == macro_value, "paper/jctc/macros.tex",
                 "\\DOI matches the reproducibility archive concept DOI in CITATION.cff",
                 citation_value, macro_value)
    constants = load_script("paper_constants_check", "paper/jctc/scripts/paper_constants.py")
    audit.record(not constants.stale(), "paper/jctc/macros.tex",
                 "generated paper constants match the library and Cargo.lock",
                 "current", "current" if not constants.stale() else "stale")
    audit.said(TEX, "E \\le E_{\\mathrm{ref}} + \\TIPFourPHitToleranceKJ")
    kcal_literal = "4" + ".184"
    audit.record("kcal" not in audit.text(TEX) and kcal_literal not in audit.text(TEX),
                 TEX, "the TIP4P hit tolerance is stated once, in kJ/mol, through its macro",
                 "macro only", "literals" if kcal_literal in audit.text(TEX) else "macro only")
    for path in TEX_SOURCES:
        hits = sorted({m.group(0) for m in FORBIDDEN_RERUN.finditer(audit.text(path))})
        audit.record(not hits, path, "no \\rerun or [RERUN remains", "none", " ".join(hits) or "none")
    for path in TEX_SOURCES:
        hits = sorted({m.group(0) for m in FORBIDDEN_PENDING.finditer(audit.text(path))})
        audit.record(not hits, path, "no pending, to be assigned, or TBD remains in TeX",
                     "none", " | ".join(hits) or "none")


REVISION_MAP = "bench/unreachable-revisions.csv"
REVISION_MAP_COLUMNS = ("revision", "why", "row_file", "system", "arm", "revision_column", "rows", "seed_min", "seed_max")


def revision_map_rows(audit: Audit) -> list[dict[str, str]]:
    """Each unreachable revision, then each unrecorded one, with the row groups of the files read that cite it."""
    unresolved = resolve_revisions(audit).unresolved
    reasons = unpushed_reasons()
    unrecorded = unrecorded_reasons()
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for path, rows in audit.read.items():
        columns = set(rows[0]) if rows else set()
        for row in rows:
            for column in REVISION_COLUMNS:
                if column not in columns:
                    continue
                value = row[column].strip()
                if value in ("", "unknown"):
                    value = UNRECORDED
                elif value not in unresolved:
                    continue
                key = (value, path, row.get("system", row.get("board", "")), row.get("arm", row.get("code", "")), column)
                groups[key].append(row)
    out = []
    for (revision, path, system, arm, column), rows in sorted(groups.items()):
        seeds = [int(r["seed"]) for r in rows if r.get("seed", "").strip().lstrip("-").isdigit()]
        why = unrecorded.get((path, column), "not listed") if revision == UNRECORDED else reasons.get(revision, "not listed")
        out.append({"revision": revision, "why": why, "row_file": path, "system": system, "arm": arm,
                    "revision_column": column, "rows": str(len(rows)),
                    "seed_min": str(min(seeds)) if seeds else "", "seed_max": str(max(seeds)) if seeds else ""})
    return out


def committed_revision_map() -> list[dict[str, str]]:
    """The immutable map with its append-only updates applied."""
    with (ROOT / REVISION_MAP).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_key = {
        tuple(row[name] for name in
              ("revision", "row_file", "system", "arm", "revision_column")): row
        for row in rows
    }
    updates = ROOT / "bench/unreachable-revisions-updates.csv"
    if updates.is_file():
        with updates.open(newline="") as handle:
            for update in csv.DictReader(handle):
                key = tuple(update[name] for name in
                            ("revision", "row_file", "system", "arm", "revision_column"))
                if update["status"] == "superseded":
                    by_key.pop(key, None)
                elif update["status"] == "active":
                    by_key[key] = {name: update[name] for name in REVISION_MAP_COLUMNS}
                else:
                    raise ValueError(f"{updates}: invalid status {update['status']!r}")
    return sorted(by_key.values(), key=lambda row: tuple(row[name] for name in REVISION_MAP_COLUMNS))


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
    rows = sorted(revision_map_rows(audit),
                  key=lambda row: tuple(row[name] for name in REVISION_MAP_COLUMNS))
    committed = committed_revision_map()
    audit.record(committed == rows, REVISION_MAP, "map plus updates equals the cited rows",
                 "committed ledgers", f"{len(rows)} row groups")
    unresolved = resolve_revisions(audit).unresolved
    mapped = {r["revision"] for r in rows} - {UNRECORDED}
    audit.record(mapped == unresolved, REVISION_MAP, "every unreachable revision is mapped",
                 f"{len(unresolved)} revisions", f"{len(mapped)} mapped")
    groups = {(r["row_file"], r["revision_column"]) for r in rows if r["revision"] == UNRECORDED}
    for key, why in sorted(unrecorded_reasons().items()):
        audit.record(key in groups, UNRECORDED_FILE, f"{key[0]} {key[1]} is unrecorded on some rows", why,
                     "listed" if key in groups else "no row group")
    for key in sorted(groups - set(unrecorded_reasons())):
        audit.record(False, UNRECORDED_FILE, f"{key[0]} {key[1]} is unrecorded on some rows", "a reason", "not listed")


def manuscript_followups(audit: Audit) -> None:
    """The cluster-preset hits, the wall-cap reconciliation, the GMIN settings, and the M55 disclosures."""
    ase = ase_groups(audit)
    _panel, gmin = gmin_panel_groups(audit)
    names = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj98": "LJ\\(_{98}\\)", "lj104": "LJ\\(_{104}\\)",
             "lj150": "LJ\\(_{150}\\)", "au55": "Au\\(_{55}\\)"}
    hits = [hits_of(ase[(system, "anneal")]) for system in ASE_PANEL]
    panel = load_script("ase_panel_followups", "scripts/ase_panel/summarize.py")
    if panel.tex_table():
        audit.said(PAPER, f"Its hits are {', '.join(map(str, hits[:-1]))}, and {hits[-1]} of 12 on "
                        f"{', '.join(names[s] for s in ASE_PANEL[:-1])}, and {names['au55']}.")
    audit.cell(PAPER, "cluster-preset arm rows per system", "12", float(len(ase[("lj38", "anneal")])))

    summary = audit.rows("bench/ase-recommended/summary.csv")
    caps = {r["system"]: float(r["wall_cap_s"]) for r in summary if r["arm"] == "recommended"}
    gofee = {(r["system"], r["arm"]): r for r in audit.rows("bench/ase-gofee-h2h/summary.csv")}
    mh38, mh55 = gofee[("lj38", "ase_mh")], gofee[("lj55", "ase_mh")]
    if panel.tex_table():
        audit.said(PAPER, f"ASE minima hopping on 0 of 12 seeds at LJ\\(_{{38}}\\) and LJ\\(_{{55}}\\), because its runs stop at wall caps "
                        f"of {caps['lj38']:.3f} and {caps['lj55']:.3f}~s.")
        audit.said(PAPER, f"They hit LJ\\(_{{38}}\\) on {mh38['hits']} of {mh38['n']} seeds in a median wall of "
                        f"{float(mh38['wall_median_s']):.1f}~s and LJ\\(_{{55}}\\) on {mh55['hits']} of {mh55['n']} in "
                        f"{float(mh55['wall_median_s']):.1f}~s, both far beyond those caps.")
    audit.cell(PAPER, "gofee ASE MH LJ38 and LJ55 are uncapped", "0 0", f"{mh38['capped']} {mh55['capped']}")
    audit.cell(PAPER, "gofee walls exceed the panel caps", "1", float(float(mh38["wall_median_s"]) > caps["lj38"]
                                                                  and float(mh55["wall_median_s"]) > caps["lj55"]))
    for system in ("lj38", "lj55"):
        audit.cell(PAPER, f"tab:ase ASE MH {system} hits", "0", float(hits_of(ase[(system, "ase_mh")])))

    g98, g75 = gmin[("lj98", "gmin")], gmin[("lj75", "gmin")]
    near98 = sum(abs(float(r["best"]) + 543.6429) < 1e-3 for r in g98)
    other98 = [float(r["best"]) for r in g98 if abs(float(r["best"]) + 543.6429) >= 1e-3]
    near75 = sum(abs(float(r["best"]) + 396.2822) < 1e-3 for r in g75)
    audit.cell(PAPER, "panel GMIN LJ98 hits and LJ75 hits", "0 0", f"{hits_of(g98)} {hits_of(g75)}")
    audit.cell(PAPER, "panel GMIN LJ98 stuck runs", "11", float(near98))
    audit.cell(PAPER, "panel GMIN LJ98 other run", "-540.7447", f"{other98[0]:.4f}")
    audit.cell(PAPER, "panel GMIN LJ75 stuck runs", "12", float(near75))
    audit.said(PAPER, "& GMIN & 0/12 & 4000000 & 0 & 119.9")
    audit.said(PAPER, "& GMIN & 0/12 & 4000000 & 0 & 182.5")
    audit.said(PAPER, "Eleven of its 12 LJ\\(_{98}\\) runs end at \\(-543.6429\\) and one at \\(-540.7447\\), and all 12 of its "
                    "LJ\\(_{75}\\) runs end at \\(-396.2822\\).")
    for path in (PAPER, COVER):
        text = audit.text(path)
        audit.record("200 of 200" not in text and "64 of 64" not in text and "earlier campaign" not in text, path,
                     "no earlier GMIN campaign in the text", "none", "none")
    audit.said(PAPER, "The panel uses three GMIN inputs.")
    audit.said(PAPER, "The Lennard-Jones rows run the Wales-group LJ\\(_{38}\\) introduction file at every Lennard-Jones size,")
    audit.said(PAPER, "The Morse rows add GMIN's native \\texttt{MORSE} keyword and \\(\\rho\\) to the same numerical settings.")
    audit.said(PAPER, "The Au\\(_{55}\\) row instead uses the Darby gold parameters, the \\(11.829521\\)~\\AA\\ "
                    "nearest-neighbor container, \\(T=0.1\\)~eV, and the LJ step and taboo distances scaled by "
                    "\\(r_0/2^{1/6}\\).")
    audit.said(PAPER, "# GMIN: LJ/Morse use TEMPERATURE 1.0 and NEWRESTART 40;")
    audit.said(PAPER, "# Au55 Gupta uses TEMPERATURE 0.1, NEWRESTART 40 and scaled RADIUS/STEP/AVOID")
    audit.said(PAPER, "with \\(T=1.0\\), \\texttt{NEWRESTART 40}, and an energy difference of 0.003")
    data = "examples/h2h_gmin_lj.data"
    for fragment in ("TEMPERATURE 1.0", "NEWRESTART 40", "EDIFF 0.003"):
        audit.said(data, fragment)
    audit.record("SYMMETRISE" not in audit.text(data), data, "the panel input has no symmetrization keyword", "absent",
                 "absent" if "SYMMETRISE" not in audit.text(data) else "present")
    audit.said("paper/jctc/data/lj75_gmin/provenance.txt", "SYMMETRISE")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_ALIGN=OFF")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_MPI=OFF")
    audit.said("bench/gmin-panel/gmin-source.txt", "-DWITH_OPENMP=OFF")
    audit.said(PAPER, "GMIN is built without ALIGN, MPI, or OpenMP and linked against the reference BLAS and LAPACK 3.12.0.")

    if panel_rows_held(gmin[("m55r6", "anneal")], anneal=True):
        cap = [int(r["evals_to_hit"]) for r in gmin[("m55r6", "anneal")] if r["hit"] == "1"]
        gm = [int(r["evals_to_hit"]) for r in gmin[("m55r6", "gmin")] if r["hit"] == "1"]
        audit.said(PAPER, f"At M\\(_{{55}}\\) and \\(\\rho=6\\), this setting hits 12 of 12 and the 12 hits come at "
                        f"{min(cap)} to {max(cap)} evaluations.")
        audit.said(PAPER, f"GMIN hits the same cluster in {min(gm)} to {max(gm)} calls.")
        audit.said(PAPER, "M\\(_{55}\\), \\(\\rho=6\\) & \\(1\\times 10^{6}\\) & this work & 12/12")
    ar = [int(r["evals_to_hit"]) for r in ase[("m55r6", "recommended")] if r["hit"] == "1"]
    audit.said(PAPER, f"The ASE panel's {len(ar)} hits at M\\(_{{55}}\\) and \\(\\rho=6\\) come at "
                    f"{min(ar)} to {max(ar)} calls.")
    audit.record("Mackay cap" not in audit.text(PAPER), PAPER, "Mackay-cap disclosure prose is removed",
                 "absent", "absent" if "Mackay cap" not in audit.text(PAPER) else "present")
    audit.record(not (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists(),
                 "src/methods/cluster_hopping/mackay_cap_tests.rs", "Mackay cap test module is removed",
                 "absent", "present" if (ROOT / "src/methods/cluster_hopping/mackay_cap_tests.rs").exists() else "absent")
    audit.said("examples/h2h_morse.rs", "let cfg = Config::recommended(spec.n);")
    audit.said("bench/ase-nonlj/README.md", "Anneal `recommended` uses `Config::recommended_with_scales`")
    audit.cell(PAPER, "the M55 rho=14 revisions", "5 7", " ".join(str(hits_of(g)) for g in (
        ase[("m55r14", "recommended")], gmin[("m55r14", "anneal")])))
    audit.said(PAPER, "Recommended is the preset of this article.")
    audit.record("two-shell icosahedron" not in audit.text(PAPER), PAPER, "cap-disclosure prose is removed",
                 "absent", "present" if "two-shell icosahedron" in audit.text(PAPER) else "absent")
    audit.record("largest cap" not in audit.text(PAPER), PAPER, "cap-size prose is removed",
                 "absent", "present" if "largest cap" in audit.text(PAPER) else "absent")
    audit.said(PAPER, "Derived adds the budget window and the Bayes screen of")


STALE_AU55 = (
    (re.compile(r"0\.014"), "the Au55 p of the earlier GMIN panel"),
    (re.compile(r"6 of the 9|six of (?:our|the) nine|6 of our 9", re.IGNORECASE), "six significant of the nine wins"),
)


def stale_sentences(text: str) -> list[str]:
    """Sentences that state an Au55 hit count or a significance claim outside the GMIN panel rows."""
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", text, flags=re.DOTALL)
    flat, found, start = collapse(prose), [], 0
    for end in [m.end() for m in SENTENCE_END.finditer(flat)] + [len(flat)]:
        sentence, start = flat[start:end].strip(), end
        au = "Au\\(_{55}\\)" in sentence
        if au and "GMIN" in sentence and re.search(r"\b0 of 12\b|\b0/12\b", sentence) and "ASE panel" not in sentence:
            found.append(sentence)
        elif au and "significant" in sentence and not re.search(r"\bnot\b|that are not", sentence):
            found.append(sentence)
        elif any(pattern.search(sentence) for pattern, _why in STALE_AU55):
            found.append(sentence)
    return found


def manuscript_au55(audit: Audit) -> None:
    """The Au55 hits of the GMIN panel and the clusters Fisher's test separates."""
    panel, groups = gmin_panel_groups(audit)
    systems = quoted_systems(panel, groups)

    def p_of(system: str) -> float:
        a, g = groups[(system, "anneal")], groups[(system, "gmin")]
        return fisher(hits_of(a), len(a), hits_of(g), len(g))

    au_a, au_g = groups[("au55", "anneal")], groups[("au55", "gmin")]
    audit.cell(PAPER, "Au55 hits of this setting and GMIN in the GMIN panel", "4 1", f"{hits_of(au_a)} {hits_of(au_g)}")
    audit.cell(PAPER, "Au55 Fisher p of the GMIN panel", "0.32", p_of("au55"))
    audit.cell(PAPER, "Au55 Fisher p is not below 0.05", "1", float(p_of("au55") >= 0.05))
    ours = [s for s in systems if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")]) and p_of(s) < 0.05]
    theirs = [s for s in systems if hits_of(groups[(s, "gmin")]) > hits_of(groups[(s, "anneal")]) and p_of(s) < 0.05]
    audit.cell(PAPER, "clusters significant for this setting", "3", float(len(ours)))
    audit.cell(PAPER, "clusters significant for GMIN", "none", " ".join(theirs) or "none")
    audit.cell(PAPER, "clusters significant overall", "3", float(len(ours) + len(theirs)))
    if "au55" in systems:
        audit.said(PAPER, f"On Au\\(_{{55}}\\) this setting hits {hits_of(au_a)} of {len(au_a)} in the GMIN panel, "
                        f"where GMIN hits {hits_of(au_g)} of {len(au_g)} and Fisher's \\(p\\) is "
                        f"{panel.show_p(p_of('au55'), tex=False)}.")
    for path in (PAPER, SI, COVER):
        stale = stale_sentences(audit.text(path))
        audit.record(not stale, path, "no Au55 count, p, or significance claim of the replaced GMIN panel",
                     "none", " | ".join(stale) or "none")


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> list[str]:
    """The names that Holm's step-down procedure rejects at level alpha, in ascending p."""
    kept = []
    for rank, (name, value) in enumerate(sorted(pvalues.items(), key=lambda item: item[1])):
        if value > alpha / (len(pvalues) - rank):
            break
        kept.append(name)
    return kept


def manuscript_review(audit: Audit) -> None:
    """The numbers the referee pass added: multiplicity, priors, tolerances and the hexamer gate."""
    panel, groups = gmin_panel_groups(audit)

    def name(system: str) -> str:
        return panel.LABELS[system][1].replace(", \\(\\rho=", " at \\(\\rho=")

    systems = quoted_systems(panel, groups)
    p_of = {s: fisher(hits_of(groups[(s, "anneal")]), len(groups[(s, "anneal")]),
                      hits_of(groups[(s, "gmin")]), len(groups[(s, "gmin")])) for s in systems}
    survive = holm(p_of)
    ahead = [s for s in survive if hits_of(groups[(s, "anneal")]) > hits_of(groups[(s, "gmin")])]
    behind = [s for s in survive if s not in ahead]
    audit.cell(PAPER, "GMIN panel clusters below 0.05 under Holm", "2", float(len(survive)))
    audit.said(PAPER, f"Under Holm's step-down correction for {len(systems)} tests, {len(survive)} clusters stay below 0.05:")
    audit.cell(PAPER, "Holm survivors for this setting", "lj75 lj98", " ".join(sorted(ahead, key=systems.index)))
    audit.cell(PAPER, "Holm survivor for GMIN", "none", " ".join(behind) or "none")
    holm_names = []
    if ahead:
        holm_names.append(f"{series(name(s) for s in ahead)} for this setting")
    if behind:
        holm_names.append(f"{series(name(s) for s in behind)} for GMIN")
    if holm_names:
        audit.said(PAPER, " and ".join(holm_names) + ".")
    dropped = [s for s in systems if p_of[s] < 0.05 and s not in survive]
    audit.cell(PAPER, "significant before Holm and not after", "lj38", " ".join(dropped))
    audit.said(PAPER, f"LJ\\(_{{38}}\\), at \\(p={p_of['lj38']:.3f}\\), does not stay below 0.05.")

    def wall(system: str, arm: str) -> float:
        return median([float(r["wall_s"]) for r in groups[(system, arm)]])

    slower = [s for s in systems if wall(s, "anneal") >= wall(s, "gmin")]
    audit.cell(PAPER, "clusters where this setting is slower", "none", " ".join(slower) or "none")
    wall_claim = (f"The median wall of a run is lower for this setting on {len(systems) - len(slower)} of the "
                  f"{len(systems)} clusters")
    if slower:
        wall_claim += f" and higher on {series(name(s) for s in slower)}"
    audit.said(PAPER, wall_claim + ".")
    if "m80r6" in slower:
        audit.said(PAPER, f"\\(4\\times 10^{{6}}\\) calls in a median of {panel.show_s(wall('m80r6', 'anneal'))}~s.")
    audit.record("168.9" not in audit.text(PAPER) and "168.9" not in audit.text("paper/jctc/cover-letter.tex"), PAPER,
                 "no template-era M80 wall", "none", "none")

    ase = ase_groups(audit)

    def ase_p(system: str, arm: str) -> float:
        a, b = ase[(system, "recommended")], ase[(system, arm)]
        return fisher(hits_of(a), len(a), hits_of(b), len(b))

    lj = [s for s in ASE_PANEL if s.startswith("lj")]
    for family, members, dropped_want in (("Lennard-Jones", lj, "lj150"), ("Morse", list(MORSE_PANEL), "m55r14")):
        tests = {f"{s}/{arm}": ase_p(s, arm) for s in members for arm in ASE_ARMS}
        kept = holm(tests)
        dropped = sorted({key.split("/")[0] for key in tests if tests[key] < 0.05 and key not in kept})
        audit.cell(PAPER, f"ASE {family} tests", str(len(tests)), float(len(tests)))
        audit.cell(PAPER, f"ASE {family} clusters below 0.05 before Holm and not after", dropped_want, " ".join(dropped))
    if load_script("ase_panel_holm", "scripts/ase_panel/summarize.py").tex_table():
        audit.said(PAPER, "Under Holm's correction for those 10 tests, LJ\\(_{150}\\), at \\(p=0.037\\) against each arm, "
                        "does not stay below 0.05.")
    audit.said(PAPER, "Under Holm's correction for those 8 tests, M\\(_{55}\\) at \\(\\rho=14\\) does not stay below 0.05.")
    runner = audit.text("scripts/ase_panel/run_one.py")
    audit.record("BasinHopping(\n                atoms,\n                logfile=None," in runner and "temperature=" not in runner
                 and "dr=" not in runner and "MinimaHopping(atoms)(totalsteps" in runner, "scripts/ase_panel/run_one.py",
                 "ASE basin hopping and minima hopping keep their temperature, step, and T0 defaults", "defaults", "defaults")
    k_b = BOLTZMANN_EV_PER_K
    audit.cell(PAPER, "ASE BH default kT in reduced units", "0.0086", 100.0 * k_b)
    audit.cell(PAPER, "ASE MH default T0 in reduced units", "0.086", 1000.0 * k_b)
    audit.said(PAPER, "ASE basin hopping therefore runs at \\(k_BT=0.0086\\) in reduced units and shifts every coordinate "
                    "by at most 0.1")


APPEND_ONLY_BASES = (
    "HEAD",
    "origin/develop",
    "origin/main",
    "origin/docs/paper-ms4",
    "origin/bench/crest-equal",
)

APPEND_ONLY_RECORDS = {
    "bench/h2h.json",
    "bench/h2h-additions.jsonl",
    "bench/tip4p-h2h/rows-h2h.csv",
    "bench/tip4p-h2h/rows-h2h-r2.csv",
    "bench/tip4p-h2h/hosts-r2.csv",
    "bench/tip4p-h2h/sessions.jsonl",
    "bench/tip4p-h2h/sessions-r2.jsonl",
    "bench/tip4p-h2h/wall-r2.csv",
    "bench/tip4p-h2h/check-r2.md",
    "bench/unreachable-revisions.csv",
    "bench/unreachable-revisions-updates.csv",
    "bench/unrecorded-revisions.csv",
    "paper/jctc/FIGURE-SOURCE.tsv",
    "paper/jctc/FIGURE-SOURCE-updates.tsv",
    "results/size-tiered-selection.csv",
    "results/size-tiered-selection-provenance.csv",
    "bench/crest-equal/call-counts.csv",
    "bench/crest-equal/rgpot-revisions.csv",
    "bench/covalent-h2h/gfn2/rgpot-revisions.csv",
    "bench/generic-alloy/g3-relabel.csv",
    "data/hf-files.csv",
    "data/hf-files-updates.csv",
    "data/hf-tree.csv",
}


def touched_record_paths(base: str) -> set[str]:
    """Every csv and jsonl path named from base to HEAD, including one absent at HEAD."""
    log = git("log", "--name-only", "--pretty=format:", "--no-renames", f"{base}..HEAD")
    return {line for line in log.stdout.splitlines() if line.endswith((".csv", ".jsonl"))}


def append_only_rows(audit: Audit) -> None:
    """Every record preserves baseline lines and the series never removes one."""
    paths = set(git("ls-files", "**/rows.csv", "rows.csv").stdout.splitlines())
    paths.update(path for path in APPEND_ONLY_RECORDS if (ROOT / path).is_file())
    existing_refs = [
        ref for ref in APPEND_ONLY_BASES
        if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0
    ]
    base_ref = next(
        (ref for ref in ("origin/develop", "develop")
         if git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0),
        None,
    )
    if base_ref is not None:
        paths.update(touched_record_paths(base_ref))
    for path in sorted(paths):
        if not (ROOT / path).is_file():
            continue
        cur_lines = audit.text(path).splitlines(keepends=True)
        rewritten = []
        refs_for_path = (
            existing_refs
            if path == "rows.csv" or path.endswith("/rows.csv")
            else [ref for ref in existing_refs if ref in ("HEAD", "origin/develop")]
        )
        for ref in refs_for_path:
            shown = git("show", f"{ref}:{path}")
            if shown.returncode != 0:
                continue
            old_lines = shown.stdout.splitlines(keepends=True)
            if len(cur_lines) < len(old_lines):
                rewritten.append(f"{ref}: truncated ({len(cur_lines)} < {len(old_lines)} lines)")
                continue
            for idx, (old_line, cur_line) in enumerate(zip(old_lines, cur_lines), start=1):
                if old_line != cur_line:
                    rewritten.append(f"{ref}: line {idx} rewritten")
                    break
        audit.record(
            not rewritten,
            path,
            "committed rows.csv lines are append-only",
            "append-only",
            "; ".join(rewritten) or "append-only",
        )
    base = base_ref
    if base is not None:
        history = git(
            "log", "-p", "--no-renames", "--format=", f"{base}..HEAD", "--",
            *sorted(paths),
        ).stdout.splitlines()
        removed = [line for line in history if line.startswith("-") and not line.startswith("---")]
        audit.record(
            not removed,
            "record-file history",
            f"{base}..HEAD removes no committed record lines",
            "no removed lines",
            f"{len(removed)} removed lines" if removed else "no removed lines",
        )


def tip4p_unscored(audit: Audit) -> None:
    """Audit the independent TIP4P tolerances and the unscored size 25."""
    wall_path = "bench/tip4p-h2h/wall-r2.csv"
    wall = next(
        row for row in csv.DictReader(audit.text(wall_path).splitlines())
        if row["molecules"] == "25"
    )
    audit.record(
        wall["anneal_hits_n"] == wall["gmin_hits_n"] == "unscored",
        wall_path,
        "TIP4P size 25 is unscored for both arms",
        "unscored unscored",
        f"{wall['anneal_hits_n']} {wall['gmin_hits_n']}",
    )
    check_row = re.search(
        r"^\|\s*25\s*\|.*$", audit.text("bench/tip4p-h2h/check-r2.md"),
        re.MULTILINE,
    )
    audit.record(
        check_row is not None
        and check_row.group(0).count("unscored") == 2
        and re.search(r"\d+/\d+", check_row.group(0)) is None,
        "bench/tip4p-h2h/check-r2.md",
        "the size-25 record is marked unscored rather than counted",
        "two unscored cells",
        check_row.group(0) if check_row else "missing",
    )
    stats = (
        wall["anneal_energy_best_kj_mol"],
        wall["anneal_energy_median_kj_mol"],
        wall["anneal_wall_median_s"],
        wall["gmin_energy_best_kj_mol"],
        wall["gmin_energy_median_kj_mol"],
        wall["gmin_wall_median_s"],
    )
    phrase = "Size 25 has no independent reference and is unscored."
    manuscript_text = collapse(audit.text(TEX))
    manuscript_tip = manuscript_text.split(
        "Against GMIN on TIP4P", 1
    )[1].split(
        "The comparisons with the Atomic Simulation Environment", 1
    )[0]
    manuscript_conclusion = manuscript_text.split(
        "In the 120-run ABBA comparison", 1
    )[1].split("Author contributions", 1)[0]
    readers = {
        TEX: manuscript_tip + " " + manuscript_conclusion,
        SI: collapse(audit.text(SI).split("% tip4p-gmin prose: start", 1)[1].split(
            "% tip4p-gmin prose: end", 1
        )[0]),
        "bench/tip4p-h2h/README-h2h.md": collapse(
            audit.text("bench/tip4p-h2h/README-h2h.md")
        ),
    }
    for path, text in readers.items():
        size_sentences = [
            sentence for sentence in re.split(r"(?<=[.!?])\s+", text)
            if re.search(r"\b25\b", sentence)
        ]
        forbidden = [
            sentence for sentence in size_sentences
            if re.search(
                r"\d+/\d+|--|\b(?:calls?|costs?)\b|Mann--Whitney|p(?:=|_)",
                sentence,
                re.IGNORECASE,
            )
        ]
        audit.record(
            phrase in text
            and all(value in text.replace("−", "-") for value in stats)
            and not forbidden,
            path,
            "TIP4P size 25 has only its reason, best, median, and median wall",
            "reason and six statistics",
            " | ".join(forbidden) or "reason and six statistics",
        )
    readme = audit.text("bench/tip4p-h2h/README-h2h.md")
    si_table = audit.text(SI).split("% tip4p-gmin table: start", 1)[1].split(
        "% tip4p-gmin table: end", 1
    )[0]
    audit.record(
        "(H₂O)₂₅ |" not in readme and "(H\\(_2\\)O)\\(_{25}\\)" not in si_table,
        PAPER,
        "TIP4P size 25 is outside every hits table",
        "absent",
        "present" if "(H₂O)₂₅ |" in readme or "_{25}" in si_table else "absent",
    )
    scoped_claims = {
        TEX: manuscript_conclusion,
        SI: readers[SI],
    }
    for path, text in scoped_claims.items():
        audit.record(
            "at every size" not in text
            and "21 and 25 molecules" not in text
            and "at 25 molecules" not in text,
            path,
            "size 25 is outside wall-p, exhausted-call, and per-call-cost claims",
            "absent",
            "present" if any(fragment in text for fragment in (
                "at every size", "21 and 25 molecules", "at 25 molecules"
            )) else "absent",
        )
    references = {
        "8": "-305.5183221676",
        "15": "-628.3722525077",
        "20": "-872.9887544952",
        "21": "-916.7074302940",
    }
    reference_text = audit.text(TEX).split(
        "The scored references are", 1
    )[1].split("These geometries", 1)[0]
    shown = dict(re.findall(
        r"\{?(\d+)\}?\s*&:\s*(-\d+\.\d+)", reference_text
    ))
    audit.record(
        shown == references,
        TEX,
        "the TIP4P scored reference list is exactly 8, 15, 20, and 21",
        json.dumps(references, sort_keys=True),
        json.dumps(shown, sort_keys=True),
    )
    audit.said(
        TEX,
        "\\cite{walesCambridgeClusterDatabase,walesGlobalMinimaWater1998,jorgensen1983}",
    )
    rates_tolerance = 0.001 * KJ_PER_KCAL
    shown_kj = format(rates_tolerance, ".6f")
    audit.said(
        SI,
        "The hit tolerance for these water-rate rows is 0.001~kcal/mol\n"
        f"({shown_kj}~kJ/mol) above the reference.",
    )
    audit.said(
        SI,
        "The hit tolerance for the scored sizes is 0.001~kJ/mol.",
    )
    rates_readme = "bench/tip4p-water-rates/README.md"
    audit.said(
        rates_readme,
        f"within 0.001 kcal/mol ({shown_kj} kJ/mol) of the Wales-Hodges\nreference",
    )
    for path in (
        "bench/tip4p-water-rates/rates-a24ba760.csv",
        "bench/tip4p-water-rates/rates.csv",
    ):
        rows = list(csv.DictReader(audit.text(path).splitlines()))
        mismatches = [
            row for row in rows
            if int(row["hit"]) != int(
                float(row["best_kj"])
                <= float(row["reference_kj"]) + rates_tolerance
            )
        ]
        audit.record(
            not mismatches,
            path,
            f"every water-rate hit flag uses the printed {shown_kj} kJ/mol tolerance",
            "all flags reproduce",
            f"{len(mismatches)} mismatches" if mismatches else "all flags reproduce",
        )
    reproduce = audit.text("paper/jctc/REPRODUCE.md")
    audit.record(
        "--rows-out /work/tip4p-h2h-rerun/rows-r2.csv" in reproduce
        and "--rows /work/tip4p-h2h-rerun/rows-r2.csv" in reproduce
        and "--reference-rows bench/tip4p-h2h/rows-h2h-r2.csv" in reproduce,
        "paper/jctc/REPRODUCE.md",
        "the TIP4P rerun writes fresh records and compares them with committed rows",
        "fresh output and committed comparison",
        "present" if "/work/tip4p-h2h-rerun/rows-r2.csv" in reproduce else "missing",
    )


def bibliography_keys(audit: Audit) -> None:
    """Require every citation in the JCTC TeX sources to resolve in refs.bib."""
    cited: set[str] = set()
    for source in sorted((ROOT / "paper/jctc").glob("*.tex")):
        path = source.relative_to(ROOT).as_posix()
        text = re.sub(r"(?m)(?<!\\)%.*$", "", audit.text(path))
        for group in re.findall(
            r"\\(?:[Cc]ite(?:alp|alt|author|num|p|t|yearpar|year)?|nocite)\*?\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}",
            text,
            re.DOTALL,
        ):
            cited.update(
                key.strip()
                for key in group.split(",")
                if key.strip() and key.strip() != "*"
            )

    bibliography = audit.text("paper/jctc/refs.bib")
    available = set(re.findall(
        r"(?mi)^@\w+\s*\{\s*([^,\s]+)\s*,",
        bibliography,
    ))
    missing = sorted(cited - available)
    audit.record(
        not missing,
        "paper/jctc/refs.bib",
        "every \\cite key resolves in the paper bibliography",
        f"{len(cited)} cited keys resolve",
        ", ".join(missing) if missing else f"{len(cited)} cited keys resolve",
    )


HISTORY_PHRASES = (
    ("previously", r"previously"),
    ("no longer", r"no longer"),
    ("used to", r"(?<!is )(?<!are )(?<!was )(?<!were )used to"),
    ("used to stop", r"used to stop"),
    ("earlier draft", r"earlier draft"),
    ("earlier version", r"earlier version"),
    ("v1.N release tag", r"v1\.\d+"),
    ("stay open for", r"stay open for"),
    ("stays open", r"stays open"),
    ("withheld", r"withheld"),
    ("removed from the article", r"removed from the article"),
    ("unrecorded", r"unrecorded"),
    ("unrecoverable", r"unrecoverable"),
    ("cannot be recovered", r"cannot be recovered"),
    ("not captured", r"not captured"),
    ("Recorded limitations", r"recorded limitations"),
    ("Left for Papers", r"left for papers"),
    ("the parent of", r"the parent of"),
)


def history_narration(audit: Audit) -> None:
    """Reject repository diary language in benchmark READMEs and paper Markdown."""
    sources = {
        *ROOT.glob("bench/**/README*.md"),
        *(ROOT / "paper/jctc").glob("*.md"),
        *(ROOT / "paper/jctc").glob("*.tex"),
        ROOT / "paper/related-work.md",
    }
    for source in sorted(sources):
        path = source.relative_to(ROOT).as_posix()
        text = audit.text(path)
        found = [
            label for label, pattern in HISTORY_PHRASES
            if re.search(rf"\b(?:{pattern})\b", text, re.IGNORECASE)
        ]
        audit.record(
            not found,
            path,
            "README and paper Markdown state the present and carry no history narration",
            "absent",
            ", ".join(found) or "absent",
        )


CONVERSION_LITERAL = re.compile(
    r"(?<![\d.])(?:27\.211_?[0-9]|0\.529_?1[0-9_]*|1\.889_?7[0-9_]*|4\.184(?![\d]))"
)


def conversion_literals(audit: Audit) -> None:
    """Hartree, Bohr, and kilocalorie conversions come from a library, never a typed literal."""
    listed = git("ls-files", "src", "examples", "tests", "scripts").stdout.splitlines()
    listed.extend((
        "bench/water-h2o16-quench/README.md",
        "bench/covalent-h2h/gfn2/scc-false-minimum/README.md",
    ))
    for path in sorted(name for name in listed if name.endswith((".rs", ".py", ".md"))):
        found = sorted({match.group(0) for match in CONVERSION_LITERAL.finditer(audit.text(path))})
        audit.record(not found, path, "no typed Hartree, Bohr, or kilocalorie conversion literal",
                     "none", " ".join(found) or "none")


def prose_receipt_lines(audit: Audit) -> None:
    """Every file:line record of the prose receipt holds its string at that line."""
    path = "paper/jctc/PROSE-RECEIPT.md"
    block = audit.text(path).split("```\n", 1)[1].split("```", 1)[0]
    for entry in block.strip().split("\n\n"):
        reference, string = entry.split("\n", 1)
        target, line = reference.rsplit(":", 1)
        lines = audit.text(target).splitlines() if (ROOT / target).is_file() else []
        at = lines[int(line) - 1] if 0 < int(line) <= len(lines) else ""
        audit.record(string in at, path, f"{reference} holds its recorded string",
                     string, at.strip() or "missing")


_CLAIMED_COMMIT = re.compile(
    r"(?:paper at |scripts of |did at |\\CommitShort\{|OLDTIP\s*=\s*\")"
    r"([0-9a-f]{7,40})"
)
_MACRO_COMMIT = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([0-9a-f]{40})\}")


def script_commit_hashes(audit: Audit) -> None:
    """A commit a paper script or the macros names is a commit this repository holds.

    A mark does not apply here: the hash is the claim. The rgpot pin is a
    dependency revision, not a commit of this repository.
    """
    paths = sorted((ROOT / "paper/jctc/scripts").glob("*.py"))
    paths = [path for path in paths if not path.name.startswith("test_")]
    paths.append(ROOT / "paper/jctc/macros.tex")
    failed = False
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        text = audit.text(rel)
        claimed = [match.group(1) for match in _CLAIMED_COMMIT.finditer(text)]
        if path.name == "macros.tex":
            claimed.extend(
                revision for name, revision in _MACRO_COMMIT.findall(text)
                if name != "RgpotRevision"
            )
        for revision in claimed:
            if commit_held(revision) or revision == _UPSTREAM_GMIN or _UPSTREAM_GMIN.startswith(revision):
                continue
            failed = True
            audit.record(False, rel, HELD_NUMBER, "absent", revision[:12])
    if not failed:
        audit.record(True, "paper/jctc/scripts", HELD_NUMBER, "absent", "absent")


def figure_row_hashes(audit: Audit) -> None:
    """Each figure row of FIGURES.md names the SHA-256 of the file it describes."""
    path = "paper/jctc/FIGURES.md"
    rows = re.findall(r"^Row (\d+) is (\S+)\.\n\n([0-9a-f]{64})$", audit.text(path), re.MULTILINE)
    audit.record(bool(rows), path, "figure rows carry hashes", "rows", len(rows))
    for number, name, digest in rows:
        figure = ROOT / "paper/jctc/figs" / name
        actual = hashlib.sha256(figure.read_bytes()).hexdigest() if figure.is_file() else "missing"
        audit.record(actual == digest, path, f"row {number} {name} sha256", digest, actual)


def untraced_check(audit: Audit) -> None:
    """A printed number that no committed row backs is quoted as written, and it fails the audit until a row holds it."""
    for path, fragment, why in UNTRACED:
        audit.said(path, fragment)
        audit.record(False, path, f"printed number that no committed row backs: {why}", "a committed row", fragment)


def run(report: bool = False, write_map: bool = False, pre_submission: bool = False) -> int:
    problems = unauditable()
    if problems:
        for problem in problems:
            print(f"paper_audit: cannot check revisions: {problem}", file=sys.stderr)
        return 2
    audit = Audit()
    for check in (manuscript_ablation, selection_tables, manuscript_retry, manuscript_gmin, manuscript_tip4p, manuscript_hexamer,
                  manuscript_cross_field, supporting_information, release, paper_0120, gad_published, openconf,
                  head_to_head, sc96_78_howto, stale_claims, tip4p_panel, reproduction_records, covalent_panels, covalent_gfn2,
                  ase_gofee, lj_large, bench_prose,
                  manuscript_ase, panel_references, generated_tables, sutton_chen, supporting_numbers, method_definitions, prior_switches, jctc_requirements, crest_equal, crest_equal_bench, au55_superseded, manuscript_new, manuscript_followups, manuscript_au55, cover_letter, manuscript_review, append_only_rows, tip4p_unscored, bibliography_keys, history_narration, conversion_literals, prose_receipt_lines, script_commit_hashes, figure_row_hashes):
        check(audit)
    if write_map:
        (ROOT / REVISION_MAP).write_text(revision_map_text(revision_map_rows(audit)))
    revision_map(audit)
    rerun_check(audit, pre_submission)
    untraced_check(audit)
    gaps = provenance(audit)
    cited = {r[c].strip() for rows in audit.read.values() for r in rows for c in REVISION_COLUMNS if c in r}
    for name, why in sorted(unpushed_reasons().items()):
        if name not in cited:
            audit.record(False, UNPUSHED_FILE, f"unpushed revision {name[:8]}", why, "no row cites it")
    for path, why in sorted(KNOWN_GAPS.items()):
        audit.record(any(word in why for word in ("unrecorded", "revision recorded", "is not in this repository", "is not on a pushed branch")), path,
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
