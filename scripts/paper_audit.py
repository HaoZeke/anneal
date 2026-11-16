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
``KNOWN_GAPS``. Printed numbers no committed row backs are listed in
``UNTRACED``; their text must still be where the list says.

    python3 scripts/paper_audit.py            # exit 1 on any drift
    python3 scripts/paper_audit.py --report   # also print every check
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = "paper/jctc/manuscript.tex"
SI = "paper/jctc/si.tex"

REVISION_COLUMNS = ("code_revision", "anneal_head", "release_revision", "head", "harness_revision")
HOST_COLUMNS = ("host",)
# Revisions of other codes. The GMIN build scripts refuse a tarball whose SVNREV differs.
EXTERNAL_REVISIONS = {"aac1548a8aec13b53ad12a8d6904625081685d2e": "GMIN source/SVNREV"}

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
    "bench/gad-published/rows.csv": "revision 48ecd4c0 is not in this repository",
    "bench/gad-published/summary.csv": "no host column; revision 238f3582 is not in this repository",
    "bench/morse-panel/seeds.csv": "no revision or host column",
    "bench/morse-h2h/seeds.csv": "no revision or host column",
    "bench/morse-h2h/results.csv": "revision fe9671c8 is not in this repository",
    "bench/fp-fingerprint/rows.csv": "no revision or host column",
    "bench/tip4p-h2h/h2h-summary.csv": "no revision or host column",
    "bench/covalent-h2h/before/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/after/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/v2/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/v3/seeds.csv": "no revision or host column",
    "bench/covalent-h2h/gfn2/seeds.csv": "no revision or host column",
    "bench/ase-gofee-h2h/results.csv": "no host column; head empty or unknown on some rows; revision 0f988569 is not in "
                                       "this repository; revision 349523d3 is not in this repository; revision befbc3e8 "
                                       "is not in this repository; revision bf09f7a2 is not in this repository; revision "
                                       "e09e0487 is not in this repository",
    "bench/growpal-h2h/seeds.csv": "no revision or host column",
    "bench/metal-cluster-h2h/ptcu-b3d03061/icosa-23.csv": "no revision or host column",
    "bench/nh3-octamer/ladder.csv": "no revision or host column",
    "bench/generic-alloy/g3-priors-off-rows.csv": "revision 95c27427 is not in this repository",
    "bench/generic-alloy/g3-priors-on-rows.csv": "revision 95c27427 is not in this repository",
    "bench/regress-bb486bfd/rows.csv": "revision bdf08495 is not in this repository",
    "bench/openconf-h2h/paired/seeds.csv": "revision d178bd15 is not in this repository",
    "bench/openconf-h2h/conf-fix/seeds.csv": "revision b09285bb is not in this repository",
    "bench/openconf-h2h/conf-fix3/seeds.csv": "revision cf660e19 is not in this repository",
    "bench/openconf-h2h/conf-fix4/seeds.csv": "revision 2a5b4553 is not in this repository",
    "bench/openconf-h2h/conf-fix4/n24/seeds.csv": "revision 26b70c01 is not in this repository",
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


def summary_against_rows(audit: Audit, summary: str, per_seed: str, arm_key: str, calls_key: str,
                         continuity: bool, mw_column: str = "mannwhitney_p_evals") -> None:
    """A summary file derived from per-seed rows must match them."""
    rows = audit.rows(per_seed)
    for line in audit.rows(summary):
        harness = {"harness_revision": line["harness_revision"]} if "harness_revision" in line else {}
        group = pick(rows, system=line["system"], **{arm_key: line["arm"]}, **harness)
        where = f"{summary} {line['system']} {line['arm']}" + (f" {line['harness_revision'][:8]}" if harness else "")
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
    for line, row in md_table(audit, path, "System | Arm | Harness"):
        group = [r for r in pick(sc_rows, system=row["System"], arm=row["Arm"])
                 if r["harness_revision"].startswith(row["Harness"])]
        where = f"{path}:{line} {row['System']} {row['Arm']} {row['Harness']}"
        audit.cell(where, "seeds", "12", float(len(group)))
        audit.cell(where, "hits", row["Hits"], frac(group))
        audit.cell(where, "calls median", row["Calls median"], median([float(r["evals_to_hit"]) for r in group]))
        audit.cell(where, "wall median", row["Wall median (s)"], median([float(r["wall_s"]) for r in group]))
    rerun = [r for r in sc_summary if r["system"] == "sc96-38" and r["arm"] == "gmin"][-1]
    audit.said(path, f"Only 9-6 at N=38 has been rerun, under harness {rerun['harness_revision'][:8]}.")
    audit.cell(path, "rerun Fisher p", "1", float(rerun["fisher_p_vs_gmin"]))
    audit.cell(path, "rerun Mann-Whitney p calls", "0.000592", float(rerun["mannwhitney_p_evals"]))
    audit.cell(path, "rerun Mann-Whitney p wall", "0.624", float(rerun["mannwhitney_p_wall"]))
    audit.cell(path, "rerun wall ratio", "1.148", float(rerun["wall_ratio"]))

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


def covalent_gfn2(audit: Audit) -> None:
    path = "bench/covalent-h2h/README.md"
    rows = audit.rows("bench/covalent-h2h/gfn2/seeds.csv")
    systems = {"Si10": "si10", "C10": "c10"}
    codes = {"anneal G": "anneal-general", "CREST": "crest"}
    groups = {}
    for line, row in md_table(audit, path, "system | code | hits | median calls | IQR | best / eV"):
        group = pick(rows, system=systems[row["system"]], code=codes[row["code"]])
        groups[(row["system"], row["code"])] = group
        where = f"{path}:{line} {row['system']} {row['code']}"
        calls = [float(r["evals_to_hit"]) for r in group]
        charged = sum(float(r["charged"]) for r in group)
        audit.cell(where, "hits", row["hits"], frac(group))
        audit.cell(where, "median calls", row["median calls"], median(calls))
        audit.cell(where, "IQR", row["IQR"], iqr(calls))
        audit.cell(where, "best", row["best / eV"], min(float(r["best"]) for r in group))
        audit.cell(where, "median wall", row["median wall / s"], median([float(r["wall_s"]) for r in group]))
        audit.cell(where, "inside / call ms", row["inside / call"].split()[0], sum(float(r["pot_s"]) for r in group) / charged * 1e3)
        audit.cell(where, "outside / call ms", row["outside / call"].split()[0], sum(float(r["outside_s"]) for r in group) / charged * 1e3)
    for system, shown in (("Si10", "0.0044"), ("C10", "0.0005")):
        a, c = groups[(system, "anneal G")], groups[(system, "CREST")]
        p = mann_whitney([float(r["evals_to_hit"]) for r in a], [float(r["evals_to_hit"]) for r in c], continuity=True)
        audit.cell(path, f"GFN2 {system} Mann-Whitney p", shown, p)
    a, c = groups[("C10", "anneal G")], groups[("C10", "CREST")]
    audit.cell(path, "GFN2 C10 Fisher p", "0.0094", fisher(hits_of(a), len(a), hits_of(c), len(c)))


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
    out = subprocess.run(["git", "cat-file", "--batch-check"], input="".join(f"{n}^{{commit}}\n" for n in names),
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    for name, line in zip(names, out):
        if line.endswith("missing"):
            for path in revisions[name]:
                note = f"revision {name[:8]} is not in this repository"
                gaps[path] = f"{gaps[path]}; {note}" if path in gaps else note
    return gaps


def run(report: bool = False) -> int:
    audit = Audit()
    for check in (manuscript_ablation, selection_tables, manuscript_retry, manuscript_gmin, manuscript_tip4p, manuscript_hexamer,
                  manuscript_cross_field, supporting_information, release, paper_0120, gad_published, openconf,
                  head_to_head, sc96_78_howto, tip4p_panel, covalent_panels, covalent_gfn2, ase_gofee, bench_prose):
        check(audit)
    for path, fragment, _why in UNTRACED:
        audit.said(path, fragment)
    gaps = provenance(audit)
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
