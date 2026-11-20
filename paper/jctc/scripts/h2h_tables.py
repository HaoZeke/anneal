#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the head-to-head tables of the manuscript and the SI from the committed summaries.

Three blocks, each between a pair of marker comments:

    tab:gofee       manuscript.tex  bench/ase-gofee-h2h/summary.csv
    tab:tip4p-gmin  manuscript.tex  bench/tip4p-h2h/rows-h2h-r2.csv
    tab:gofee-si    si.tex          bench/ase-gofee-h2h/asg2-summary.csv, arms measured there

    python3 paper/jctc/scripts/h2h_tables.py            # rewrite the blocks
    python3 paper/jctc/scripts/h2h_tables.py --check    # exit 1 if one is stale
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"
GOFEE = ROOT / "bench" / "ase-gofee-h2h"
TIP4P_ROWS = ROOT / "bench" / "tip4p-h2h" / "rows-h2h-r2.csv"
TIP4P_UNSCORED = 25
TIP4P_M = 4

MARKS = {
    "gofee": ("% gofee table: start", "% gofee table: end"),
    "tip4p": ("% tip4p-gmin table: start", "% tip4p-gmin table: end"),
    "gofee-si": ("% gofee-si table: start", "% gofee-si table: end"),
}
ARM_LABEL = {"paper": "This work", "ase_bh": "ASE BH", "ase_mh": "ASE MH", "agox_gofee": "GOFEE"}
LJ_LABEL = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj75": "LJ\\(_{75}\\)"}
OTHER_LABEL = {"au55": "Au\\(_{55}\\)", "cu38": "Cu\\(_{38}\\)", "cu12au26": "Cu\\(_{12}\\)Au\\(_{26}\\)",
               "sw10": "Si\\(_{10}\\)", "tc10": "C\\(_{10}\\)", "tip4p8": "(H\\(_2\\)O)\\(_8\\)"}
TIP4P_LABEL = {"tip4p8": "(H\\(_2\\)O)\\(_8\\)", "tip4p15": "(H\\(_2\\)O)\\(_{15}\\)",
               "tip4p20": "(H\\(_2\\)O)\\(_{20}\\)", "tip4p21": "(H\\(_2\\)O)\\(_{21}\\)",
               "tip4p25": "(H\\(_2\\)O)\\(_{25}\\)"}


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * p
    lo = math.floor(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def fisher_exact(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, the same sum as scripts/paper_audit.py."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney_exact(x: list[float], y: list[float]) -> float:
    """Two-sided exact Mann-Whitney p by enumerating the rank sum."""
    n1, n2 = len(x), len(y)
    pooled = list(x) + list(y)
    order = sorted(enumerate(pooled), key=lambda item: item[1])
    ranks = [0.0] * len(pooled)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = rank
        i = j + 1
    ranks2 = [int(round(rank * 2.0)) for rank in ranks]
    obs_u2 = sum(ranks2[:n1]) - n1 * (n1 + 1)
    dp: list[dict[int, int]] = [{0: 1}] + [{} for _ in range(n1)]
    seen = 0
    for rank2 in ranks2:
        seen += 1
        top = min(seen, n1)
        for picked in range(top, 0, -1):
            current = dp[picked]
            previous = dp[picked - 1]
            for score, ways in previous.items():
                current[score + rank2] = current.get(score + rank2, 0) + ways
    total = math.comb(n1 + n2, n1)
    low = sum(ways for score, ways in dp[n1].items() if score - n1 * (n1 + 1) <= obs_u2) / total
    high = sum(ways for score, ways in dp[n1].items() if score - n1 * (n1 + 1) >= obs_u2) / total
    return min(1.0, 2.0 * min(low, high))


def bonferroni(p: float, m: int = TIP4P_M) -> float:
    return min(1.0, m * p)


def tip4p_stats() -> list[dict]:
    """The twelve-seed TIP4P head-to-head, one record per size, from the per-seed rows.

    Sizes with a Cambridge header are scored. (H2O)25 is not. Fisher and the
    calls-to-hit Mann-Whitney p are Bonferroni-corrected with m=4. A miss is
    ranked at its recorded call count, which is the budget. Wall is the median
    total wall time per run. A run that hits stops early, so it also depends on
    the calls the search spent. Per call is the median over runs of wall_s over
    calls, which leaves only the cost of a call.
    """
    rows = read(TIP4P_ROWS)
    grouped: dict[int, dict[str, list[dict[str, str]]]] = {}
    order: list[int] = []
    for row in rows:
        size = int(row["molecules"])
        if size not in grouped:
            order.append(size)
            grouped[size] = {}
        grouped[size].setdefault(row["arm"], []).append(row)
    stats = []
    for size in order:
        anneal = grouped[size]["anneal"]
        gmin = grouped[size]["gmin"]
        anneal_s = sorted(anneal, key=lambda row: int(row["seed"]))
        gmin_s = sorted(gmin, key=lambda row: int(row["seed"]))
        ah = sum(int(row["hit"]) for row in anneal)
        gh = sum(int(row["hit"]) for row in gmin)
        scored = size != TIP4P_UNSCORED
        fisher_raw = fisher_exact(ah, len(anneal), gh, len(gmin)) if scored else None
        calls_raw = mann_whitney_exact(
            [float(row["calls"]) for row in anneal],
            [float(row["calls"]) for row in gmin],
        ) if scored else None
        energies_a = [float(row["best_energy"]) for row in anneal]
        energies_g = [float(row["best_energy"]) for row in gmin]
        walls_a = [float(row["wall_s"]) for row in anneal]
        walls_g = [float(row["wall_s"]) for row in gmin]
        stats.append({
            "molecules": size,
            "scored": scored,
            "n": len(anneal),
            "budget": max(int(row["calls"]) for row in anneal + gmin),
            "seeds": tuple(int(row["seed"]) for row in anneal_s),
            "same_start": all(a["start_id"] == g["start_id"] for a, g in zip(anneal_s, gmin_s)),
            "anneal_hits": ah,
            "gmin_hits": gh,
            "fisher_raw": fisher_raw,
            "calls_raw": calls_raw,
            "fisher_p": None if fisher_raw is None else bonferroni(fisher_raw),
            "calls_p": None if calls_raw is None else bonferroni(calls_raw),
            "anneal_best": min(energies_a),
            "anneal_median": type7(energies_a),
            "gmin_best": min(energies_g),
            "gmin_median": type7(energies_g),
            "anneal_wall": type7(walls_a),
            "gmin_wall": type7(walls_g),
            "anneal_per_call": type7([float(row["wall_s"]) / int(row["calls"]) for row in anneal]),
            "gmin_per_call": type7([float(row["wall_s"]) / int(row["calls"]) for row in gmin]),
        })
    return stats


def show_s(value: float) -> str:
    if value < 100.0:
        return f"{value:.3f}"
    return f"{value:.1f}"


def show_us(seconds: float) -> str:
    return f"{seconds * 1e6:.2f}"


def show_calls(value: str) -> str:
    return f"{float(value):.0f}" if value else "--"


def show_p(value: float) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if value >= 1.0:
        return "1"
    if value >= 0.01:
        return f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def budget_tex(budget: str) -> str:
    mantissa, exponent = f"{int(budget):.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def gofee_block() -> str:
    rows = read(GOFEE / "summary.csv")
    body = []
    for system in LJ_LABEL:
        for arm in ARM_LABEL:
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                LJ_LABEL[system] if arm == "paper" else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if arm == "paper" else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
        if system != "lj75":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against ASE basin hopping (BH), ASE minima hopping (MH)",
        "and AGOX GOFEE on the same three Lennard-Jones clusters.",
        "Seeds 2000--2005, one core, budget in force calls.",
        "Capped counts the runs stopped by a wall cap of 2 hours, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.~\\rerun{5}}",
        "\\label{tab:gofee}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "Cluster & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def tip4p_block() -> str:
    body = []
    for row in tip4p_stats():
        if row["scored"]:
            hits_a = f"{row['anneal_hits']}/{row['n']}"
            hits_g = f"{row['gmin_hits']}/{row['n']}"
            fisher_p = show_p(row["fisher_p"])
            calls_p = show_p(row["calls_p"])
        else:
            hits_a = hits_g = fisher_p = calls_p = "--"
        body.append(" & ".join([
            TIP4P_LABEL[f"tip4p{row['molecules']}"], budget_tex(str(row["budget"])),
            hits_a, hits_g, show_s(row["anneal_wall"]), show_s(row["gmin_wall"]),
            show_us(row["anneal_per_call"]), show_us(row["gmin_per_call"]),
            fisher_p, calls_p,
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This setting against GMIN on rigid TIP4P water clusters.",
        "Seeds 2000--2011, one core, budget in force calls.",
        "Wall per run is the median total wall time of a run in seconds.",
        "A run stops at a hit, so it depends on both the cost of a call and the calls spent.",
        "Per call is the median over runs of wall time over calls in \\(\\mu\\)s, the cost of a call alone.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits and \\(p_\\mathrm{calls}\\)",
        "the two-sided exact Mann--Whitney test on calls to a hit, with a miss ranked at the budget.",
        "Both are Bonferroni-corrected with \\(m=4\\) over the four sizes that have a Cambridge header.",
        "\\((\\mathrm{H_2O})_{25}\\) is not scored.}",
        "\\label{tab:tip4p-gmin}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Wall per run (s)} & "
        "\\multicolumn{2}{c}{Per call (\\(\\mu\\)s)} & & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
        "Cluster & Budget & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{F}\\) & \\(p_\\mathrm{calls}\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def gofee_si_block() -> str:
    rows = [r for r in read(GOFEE / "asg2-summary.csv") if r["source"] == "measured"]
    body = []
    for system in OTHER_LABEL:
        first = True
        for arm in ("ase_bh", "ase_mh", "agox_gofee"):
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                OTHER_LABEL[system] if first else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if first else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
            first = False
        if system != "tip4p8":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{ASE basin hopping (BH), ASE minima hopping (MH) and AGOX GOFEE on the",
        "non-Lennard-Jones panel systems, on the potential of this code.",
        "Six seeds, one core, budget in force calls, and a wall cap of 2 hours per run.",
        "Capped counts the runs stopped by the cap, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.}",
        "\\label{tab:gofee-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "System & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


BLOCKS = {
    "gofee": (MANUSCRIPT, gofee_block),
    "tip4p": (MANUSCRIPT, tip4p_block),
    "gofee-si": (SI, gofee_si_block),
}


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{start!r}: the file needs exactly one block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{block}\n{end}{tail}"


def main(argv: list[str]) -> int:
    stale = False
    for name, (path, make) in BLOCKS.items():
        text = path.read_text()
        updated = splice(text, MARKS[name], make())
        if "--check" in argv:
            if updated != text:
                print(f"the {name} table in {path.name} is stale", file=sys.stderr)
                stale = True
        else:
            path.write_text(updated)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
