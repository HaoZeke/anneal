#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Markdown tables of the covalent panel.

    tables.py <before results.csv> <after results.csv>
    tables.py panel <results.csv> <seeds.csv>

The first form is the first panel, before and after the fix; GMIN and ASE
ran once, in the before session, and appear in both columns. The second
prints one session's codes and the two-sided p-values of each comparison:
Mann-Whitney U on calls to the hit, a miss counted at the budget, and
Fisher's exact test on hits.
"""

from __future__ import annotations

import csv
import sys

from scipy.stats import fisher_exact, mannwhitneyu

SYSTEMS = ("sw10", "sw13", "sw20", "tc10", "tc20")
CODES = ("anneal", "anneal-nopriors", "anneal-general", "gmin", "ase")
PANEL = (
    "anneal",
    "anneal-nopriors@v1",
    "anneal-nopriors",
    "anneal-general@v1",
    "anneal-general",
    "anneal-sloppy",
    "anneal-nopriors-sloppy",
    "anneal-general-sloppy",
    "gmin",
    "gmin-tight",
    "ase",
)
NAMES = {
    "anneal": "anneal R",
    "anneal-nopriors": "anneal P",
    "anneal-general": "anneal G",
    "anneal-nopriors@v1": "anneal P v1",
    "anneal-general@v1": "anneal G v1",
    "anneal-sloppy": "anneal R 5e-3",
    "anneal-nopriors-sloppy": "anneal P 5e-3",
    "anneal-general-sloppy": "anneal G 5e-3",
    "gmin": "GMIN 5e-3",
    "gmin-tight": "GMIN 1e-6",
    "ase": "ASE BH",
}
# Each pair quenches to the same tolerance, or is the same arm before and
# after.
PAIRS = (
    ("anneal-nopriors", "anneal-nopriors@v1"),
    ("anneal-general", "anneal-general@v1"),
    ("anneal-nopriors", "anneal"),
    ("anneal-general", "anneal"),
    ("anneal", "gmin-tight"),
    ("anneal-nopriors", "gmin-tight"),
    ("anneal-general", "gmin-tight"),
    ("anneal-sloppy", "gmin"),
    ("anneal-nopriors-sloppy", "gmin"),
    ("anneal-general-sloppy", "gmin"),
)


def load(path: str) -> dict[tuple[str, str], dict[str, str]]:
    with open(path, newline="") as handle:
        return {(row["system"], row["code"]): row for row in csv.DictReader(handle)}


def calls(row: dict[str, str]) -> str:
    return f"{float(row['evals_median']):,.0f}"


def seed_rows(path: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault((row["system"], row["code"]), []).append(row)
    return out


def p_values(a: list[dict[str, str]], b: list[dict[str, str]]) -> tuple[float, float]:
    calls_a = [float(row["evals_to_hit"]) for row in a]
    calls_b = [float(row["evals_to_hit"]) for row in b]
    hits_a = sum(row["hit"] == "true" for row in a)
    hits_b = sum(row["hit"] == "true" for row in b)
    mw = mannwhitneyu(calls_a, calls_b, alternative="two-sided").pvalue
    fisher = fisher_exact([[hits_a, len(a) - hits_a], [hits_b, len(b) - hits_b]]).pvalue
    return float(mw), float(fisher)


def panel(results_path: str, seeds_path: str) -> int:
    results, seeds = load(results_path), seed_rows(seeds_path)
    print("| system | code | hits | median calls | IQR | best | pot_s | outside_s |")
    print("|---|---|---|---|---|---|---|---|")
    for system in SYSTEMS:
        for code in PANEL:
            r = results.get((system, code))
            if r is None:
                continue
            print(
                f"| {system} | {NAMES[code]} | {r['hits']}/{r['n']} | {calls(r)} "
                f"| {float(r['evals_iqr']):,.0f} | {float(r['best_min']):.6f} "
                f"| {float(r['pot_median_s']):.3f} | {float(r['outside_median_s']):.3f} |"
            )
    print()
    print("| system | comparison | median calls | hits | Mann-Whitney p | Fisher p |")
    print("|---|---|---|---|---|---|")
    for system in SYSTEMS:
        for left, right in PAIRS:
            a, b = seeds.get((system, left)), seeds.get((system, right))
            if not a or not b:
                continue
            mw, fisher = p_values(a, b)
            ra, rb = results[(system, left)], results[(system, right)]
            print(
                f"| {system} | {NAMES[left]} vs {NAMES[right]} | {calls(ra)} vs {calls(rb)} "
                f"| {ra['hits']}/{ra['n']} vs {rb['hits']}/{rb['n']} | {mw:.2g} | {fisher:.2g} |".replace("| nan |", "| tied |")
            )
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "panel":
        return panel(argv[2], argv[3])
    before, after = load(argv[1]), load(argv[2])
    print("| system | code | hits before | hits after | median calls before | median calls after "
          "| best before | best after | pot_s before | pot_s after | outside_s before | outside_s after |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for system in SYSTEMS:
        for code in CODES:
            b = before.get((system, code))
            if b is None:
                continue
            a = after.get((system, code), b)
            print(
                f"| {system} | {NAMES[code]} | {b['hits']}/{b['n']} | {a['hits']}/{a['n']} "
                f"| {calls(b)} | {calls(a)} | {float(b['best_min']):.6f} | {float(a['best_min']):.6f} "
                f"| {float(b['pot_median_s']):.3f} | {float(a['pot_median_s']):.3f} "
                f"| {float(b['outside_median_s']):.3f} | {float(a['outside_median_s']):.3f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
