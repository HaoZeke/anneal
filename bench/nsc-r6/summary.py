#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Recompute the README table from rows.csv.

A hit is a best energy within 1e-3 of the reference. evals_to_hit is the
charged count at the first improvement within that tolerance; a miss counts
at the budget. Fisher exact (two-sided) on hits, Mann-Whitney U (two-sided)
on evals_to_hit.

    python3 bench/nsc-r6/summary.py
"""

import csv
from pathlib import Path
from statistics import median

from scipy.stats import fisher_exact, mannwhitneyu

ROWS = Path(__file__).with_name("rows.csv")
PATHS = ["cluster_search::search", "h2h_seed recommended"]
SYSTEMS = ["lj103", "m38r6", "m55r14"]
ARMS = ["A", "B", "C"]
PAIRS = [("B", "A"), ("C", "A"), ("C", "B")]


def p_values(x, y):
    hx = sum(int(r["hit"]) for r in x)
    hy = sum(int(r["hit"]) for r in y)
    fisher = fisher_exact([[hx, len(x) - hx], [hy, len(y) - hy]])[1]
    cx = [int(r["evals_to_hit"]) for r in x]
    cy = [int(r["evals_to_hit"]) for r in y]
    if sorted(cx) == sorted(cy):
        mw = 1.0
    else:
        mw = mannwhitneyu(cx, cy, alternative="two-sided").pvalue
    return fisher, mw


def main():
    rows = list(csv.DictReader(ROWS.open()))
    head = ["path", "system"]
    head += [f"{a} hits" for a in ARMS] + [f"{a} median" for a in ARMS]
    for x, y in PAIRS:
        head += [f"Fisher {x}/{y}", f"MW {x}/{y}"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for path in PATHS:
        for system in SYSTEMS:
            arm = {
                a: [r for r in rows if r["path"] == path and r["system"] == system and r["arm"] == a]
                for a in ARMS
            }
            cells = [path, system]
            cells += [f"{sum(int(r['hit']) for r in arm[a])}/{len(arm[a])}" for a in ARMS]
            cells += [f"{median(int(r['evals_to_hit']) for r in arm[a]):.0f}" for a in ARMS]
            for x, y in PAIRS:
                fisher, mw = p_values(arm[x], arm[y])
                cells += [f"{fisher:.3g}", f"{mw:.3g}"]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
