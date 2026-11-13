#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Markdown tables of the covalent panel, before and after the fix.

    tables.py <before results.csv> <after results.csv>

GMIN and ASE ran once, in the before session, and appear in both columns.
"""

from __future__ import annotations

import csv
import sys

SYSTEMS = ("sw10", "sw13", "sw20", "tc10", "tc20")
CODES = ("anneal", "anneal-nopriors", "anneal-general", "gmin", "ase")
NAMES = {
    "anneal": "anneal R",
    "anneal-nopriors": "anneal P",
    "anneal-general": "anneal G",
    "gmin": "GMIN",
    "ase": "ASE BH",
}


def load(path: str) -> dict[tuple[str, str], dict[str, str]]:
    with open(path, newline="") as handle:
        return {(row["system"], row["code"]): row for row in csv.DictReader(handle)}


def calls(row: dict[str, str]) -> str:
    return f"{float(row['evals_median']):,.0f}"


def main(argv: list[str]) -> int:
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
