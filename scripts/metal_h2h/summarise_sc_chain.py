#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the half-budget chain summary from the raw rows.

The baseline is the anneal arm of the twelve-seed table. The chain
rows are the runs that spend the first half on the recommended chain
and the rest on two walkers. A median is the middle value, or the
mean of the two middle values when the count is even.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "bench/metal-cluster-h2h/sc-n12/seeds.csv"
ROWS = ROOT / "bench/metal-cluster-h2h/sc-chain/rows.csv"
SUMMARY = ROOT / "bench/metal-cluster-h2h/sc-chain/summary.csv"

SYSTEMS = ("sc126-80", "sc96-80", "sc108-80", "sc108-55")
HEADER = (
    "variant,system,hits,n,best_median,wall_median_s,"
    "pot_median_s,outside_median_s\n"
)


def _hit(value: str) -> bool:
    return value in ("1", "true")


def _line(variant: str, system: str, group: list[dict[str, str]]) -> str:
    hits = sum(_hit(row["hit"]) for row in group)
    return (
        f"{variant},{system},{hits},{len(group)},"
        f"{median(float(row['best']) for row in group):.8f},"
        f"{median(float(row['wall_s']) for row in group):.6f},"
        f"{median(float(row['pot_s']) for row in group):.6f},"
        f"{median(float(row['outside_s']) for row in group):.6f}\n"
    )


def summary_text() -> str:
    seeds = [
        row
        for row in csv.DictReader(SEEDS.open())
        if row["arm"] == "anneal" and row["system"] in SYSTEMS
    ]
    runs = list(csv.DictReader(ROWS.open()))
    lines = [HEADER]
    for system in SYSTEMS:
        group = [row for row in seeds if row["system"] == system]
        if len(group) != 12:
            raise SystemExit(f"baseline {system} has {len(group)} rows, expected 12")
        lines.append(_line("baseline", system, group))
    for system in SYSTEMS:
        group = [row for row in runs if row["system"] == system]
        if len(group) != 12:
            raise SystemExit(f"chain {system} has {len(group)} rows, expected 12")
        lines.append(_line("chain", system, group))
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero unless the summary on disk matches the rows",
    )
    args = parser.parse_args()
    text = summary_text()
    if args.check:
        found = SUMMARY.read_bytes()
        if found != text.encode():
            raise SystemExit("summary.csv does not match the raw rows")
        return
    SUMMARY.write_bytes(text.encode())


if __name__ == "__main__":
    main()
