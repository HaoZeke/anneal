#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Rebuild the same-session portfolio summary from its own rows.

Both arms are the runs in sc-session/rows.csv. The published
twelve-seed table is not read. A median is the middle value, or the
mean of the two middle values when the count is even.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench/metal-cluster-h2h/sc-session/rows.csv"
SUMMARY = ROOT / "bench/metal-cluster-h2h/sc-session/summary.csv"

SYSTEMS = ("sc126-80", "sc96-80", "sc108-80", "sc108-55")
VARIANTS = ("baseline", "portfolio")
PER_ARM = 24
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
    runs = list(csv.DictReader(ROWS.open()))
    lines = [HEADER]
    for variant in VARIANTS:
        for system in SYSTEMS:
            group = [
                row
                for row in runs
                if row["variant"] == variant and row["system"] == system
            ]
            if len(group) != PER_ARM:
                raise SystemExit(
                    f"{variant} {system} has {len(group)} rows, expected {PER_ARM}"
                )
            lines.append(_line(variant, system, group))
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
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_bytes(text.encode())


if __name__ == "__main__":
    main()
