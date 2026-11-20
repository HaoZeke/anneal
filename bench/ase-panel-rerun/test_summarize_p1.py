#!/usr/bin/env python3
# Copyright (c) anneal developers
"""LJ55 basin hopping reports only the wall-stopped call range."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize_p1  # noqa: E402

ROWS = HERE / "lj55" / "rows.csv"


def wall_range() -> tuple[int, int]:
    with ROWS.open(newline="") as handle:
        charged = [
            int(row["charged"])
            for row in csv.DictReader(handle)
            if row["system"] == "lj55" and row["arm"] == "ase_bh" and row["stop"] == "wall"
        ]
    return min(charged), max(charged)


def main() -> None:
    low, high = wall_range()
    if (low, high) != (200184, 245800):
        raise SystemExit(f"lj55 ase_bh wall range {low}-{high}")
    _summary, report = summarize_p1.render("lj55")
    sentence = (
        "Basin hopping stopped at the wall cap on 11/12 seeds; "
        "those rows charged 200,184-245,800 calls."
    )
    if sentence not in report:
        raise SystemExit("lj55 readme does not report the wall-stopped range")
    if "75,325" in report:
        raise SystemExit("lj55 readme includes the hit row in the wall range")


if __name__ == "__main__":
    main()
