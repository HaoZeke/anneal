#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint type-7 wall medians and hit counts for the LJ75 AGOX rows.

Reads bench/agox-lj75/rows.csv. Writes bench/agox-lj75/summary.csv.
Does not start a search.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench" / "agox-lj75" / "rows.csv"
SUMMARY = ROOT / "bench" / "agox-lj75" / "summary.csv"
CODES = ("agox", "agox-documented", "anneal")
SEEDS = (6400, 6401, 6402, 6403, 6404, 6405)
REFERENCE = -397.492331


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    if count == 0:
        raise SystemExit("no values for a median")
    if count == 1:
        return float(ordered[0])
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    right = ordered[int(lower)]
    return float(left * (1.0 - weight) + right * weight)


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def load() -> list[dict[str, str]]:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 18:
        raise SystemExit(f"{ROWS} has {len(rows)} rows, expected 18")
    seen = {(row["code"], int(row["seed"])) for row in rows}
    for code in CODES:
        for seed in SEEDS:
            if (code, seed) not in seen:
                raise SystemExit(f"missing {code} seed {seed}")
    for row in rows:
        if row["system"] != "lj75":
            raise SystemExit(f"unexpected system {row['system']}")
        if row["host"] != "bench":
            raise SystemExit(f"host is {row['host']}, expected bench")
        if row["budget"] != "4000000":
            raise SystemExit(f"budget is {row['budget']}")
    return rows


def by_code(rows: list[dict[str, str]], code: str) -> list[dict[str, str]]:
    chosen = [row for row in rows if row["code"] == code]
    chosen.sort(key=lambda row: int(row["seed"]))
    return chosen


def paired_ratio(left: list[dict[str, str]], right: list[dict[str, str]]) -> list[float]:
    ratios = []
    for one, other in zip(left, right):
        if one["seed"] != other["seed"]:
            raise SystemExit("seed order drifted")
        ratios.append(float(one["wall_s"]) / float(other["wall_s"]))
    return ratios


def main() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7 median drifted")
    rows = load()
    grouped = {code: by_code(rows, code) for code in CODES}
    anneal = grouped["anneal"]
    lines = [
        "code,hits,n,median_wall_s,median_best,gap_to_reference,median_over_anneal,median_anneal_over",
    ]
    print("LJ75 seeds 6400-6405, budget 4000000, host bench, reference -397.492331")
    for code in CODES:
        group = grouped[code]
        hits = sum(int(row["hit"]) for row in group)
        wall = median([float(row["wall_s"]) for row in group])
        best = median([float(row["best"]) for row in group])
        gap = best - REFERENCE
        if code == "anneal":
            over = 1.0
            back = 1.0
        else:
            over = median(paired_ratio(group, anneal))
            back = median(paired_ratio(anneal, group))
        print(
            f"{code} hits {hits}/6 median_wall_s {wall:.6f} "
            f"median_best {best:.6f} gap {gap:.6f} "
            f"median_over_anneal {over:.6f} median_anneal_over {back:.6f}"
        )
        lines.append(
            f"{code},{hits},6,{wall:.6f},{best:.6f},{gap:.6f},{over:.6f},{back:.6f}"
        )
    SUMMARY.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
