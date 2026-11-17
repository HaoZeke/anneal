#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize interleaved base/HEAD outside-potential timings."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SYSTEMS = ("lj150", "lj104", "m80r6", "au55")
SUMMARY_FIELDS = [
    "system",
    "base_n",
    "new_n",
    "base_inside_median_s",
    "new_inside_median_s",
    "inside_relative_change",
    "inside_mannwhitney_u",
    "inside_mannwhitney_p",
    "base_outside_median_s",
    "new_outside_median_s",
    "outside_relative_change",
    "outside_mannwhitney_u",
    "outside_mannwhitney_p",
]
TRAJECTORY_FIELDS = [
    "system",
    "seed",
    "base_evals_to_hit",
    "new_evals_to_hit",
    "base_charged",
    "new_charged",
    "base_best",
    "new_best",
    "identical",
]


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = 1.0 + (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if fraction == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - fraction) + ordered[lower] * fraction


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    count = len(pooled)
    rank: dict[float, float] = {}
    ties = 0.0
    index = 0
    while index < count:
        end = index
        while end + 1 < count and pooled[end + 1] == pooled[index]:
            end += 1
        rank[pooled[index]] = (index + end + 2) / 2.0
        width = end - index + 1
        ties += width**3 - width
        index = end + 1
    u = sum(rank[value] for value in left) - n1 * (n1 + 1) / 2.0
    variance = n1 * n2 / 12.0 * ((count + 1) - ties / (count * (count - 1)))
    if variance <= 0.0:
        return u, 1.0
    deviation = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(deviation / math.sqrt(variance) / math.sqrt(2.0)))


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trajectory-diff", type=Path, required=True)
    args = parser.parse_args()
    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    sessions = {row["session"] for row in rows}
    hosts = {row["host"] for row in rows}
    if len(sessions) != 1 or hosts != {"bench"}:
        raise SystemExit(f"expected one bench session, got sessions={sessions} hosts={hosts}")
    summaries = []
    trajectory = []
    for system in SYSTEMS:
        groups = {
            arm: [row for row in rows if row["system"] == system and row["arm"] == arm]
            for arm in ("base", "new")
        }
        for arm, group in groups.items():
            seeds = sorted(int(row["seed"]) for row in group)
            if seeds != list(range(2000, 2012)):
                raise SystemExit(f"{system} {arm}: seeds {seeds}")
        by_key = {(row["arm"], int(row["seed"])): row for row in groups["base"] + groups["new"]}
        for seed in range(2000, 2012):
            base = by_key[("base", seed)]
            new = by_key[("new", seed)]
            identical = all(
                base[field] == new[field] for field in ("evals_to_hit", "charged", "best")
            )
            trajectory.append(
                {
                    "system": system,
                    "seed": str(seed),
                    "base_evals_to_hit": base["evals_to_hit"],
                    "new_evals_to_hit": new["evals_to_hit"],
                    "base_charged": base["charged"],
                    "new_charged": new["charged"],
                    "base_best": base["best"],
                    "new_best": new["best"],
                    "identical": "1" if identical else "0",
                }
            )
            if not identical:
                raise SystemExit(f"trajectory changed for {system} seed {seed}")
        result = {"system": system, "base_n": "12", "new_n": "12"}
        for metric, column in (("inside", "pot_s"), ("outside", "outside_s")):
            base_values = [float(row[column]) for row in groups["base"]]
            new_values = [float(row[column]) for row in groups["new"]]
            base_median = type7(base_values, 0.5)
            new_median = type7(new_values, 0.5)
            u, p = mann_whitney(base_values, new_values)
            result[f"base_{metric}_median_s"] = f"{base_median:.6f}"
            result[f"new_{metric}_median_s"] = f"{new_median:.6f}"
            result[f"{metric}_relative_change"] = f"{new_median / base_median - 1.0:.6f}"
            result[f"{metric}_mannwhitney_u"] = f"{u:.1f}"
            result[f"{metric}_mannwhitney_p"] = f"{p:.6g}"
        summaries.append(result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write(args.summary, SUMMARY_FIELDS, summaries)
    write(args.trajectory_diff, TRAJECTORY_FIELDS, trajectory)


if __name__ == "__main__":
    main()
