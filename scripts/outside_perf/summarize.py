#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize one three-arm outside-potential session."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

SYSTEMS = ("lj150", "lj104", "m80r6", "m80r14", "au55", "blj55s130", "blj38s105")
BLJ_SYSTEMS = {"blj55s130", "blj38s105"}
ARMS = ("develop", "develop_prime", "new")
SEEDS = list(range(2000, 2012))
SUMMARY_FIELDS = [
    "system",
    "outside_metric",
    "n",
    "develop_outside_q1",
    "develop_outside_median",
    "develop_outside_q3",
    "develop_prime_outside_q1",
    "develop_prime_outside_median",
    "develop_prime_outside_q3",
    "new_outside_q1",
    "new_outside_median",
    "new_outside_q3",
    "aa_relative_change",
    "aa_mannwhitney_u",
    "aa_mannwhitney_p",
    "new_relative_change",
    "new_mannwhitney_u",
    "new_mannwhitney_p",
]
TRAJECTORY_FIELDS = [
    "system",
    "seed",
    "develop_hit",
    "develop_prime_hit",
    "new_hit",
    "develop_evals_to_hit",
    "develop_prime_evals_to_hit",
    "new_evals_to_hit",
    "develop_charged",
    "develop_prime_charged",
    "new_charged",
    "develop_best",
    "develop_prime_best",
    "new_best",
    "new_identical",
    "aa_identical",
]
IDENTITY_FIELDS = ("hit", "evals_to_hit", "charged", "best")


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


def outside_values(rows: list[dict[str, str]], system: str) -> list[float]:
    if system in BLJ_SYSTEMS:
        return [float(row["outside_s"]) * 1.0e5 / float(row["charged"]) for row in rows]
    return [float(row["outside_s"]) for row in rows]


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
    sessions = sorted({row["session"] for row in rows})
    if len(sessions) != 1:
        raise SystemExit(f"expected one session, got {sessions}")
    if len(rows) != 252:
        raise SystemExit(f"expected 252 rows, got {len(rows)}")
    if {row["core"] for row in rows} != {"0"} or {row["affinity"] for row in rows} != {"0"}:
        raise SystemExit("rows were not pinned to core 0")
    if len({row["machine_sha256"] for row in rows}) != 1:
        raise SystemExit("rows span more than one machine digest")
    summaries = []
    trajectory = []
    mismatches = []
    for system in SYSTEMS:
        groups = {
            arm: [row for row in rows if row["system"] == system and row["arm"] == arm]
            for arm in ARMS
        }
        for arm, group in groups.items():
            seeds = sorted(int(row["seed"]) for row in group)
            if seeds != SEEDS:
                raise SystemExit(f"{system} {arm}: seeds {seeds}")
        by_key = {
            (row["arm"], int(row["seed"])): row
            for arm in ARMS
            for row in groups[arm]
        }
        for seed in SEEDS:
            observed = {arm: by_key[(arm, seed)] for arm in ARMS}
            new_identical = all(
                observed["develop"][field] == observed["new"][field] for field in IDENTITY_FIELDS
            )
            aa_identical = all(
                observed["develop"][field] == observed["develop_prime"][field]
                for field in IDENTITY_FIELDS
            )
            trajectory.append(
                {
                    "system": system,
                    "seed": str(seed),
                    "develop_hit": observed["develop"]["hit"],
                    "develop_prime_hit": observed["develop_prime"]["hit"],
                    "new_hit": observed["new"]["hit"],
                    "develop_evals_to_hit": observed["develop"]["evals_to_hit"],
                    "develop_prime_evals_to_hit": observed["develop_prime"]["evals_to_hit"],
                    "new_evals_to_hit": observed["new"]["evals_to_hit"],
                    "develop_charged": observed["develop"]["charged"],
                    "develop_prime_charged": observed["develop_prime"]["charged"],
                    "new_charged": observed["new"]["charged"],
                    "develop_best": observed["develop"]["best"],
                    "develop_prime_best": observed["develop_prime"]["best"],
                    "new_best": observed["new"]["best"],
                    "new_identical": "1" if new_identical else "0",
                    "aa_identical": "1" if aa_identical else "0",
                }
            )
            if not new_identical:
                mismatches.append(f"new differs for {system} seed {seed}")
        values = {arm: outside_values(groups[arm], system) for arm in ARMS}
        result = {
            "system": system,
            "outside_metric": "outside_s_per_1e5_calls" if system in BLJ_SYSTEMS else "outside_s",
            "n": "12",
        }
        for prefix in ("develop", "develop_prime", "new"):
            result[f"{prefix}_outside_q1"] = f"{type7(values[prefix], 0.25):.6f}"
            result[f"{prefix}_outside_median"] = f"{type7(values[prefix], 0.5):.6f}"
            result[f"{prefix}_outside_q3"] = f"{type7(values[prefix], 0.75):.6f}"
        for label, arm in (("aa", "develop_prime"), ("new", "new")):
            base = type7(values["develop"], 0.5)
            other = type7(values[arm], 0.5)
            u, p = mann_whitney(values["develop"], values[arm])
            result[f"{label}_relative_change"] = f"{other / base - 1.0:.6f}"
            result[f"{label}_mannwhitney_u"] = f"{u:.1f}"
            result[f"{label}_mannwhitney_p"] = f"{p:.6g}"
        summaries.append(result)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write(args.summary, SUMMARY_FIELDS, summaries)
    write(args.trajectory_diff, TRAJECTORY_FIELDS, trajectory)
    if mismatches:
        raise SystemExit("\n".join(mismatches))


if __name__ == "__main__":
    main()
