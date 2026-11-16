#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fisher, tie-corrected Mann-Whitney, and the median gap for the LJ75 trail gate.

Reads bench/leave-lj75/rows.csv. Writes bench/leave-lj75/summary.csv.
Misses are censored at the budget for the rank test. Does not start a search.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench" / "leave-lj75" / "rows.csv"
SUMMARY = ROOT / "bench" / "leave-lj75" / "summary.csv"
REFERENCE = -397.492331
SEEDS = tuple(range(2000, 2012))
ARMS = ("base", "leave")


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
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


def load() -> dict[str, list[dict[str, str]]]:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        if row["system"] != "lj75" or row["budget"] != "4000000":
            raise SystemExit(f"unexpected row {row['system']} {row['budget']}")
        if row["hostname"] != "bench":
            raise SystemExit(f"host is {row['hostname']}")
        grouped[row["arm"]].append(row)
    for arm in ARMS:
        seeds = sorted(int(row["seed"]) for row in grouped[arm])
        if tuple(seeds) != SEEDS:
            raise SystemExit(f"{arm} seeds {seeds}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def fisher(hits_leave: int, hits_base: int) -> float:
    table = [[hits_leave, 12 - hits_leave], [hits_base, 12 - hits_base]]
    return float(fisher_exact(table)[1])


def mann_whitney(leave: list[float], base: list[float]) -> tuple[float, float]:
    if len(set(leave + base)) == 1:
        return len(leave) * len(base) / 2.0, 1.0
    result = mannwhitneyu(leave, base, alternative="two-sided", method="asymptotic", use_continuity=True)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    grouped = load()
    lines = [
        "arm,hits,n,median_best,gap_to_reference,median_inside_s,median_outside_s,median_wall_s",
    ]
    stats = {}
    for arm in ARMS:
        group = grouped[arm]
        hits = sum(row["hit"] == "1" for row in group)
        best = median([float(row["best"]) for row in group])
        inside = median([float(row["inside_s"]) for row in group])
        outside = median([float(row["outside_s"]) for row in group])
        wall = median([float(row["wall_s"]) for row in group])
        stats[arm] = {
            "hits": hits,
            "calls": [calls(row) for row in group],
            "best": best,
            "gap": best - REFERENCE,
            "inside": inside,
            "outside": outside,
            "wall": wall,
        }
        print(
            f"{arm} hits {hits}/12 median_best {best:.8f} gap {best - REFERENCE:.8f} "
            f"inside {inside:.6f} outside {outside:.6f} wall {wall:.6f}"
        )
        lines.append(
            f"{arm},{hits},12,{best:.8f},{best - REFERENCE:.8f},{inside:.6f},{outside:.6f},{wall:.6f}"
        )
    fisher_p = fisher(stats["leave"]["hits"], stats["base"]["hits"])
    u_stat, mw_p = mann_whitney(stats["leave"]["calls"], stats["base"]["calls"])
    print(f"fisher_p {fisher_p:.6g} mann_whitney_U {u_stat:.6g} mann_whitney_p {mw_p:.6g}")
    lines.append(f"compare,fisher_p,{fisher_p:.6g},mann_whitney_U,{u_stat:.6g},mann_whitney_p,{mw_p:.6g}")
    SUMMARY.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
