"""Aggregate sharded SOTA comparison CSVs and report dominance metrics.

Reads the per-shard CSVs written by ``sota_cutest.py`` and reports, per
method: win counts (cell best within tolerance), mean rank, solved
fraction under the Dolan-More criterion with the cross-method best as
the reference, and head-to-head records of the anneal drivers against
each baseline.

Usage:
  python -m experiments.scripts.summarize_sota results/sota_shard_*.csv
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

DM_TAU = 1e-3
WIN_ATOL = 1e-9
WIN_RTOL = 1e-9
VALID_STATUSES = {"ok", "budget_exhausted"}


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _valid_finite(row):
    return row.get("status", "ok") in VALID_STATUSES and math.isfinite(
        float(row["best"])
    )


def summarize_rows(rows):
    """Return tie-aware metrics from successful, finite solver cells."""
    cells = defaultdict(dict)
    for row in rows:
        key = (row["problem"], row["seed"])
        cells[key][row["method"]] = row

    methods = sorted({row["method"] for row in rows})
    wins = defaultdict(int)
    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    solved = defaultdict(int)
    solved_n = defaultdict(int)

    for reported in cells.values():
        cell = {
            method: float(row["best"])
            for method, row in reported.items()
            if _valid_finite(row)
        }
        if not cell:
            for method in methods:
                solved_n[method] += 1
            continue
        best = min(cell.values())
        starts = {
            float(row["initial"])
            for row in reported.values()
            if "initial" in row and math.isfinite(float(row["initial"]))
        }
        if len(starts) > 1:
            raise ValueError(
                f"problem-seed cell has inconsistent starting objectives: {starts}"
            )
        if starts:
            initial = starts.pop()
            if best > initial + WIN_ATOL + WIN_RTOL * abs(initial):
                raise ValueError("best observed value is worse than the retained start")
            target = best + DM_TAU * (initial - best)
        else:
            target = best + DM_TAU * max(max(cell.values()) - best, 1.0)
        for method in methods:
            solved_n[method] += 1
        for m, v in cell.items():
            if v <= best + WIN_ATOL + WIN_RTOL * abs(best):
                wins[m] += 1
            if v <= target:
                solved[m] += 1
        ordered = sorted(cell.items(), key=lambda item: item[1])
        i = 0
        while i < len(ordered):
            j = i + 1
            plateau = ordered[i][1]
            tolerance = WIN_ATOL + WIN_RTOL * abs(plateau)
            while j < len(ordered) and abs(ordered[j][1] - plateau) <= tolerance:
                j += 1
            average_rank = 0.5 * ((i + 1) + j)
            for method, _ in ordered[i:j]:
                rank_sum[method] += average_rank
                rank_n[method] += 1
            i = j

    return {
        method: {
            "wins": wins[method],
            "mean_rank": rank_sum[method] / rank_n[method],
            "near_best": solved[method],
            "eligible_cells": solved_n[method],
        }
        for method in methods
        if rank_n[method]
    }


def failure_aware_mean_ranks(rows):
    """Rank every reported method, tying unsuccessful cells at the bottom."""
    cells = defaultdict(dict)
    methods = sorted({row["method"] for row in rows})
    for row in rows:
        cells[(row["problem"], row["seed"])][row["method"]] = row

    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    for cell in cells.values():
        successful = sorted(
            (
                (method, float(row["best"]))
                for method, row in cell.items()
                if _valid_finite(row)
            ),
            key=lambda item: item[1],
        )
        i = 0
        while i < len(successful):
            j = i + 1
            plateau = successful[i][1]
            tolerance = WIN_ATOL + WIN_RTOL * abs(plateau)
            while j < len(successful) and abs(successful[j][1] - plateau) <= tolerance:
                j += 1
            average_rank = 0.5 * ((i + 1) + j)
            for method, _ in successful[i:j]:
                rank_sum[method] += average_rank
                rank_n[method] += 1
            i = j

        unsuccessful = [
            method
            for method in methods
            if method not in cell or not _valid_finite(cell[method])
        ]
        if unsuccessful:
            first = len(successful) + 1
            last = len(successful) + len(unsuccessful)
            average_rank = 0.5 * (first + last)
            for method in unsuccessful:
                rank_sum[method] += average_rank
                rank_n[method] += 1

    return {
        method: rank_sum[method] / rank_n[method]
        for method in methods
        if rank_n[method]
    }


def main(paths):
    rows = load_rows(paths)
    if not rows:
        print("no rows")
        return 1
    summary = summarize_rows(rows)
    penalized_ranks = failure_aware_mean_ranks(rows)
    all_cells = {(row["problem"], row["seed"]) for row in rows}
    n_cells = len(all_cells)
    methods = sorted({row["method"] for row in rows})
    failures = defaultdict(int)
    for row in rows:
        if not _valid_finite(row):
            failures[row["method"]] += 1

    print(f"{n_cells} cells, {len(methods)} methods\n")
    print(
        f"{'method':>16} {'wins':>6} {'win%':>6} {'meanrank':>9} "
        f"{'near-best%':>10} {'fail':>6}"
    )
    for m in sorted(methods, key=lambda k: -summary.get(k, {}).get("wins", 0)):
        metrics = summary.get(m)
        if metrics is None:
            print(
                f"{m:>16} {0:>6} {0.0:>5.1f}% "
                f"{penalized_ranks[m]:>9.2f} {0.0:>9.1f}% {failures[m]:>6}"
            )
            continue
        near = 100.0 * metrics["near_best"] / metrics["eligible_cells"]
        print(
            f"{m:>16} {metrics['wins']:>6} "
            f"{100.0 * metrics['wins'] / n_cells:>5.1f}% "
            f"{penalized_ranks[m]:>9.2f} {near:>9.1f}% {failures[m]:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
