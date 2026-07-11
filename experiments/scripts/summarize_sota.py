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
        if _valid_finite(row):
            cells[key][row["method"]] = float(row["best"])

    methods = sorted({m for cell in cells.values() for m in cell})
    wins = defaultdict(int)
    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    solved = defaultdict(int)
    solved_n = defaultdict(int)

    for cell in cells.values():
        if not cell:
            continue
        best = min(cell.values())
        worst = max(cell.values())
        spread = max(worst - best, 1.0)
        for m, v in cell.items():
            if v <= best + WIN_ATOL + WIN_RTOL * abs(best):
                wins[m] += 1
            solved_n[m] += 1
            if v <= best + DM_TAU * spread:
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


def main(paths):
    rows = load_rows(paths)
    if not rows:
        print("no rows")
        return 1
    summary = summarize_rows(rows)
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
                f"{m:>16} {0:>6} {0.0:>5.1f}% {'nan':>9} {0.0:>9.1f}% {failures[m]:>6}"
            )
            continue
        near = 100.0 * metrics["near_best"] / metrics["eligible_cells"]
        print(
            f"{m:>16} {metrics['wins']:>6} "
            f"{100.0 * metrics['wins'] / n_cells:>5.1f}% "
            f"{metrics['mean_rank']:>9.2f} {near:>9.1f}% {failures[m]:>6}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
