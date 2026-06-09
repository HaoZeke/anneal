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


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main(paths):
    rows = load_rows(paths)
    if not rows:
        print("no rows")
        return 1
    cells = defaultdict(dict)
    dims = {}
    for row in rows:
        key = (row["problem"], row["seed"])
        value = float(row["best"])
        cells[key][row["method"]] = value
        dims[row["problem"]] = int(row["dim"])

    methods = sorted({m for cell in cells.values() for m in cell})
    wins = defaultdict(int)
    rank_sum = defaultdict(float)
    rank_n = defaultdict(int)
    solved = defaultdict(int)
    solved_n = defaultdict(int)
    head_to_head = defaultdict(lambda: [0, 0, 0])  # better, tied, worse

    for cell in cells.values():
        finite = {m: v for m, v in cell.items() if math.isfinite(v)}
        if not finite:
            continue
        best = min(finite.values())
        worst = max(finite.values())
        spread = max(worst - best, 1.0)
        for m, v in finite.items():
            if v <= best + WIN_ATOL + WIN_RTOL * abs(best):
                wins[m] += 1
            solved_n[m] += 1
            if v <= best + DM_TAU * spread:
                solved[m] += 1
        order = sorted(finite, key=finite.get)
        for rank, m in enumerate(order, start=1):
            rank_sum[m] += rank
            rank_n[m] += 1
        for ours in ("portfolio", "hybrid_de"):
            if ours not in finite:
                continue
            for baseline in ("basinhopping", "dual_annealing", "diff_evol", "classical"):
                if baseline not in finite:
                    continue
                pair = head_to_head[(ours, baseline)]
                margin = WIN_ATOL + WIN_RTOL * abs(finite[baseline])
                if finite[ours] < finite[baseline] - margin:
                    pair[0] += 1
                elif finite[ours] > finite[baseline] + margin:
                    pair[2] += 1
                else:
                    pair[1] += 1

    n_cells = len(cells)
    print(f"{n_cells} cells, {len(methods)} methods\n")
    print(f"{'method':>16} {'wins':>6} {'win%':>6} {'meanrank':>9} {'near-best%':>10}")
    for m in sorted(methods, key=lambda k: -wins[k]):
        mean_rank = rank_sum[m] / max(rank_n[m], 1)
        near = 100.0 * solved[m] / max(solved_n[m], 1)
        print(
            f"{m:>16} {wins[m]:>6} {100.0 * wins[m] / n_cells:>5.1f}% "
            f"{mean_rank:>9.2f} {near:>9.1f}%"
        )
    print("\nhead-to-head (better / tied / worse):")
    for (ours, baseline), (b, t, w) in sorted(head_to_head.items()):
        verdict = "DOMINATES" if w == 0 and b > 0 else ""
        print(f"  {ours:>10} vs {baseline:<14} {b:>4} / {t:>4} / {w:>4}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
