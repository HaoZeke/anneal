"""Render Dolan-Moré, Moré-Wild, and Pareto figures from a benchmark CSV.

Reads the long-form CSV emitted by `run_benchmarks.py` (one row per
(problem, solver, seed) triple), pivots into the matrix form each plot
helper expects, and writes three PNGs into the output directory."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from experiments.plots.data_profile import plot_data_profile
from experiments.plots.pareto import plot_pareto
from experiments.plots.performance_profile import plot_performance_profile


def load_long_csv(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(rows):
    """Best-of-seeds per (problem, solver). Cost = fevals on solved runs;
    NaN otherwise (failed)."""
    best = defaultdict(lambda: {"fevals": [], "best_val": [], "dim": None, "solved": []})
    for r in rows:
        key = (r["problem"], r["driver"])
        best[key]["fevals"].append(int(r["fevals"]))
        best[key]["best_val"].append(float(r["best_val"]))
        best[key]["dim"] = int(r["dim"])
        best[key]["solved"].append(int(r["solved"]))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("out_dir")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_long_csv(args.csv_path)
    agg = aggregate(rows)

    problems = sorted({r["problem"] for r in rows})
    solvers = sorted({r["driver"] for r in rows})
    n_p, n_s = len(problems), len(solvers)

    fevals_matrix = np.full((n_p, n_s), np.nan, dtype=float)
    val_matrix = np.full((n_p, n_s), np.nan, dtype=float)
    dims = np.zeros(n_p, dtype=int)
    for i, prob in enumerate(problems):
        for j, solver in enumerate(solvers):
            entry = agg.get((prob, solver))
            if entry is None or not entry["solved"]:
                continue
            solved_idx = [k for k, s in enumerate(entry["solved"]) if s]
            if not solved_idx:
                continue
            fevals_matrix[i, j] = float(np.median([entry["fevals"][k] for k in solved_idx]))
            val_matrix[i, j] = float(np.median([entry["best_val"][k] for k in solved_idx]))
            dims[i] = entry["dim"]

    plot_performance_profile(
        fevals_matrix,
        solvers,
        title="Performance profile (function evaluations)",
        out_path=os.path.join(args.out_dir, "performance_profile.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'performance_profile.png')}")

    plot_data_profile(
        fevals_matrix,
        dims,
        solvers,
        title="Data profile (simplex-gradient budgets)",
        out_path=os.path.join(args.out_dir, "data_profile.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'data_profile.png')}")

    pareto_runs = []
    for j, solver in enumerate(solvers):
        pts = []
        for r in rows:
            if r["driver"] != solver:
                continue
            pts.append((float(r["fevals"]), float(r["best_val"])))
        pareto_runs.append((solver, np.asarray(pts)))
    plot_pareto(
        pareto_runs,
        cost_label="Function evaluations",
        value_label="Best objective value",
        title="Accuracy-vs-cost Pareto",
        out_path=os.path.join(args.out_dir, "pareto.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'pareto.png')}")


if __name__ == "__main__":
    sys.exit(main())
