"""Render Dolan-Moré, Moré-Wild, and Pareto figures from a benchmark CSV.

Reads the long-form CSV emitted by `run_benchmarks.py` (one row per
(problem, solver, seed) triple), pivots into the matrix form each plot
helper expects, and writes three PNGs into the output directory."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

from experiments.plots.data_profile import plot_data_profile
from experiments.plots.pareto import plot_pareto
from experiments.plots.performance_profile import plot_performance_profile


DRIVER_ORDER = [
    "classical",
    "bayesian_mixing_sa",
    "mcmc_sa_budgeted",
    "mcmc_sa_sparse_budgeted",
    "pt_sa_budgeted",
    "mcmc_sa",
    "mcmc_sa_sparse",
    "bgsa",
    "bgsa_auto",
    "bgsa_metad",
    "bgsa_pt_metad",
]

DRIVER_LABELS = {
    "classical": "Classical SA",
    "bayesian_mixing_sa": "Bayesian mixing",
    "mcmc_sa_budgeted": "MCMC-SA (budgeted)",
    "mcmc_sa_sparse_budgeted": "Sparse MCMC-SA (budgeted)",
    "pt_sa_budgeted": "PT-SA (budgeted)",
    "mcmc_sa": "MCMC-SA",
    "mcmc_sa_sparse": "Sparse MCMC-SA",
    "bgsa": "BGSA",
    "bgsa_auto": "Automatic BGSA",
    "bgsa_metad": "MetaD BGSA",
    "bgsa_pt_metad": "PT-MetaD BGSA",
}


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _status_ok(row: dict) -> bool:
    return row.get("status", "ok") == "ok"


def ordered_solvers(rows: list[dict]) -> list[str]:
    present = {str(row["driver"]) for row in rows}
    known = [driver for driver in DRIVER_ORDER if driver in present]
    unknown = sorted(present.difference(DRIVER_ORDER))
    return known + unknown


def display_solver_names(solver_names):
    return [DRIVER_LABELS.get(name, name.replace("_", " ")) for name in solver_names]


def load_long_csv(path: str):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def aggregate(rows):
    """Best-of-seeds per (problem, solver). Cost = fevals on solved runs;
    NaN otherwise (failed)."""
    best = defaultdict(
        lambda: {"fevals": [], "best_val": [], "dim": None, "solved": []}
    )
    for r in rows:
        key = (r["problem"], r["driver"])
        best[key]["fevals"].append(int(r["fevals"]))
        best[key]["best_val"].append(float(r["best_val"]))
        best[key]["dim"] = int(r["dim"])
        best[key]["solved"].append(int(r["solved"]))
    return best


def data_profile_kappa_max(fevals: np.ndarray, dims: np.ndarray) -> float:
    fevals = np.asarray(fevals, dtype=float)
    dims = np.asarray(dims, dtype=float)
    budget = fevals / (dims[:, None] + 1.0)
    finite = budget[np.isfinite(budget)]
    if finite.size == 0:
        return 200.0
    return float(max(200.0, np.quantile(finite, 0.95) * 1.05))


def pareto_points_by_solver(rows: list[dict], solvers: list[str]):
    cell_best: dict[tuple[str, str], float] = defaultdict(lambda: float("inf"))
    for row in rows:
        value = _float_or_nan(row.get("best_val"))
        if not _status_ok(row) or not math.isfinite(value):
            continue
        key = (str(row["problem"]), str(row["seed"]))
        cell_best[key] = min(cell_best[key], value)

    buckets: dict[str, list[tuple[float, float]]] = {solver: [] for solver in solvers}
    for row in rows:
        solver = str(row["driver"])
        if solver not in buckets or not _status_ok(row):
            continue
        value = _float_or_nan(row.get("best_val"))
        fevals = _float_or_nan(row.get("fevals"))
        if not math.isfinite(value) or not math.isfinite(fevals):
            continue
        key = (str(row["problem"]), str(row["seed"]))
        best = cell_best[key]
        if not math.isfinite(best):
            continue
        f_x0 = _float_or_nan(row.get("f_x0"))
        denom = max(abs(best), abs(f_x0) if math.isfinite(f_x0) else 0.0, 1.0)
        relative_gap = max(value - best, 0.0) / denom
        buckets[solver].append((fevals, relative_gap))

    return [
        (solver, np.asarray(points, dtype=float))
        for solver, points in buckets.items()
        if points
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("out_dir")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_long_csv(args.csv_path)
    agg = aggregate(rows)

    problems = sorted({r["problem"] for r in rows})
    solvers = ordered_solvers(rows)
    solver_labels = display_solver_names(solvers)
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
            fevals_matrix[i, j] = float(
                np.median([entry["fevals"][k] for k in solved_idx])
            )
            val_matrix[i, j] = float(
                np.median([entry["best_val"][k] for k in solved_idx])
            )
            dims[i] = entry["dim"]

    plot_performance_profile(
        fevals_matrix,
        solver_labels,
        title="CUTEst performance profile",
        out_path=os.path.join(args.out_dir, "performance_profile.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'performance_profile.png')}")

    plot_data_profile(
        fevals_matrix,
        dims,
        solver_labels,
        kappa_max=data_profile_kappa_max(fevals_matrix, dims),
        title="CUTEst data profile",
        out_path=os.path.join(args.out_dir, "data_profile.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'data_profile.png')}")

    label_map = dict(zip(solvers, solver_labels))
    pareto_runs = [
        (label_map[solver], points)
        for solver, points in pareto_points_by_solver(rows, solvers)
    ]
    plot_pareto(
        pareto_runs,
        cost_label="Objective-equivalent evaluations",
        value_label="Relative gap to cell best",
        title="CUTEst accuracy-cost Pareto",
        log_x=True,
        symlog_y=True,
        y_linthresh=1e-3,
        out_path=os.path.join(args.out_dir, "pareto.png"),
    )
    print(f"Wrote {os.path.join(args.out_dir, 'pareto.png')}")


if __name__ == "__main__":
    sys.exit(main())
