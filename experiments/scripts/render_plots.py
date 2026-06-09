"""Render Dolan-Moré, Moré-Wild, and Pareto figures from a benchmark CSV.

Reads the long-form CSV emitted by `run_benchmarks.py` (one row per
(problem, solver, seed) triple), pivots into the matrix form each plot
helper expects, and writes three PNGs into the output directory."""

from __future__ import annotations

import argparse
import csv
import os
import math
import sys
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from experiments.cutest_convergence import DEFAULT_TAU, converged_mask
from experiments.plots.data_profile import plot_data_profile
from experiments.plots.pareto import plot_pareto
from experiments.plots.performance_profile import plot_performance_profile


DRIVER_ORDER = [
    "classical",
    "bayesian_mixing_sa",
    "scipy_lbfgsb",
    "scipy_de",
    "scipy_dual_annealing",
    "scipy_basinhopping",
    "scipy_direct",
    "scipy_shgo",
    "scipy_cobyqa",
    "pdfo_bobyqa",
    "cma_es",
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
    "scipy_lbfgsb": "SciPy L-BFGS-B",
    "scipy_de": "SciPy DE",
    "scipy_dual_annealing": "SciPy dual annealing",
    "scipy_basinhopping": "SciPy basin hopping",
    "scipy_direct": "SciPy DIRECT",
    "scipy_shgo": "SciPy SHGO",
    "scipy_cobyqa": "SciPy COBYQA",
    "pdfo_bobyqa": "PDFO BOBYQA",
    "cma_es": "CMA-ES",
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


DEFAULT_FIGURE_DRIVERS = [
    "classical",
    "bayesian_mixing_sa",
    "mcmc_sa_budgeted",
    "mcmc_sa_sparse_budgeted",
    "pt_sa_budgeted",
    "bgsa",
    "bgsa_auto",
]


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _status_ok(row: dict) -> bool:
    return row.get("status", "ok") == "ok"


def _split_driver_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    drivers = [part.strip() for part in raw.split(",") if part.strip()]
    return drivers or None


def ordered_solvers(rows: list[dict], requested: list[str] | None = None) -> list[str]:
    present = {str(row["driver"]) for row in rows}
    if requested is not None:
        missing = [driver for driver in requested if driver not in present]
        if missing:
            raise ValueError(f"requested driver(s) absent from CSV: {', '.join(missing)}")
        return list(dict.fromkeys(requested))
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


def _int_or_zero(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _seed_sort_key(seed: str) -> tuple[int, int | str]:
    try:
        return (0, int(seed))
    except ValueError:
        return (1, seed)


def profile_matrix_by_cell(
    rows: list[dict], solvers: list[str], tau: float = DEFAULT_TAU
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Return objective-equivalent costs by problem-seed cell and solver.

    A cell contributes its cost only when the solver converged under the
    Moré-Wild test at tolerance ``tau`` (best across the compared solvers as
    the reference), not under the earlier box-centre flag.
    """
    cells = sorted(
        {(str(row["problem"]), str(row["seed"])) for row in rows},
        key=lambda key: (key[0], _seed_sort_key(key[1])),
    )
    solver_index = {solver: idx for idx, solver in enumerate(solvers)}
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    converged = converged_mask(rows, solvers, tau)

    fevals = np.full((len(cells), len(solvers)), np.nan, dtype=float)
    dims = np.zeros(len(cells), dtype=int)
    for row in rows:
        cell = (str(row["problem"]), str(row["seed"]))
        i = cell_index[cell]
        dims[i] = _int_or_zero(row.get("dim"))
        solver = str(row["driver"])
        if solver not in solver_index:
            continue
        cost = _float_or_nan(row.get("fevals"))
        if converged.get((cell[0], cell[1], solver), False) and math.isfinite(cost):
            fevals[i, solver_index[solver]] = cost
    return cells, fevals, dims


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
    p.add_argument(
        "--drivers",
        default=",".join(DEFAULT_FIGURE_DRIVERS),
        help="Comma-separated driver subset for manuscript figures; use all for every CSV driver.",
    )
    p.add_argument("--tau", type=float, default=DEFAULT_TAU,
                   help="Moré-Wild convergence tolerance for the profiles.")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_long_csv(args.csv_path)

    requested = None if args.drivers == "all" else _split_driver_list(args.drivers)
    solvers = ordered_solvers(rows, requested=requested)
    solver_labels = display_solver_names(solvers)
    _, fevals_matrix, dims = profile_matrix_by_cell(rows, solvers, args.tau)

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
