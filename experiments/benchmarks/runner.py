"""Benchmark runner: drive every (solver, problem, seed) combination
and emit a long-form CSV consumed by the Dolan-Moré / Moré-Wild
profile and Pareto plotters.

Schema: problem, dim, solver, seed, fevals, best_val, wall_time_s, solved

solved=1 iff `best_val <= f_star + tol` for the problem's known
optimum and a per-problem tolerance (default `max(1e-6, 0.01 * |f_star|)`).
Use NaN for fevals/best_val on failed runs so the profile helpers
treat them as `+inf`.

The solver table is data-driven so additional runners can be added as
columns without changing the profile format.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np

from experiments.benchmarks.catalog import CATALOG, Problem, list_problems


@dataclass
class RunResult:
    problem: str
    dim: int
    solver: str
    seed: int
    fevals: int
    best_val: float
    wall_time_s: float
    solved: int


def _tolerance(p: Problem, atol: float = 1e-6, rtol: float = 0.01) -> float:
    return max(atol, rtol * abs(p.f_star))


def _run_anneal(p: Problem, variant: str, seed: int, n_epochs: int,
                steps_per_epoch: int, dtype: str = "float64") -> RunResult:
    t0 = time.perf_counter()
    # Use a problem-specific objective registration is not the path here;
    # we drive sa_run via direct callable rather than its registry so the
    # benchmarks stay self-contained.
    rng = np.random.default_rng(seed)

    # Mirror sa_run inline so we can use any catalog problem.
    np_dtype = np.dtype(dtype)
    cur_pos = rng.uniform(p.low, p.high, size=p.low.shape).astype(np_dtype)
    cur_val = np_dtype.type(p.fn(cur_pos))
    best_val = cur_val
    n_calls = 1

    sigma = max(0.05, 0.1 * float(np.linalg.norm(p.high - p.low)) / np.sqrt(p.dim))
    gamma = sigma
    t_init = max(1.0, abs(p.f_star) / 10.0) if p.f_star != 0 else 1.0
    q_v = 2.62

    from experiments.shared.runner import (
        cauchy_propose,
        gaussian_propose,
        log_cool,
        metropolis_accept_prob,
        reciprocal_cool,
        tsallis_cool,
        tsallis_visit_propose,
    )

    for epoch in range(n_epochs):
        if variant == "boltzmann":
            temp = log_cool(t_init, 2.0, epoch, np_dtype.type)
        elif variant == "fast":
            temp = reciprocal_cool(t_init, epoch, np_dtype.type)
        else:
            temp = tsallis_cool(t_init, q_v, epoch, np_dtype.type)

        for _ in range(steps_per_epoch):
            if variant == "boltzmann":
                proposal = gaussian_propose(rng, cur_pos, sigma, np_dtype.type)
            elif variant == "fast":
                proposal = cauchy_propose(rng, cur_pos, gamma, np_dtype.type)
            else:
                proposal = tsallis_visit_propose(rng, cur_pos, q_v, temp, np_dtype.type)

            proposal_val = np_dtype.type(p.fn(proposal))
            n_calls += 1
            delta = np_dtype.type(proposal_val - cur_val)
            prob = metropolis_accept_prob(delta, temp, np_dtype.type)
            u = np_dtype.type(rng.random())
            if u < prob:
                cur_pos = proposal
                cur_val = proposal_val
                if proposal_val < best_val:
                    best_val = proposal_val

    wall = time.perf_counter() - t0
    tol = _tolerance(p)
    solved = int(float(best_val) <= p.f_star + tol)
    return RunResult(
        problem=p.name,
        dim=p.dim,
        solver=variant,
        seed=seed,
        fevals=n_calls,
        best_val=float(best_val),
        wall_time_s=wall,
        solved=solved,
    )


def run_benchmarks(
    *,
    problems: list[str] | None = None,
    solvers: tuple[str, ...] = ("boltzmann", "fast", "gsa"),
    seeds: int = 5,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    dtype: str = "float64",
    out_path: str | None = None,
) -> list[RunResult]:
    """Run the cross-product (problem, solver, seed) and optionally write CSV."""
    problems = problems or list_problems()
    rows: list[RunResult] = []
    for name in problems:
        p = CATALOG[name]
        for solver in solvers:
            for seed in range(seeds):
                rows.append(_run_anneal(p, solver, seed, n_epochs, steps_per_epoch, dtype))
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "problem", "dim", "solver", "seed",
                "fevals", "best_val", "wall_time_s", "solved",
            ])
            w.writeheader()
            for r in rows:
                w.writerow(r.__dict__)
    return rows
