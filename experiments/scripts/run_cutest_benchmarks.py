"""Run the three SA drivers (classical, dense MCMC-SA, sparse MCMC-SA)
across the 12-problem CUTEst manifest and emit the long-form CSV
consumed by the Dolan-Moré / Pareto plotters.

Schema: problem, dim, driver, seed, fevals, best_val, wall_time_s,
solved (1 if best_val within 5% of f(x0) -- a weak surrogate for
"reached a low region", since most CUTEst problems do not ship a
known global minimum we can test against directly).

This is the headline figure-feed for IISE Section 6 (benchmark
comparison) and Section 7 (Pareto front)."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

from experiments.benchmarks.cutest_runner import load_default_manifest
from experiments.shared.runner import (
    gelman_rubin_max,
    gaussian_propose,
    log_cool,
    metropolis_accept_prob,
)

TARGET_ACCEPT_RATE = 0.234
TARGET_SWAP_RATE = 0.234
FINITE_DIFFERENCE_GRAD_STEP = 1e-6
COVERED_BOUND_POLISH_BUDGET_DIVISOR = 2
QMC_DIFFERENTIAL_MUTATION_WEIGHT = 0.5
QMC_DIFFERENTIAL_CROSSOVER_RATE = 0.9


def _low_discrepancy_starts(
    low, high, n_points, seed, design_low=None, design_high=None
):
    try:
        from experiments.scripts.demo_bgsa import low_discrepancy_init
    except Exception:
        from demo_bgsa import low_discrepancy_init

    return low_discrepancy_init(
        np.random.default_rng(seed),
        int(n_points),
        np.asarray(low, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        design_low=design_low,
        design_high=design_high,
    )


def _design_bounds(prob):
    low = np.asarray(getattr(prob, "design_low", prob.low), dtype=np.float64)
    high = np.asarray(getattr(prob, "design_high", prob.high), dtype=np.float64)
    return low, high


_CUTEST_FIELDNAMES = [
    "problem", "dim", "driver", "seed", "fevals",
    "best_val", "wall_time_s", "f_x0", "solved",
]


def _write_cutest_rows(path, rows):
    """Write all rows to `path` (checkpoint-safe; called after each problem)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CUTEST_FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def _straggler_indices(chain_pos, top_k):
    if top_k <= 0 or top_k >= len(chain_pos):
        return list(range(len(chain_pos)))
    pooled = np.mean(chain_pos, axis=0)
    dists = [(i, np.linalg.norm(p - pooled)) for i, p in enumerate(chain_pos)]
    dists.sort(key=lambda x: -x[1])
    return [i for i, _ in dists[:top_k]]


def _step_chain(prob, rng, cur_pos, cur_val, best_val, temp, sigma):
    proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
    proposal = np.clip(proposal, prob.low, prob.high)
    proposal_val = prob.fn(proposal)
    delta = proposal_val - cur_val
    p = metropolis_accept_prob(delta, temp, np.float64)
    if rng.random() < p:
        cur_pos = proposal
        cur_val = proposal_val
        if proposal_val < best_val:
            best_val = proposal_val
    return cur_pos, cur_val, best_val


def _step_chain_observed(prob, rng, cur_pos, cur_val, best_val, temp, sigma):
    proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
    proposal = np.clip(proposal, prob.low, prob.high)
    proposal_val = prob.fn(proposal)
    delta = proposal_val - cur_val
    p = metropolis_accept_prob(delta, temp, np.float64)
    accepted = bool(rng.random() < p)
    improved = False
    if accepted:
        cur_pos = proposal
        cur_val = proposal_val
        if proposal_val < best_val:
            best_val = proposal_val
            improved = True
    return cur_pos, cur_val, best_val, accepted, improved


def classical_sa(prob, seed, n_epochs, k_fixed, sigma=None, t_init=5.0):
    rng = np.random.default_rng(seed)
    if sigma is None:
        sigma = _auto_sigma(prob)
    design_low, design_high = _design_bounds(prob)
    cur_pos = _low_discrepancy_starts(
        prob.low, prob.high, 1, seed, design_low, design_high
    )[0]
    cur_val = prob.fn(cur_pos)
    best_val = cur_val
    n_calls = 1
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        for _ in range(k_fixed):
            proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
            proposal = np.clip(proposal, prob.low, prob.high)
            proposal_val = prob.fn(proposal)
            n_calls += 1
            delta = proposal_val - cur_val
            p = metropolis_accept_prob(delta, temp, np.float64)
            if rng.random() < p:
                cur_pos = proposal
                cur_val = proposal_val
                if proposal_val < best_val:
                    best_val = proposal_val
    return best_val, n_calls


def mcmc_sa(
    prob,
    seed,
    n_epochs,
    n_chains,
    k_min,
    k_check,
    k_max,
    rhat_threshold,
    sigma=None,
    t_init=5.0,
    sparse=False,
    straggler_top_k=0,
):
    if sigma is None:
        sigma = _auto_sigma(prob)
    design_low, design_high = _design_bounds(prob)
    starts = _low_discrepancy_starts(
        prob.low, prob.high, n_chains, seed, design_low, design_high
    )
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [prob.fn(p) for p in chain_pos]
    chain_best_val = list(chain_val)
    n_calls = n_chains
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        traces = [[] for _ in range(n_chains)]
        for _ in range(k_min):
            for c in range(n_chains):
                proposal = gaussian_propose(rngs[c], chain_pos[c], sigma, np.float64)
                proposal = np.clip(proposal, prob.low, prob.high)
                proposal_val = prob.fn(proposal)
                n_calls += 1
                delta = proposal_val - chain_val[c]
                p = metropolis_accept_prob(delta, temp, np.float64)
                if rngs[c].random() < p:
                    chain_pos[c] = proposal
                    chain_val[c] = proposal_val
                    if proposal_val < chain_best_val[c]:
                        chain_best_val[c] = proposal_val
                traces[c].append(chain_pos[c].copy())
        total_steps = k_min
        rhat = gelman_rubin_max(traces)
        while rhat > rhat_threshold and total_steps < k_max:
            if sparse and 0 < straggler_top_k < n_chains:
                active = _straggler_indices(chain_pos, straggler_top_k)
            else:
                active = list(range(n_chains))
            batch = min(k_check, k_max - total_steps)
            for _ in range(batch):
                for c in active:
                    chain_pos[c], chain_val[c], chain_best_val[c] = _step_chain(
                        prob,
                        rngs[c],
                        chain_pos[c],
                        chain_val[c],
                        chain_best_val[c],
                        temp,
                        sigma,
                    )
                    n_calls += 1
                    traces[c].append(chain_pos[c].copy())
                for c in range(n_chains):
                    if c not in active:
                        traces[c].append(chain_pos[c].copy())
            total_steps += batch
            rhat = gelman_rubin_max(traces)
    return min(chain_best_val), n_calls


def _append_budgeted_round(traces, chain_pos, step_active):
    step_active = set(step_active)
    for c in range(len(chain_pos)):
        if c not in step_active:
            traces[c].append(chain_pos[c].copy())


def _budgeted_step_round(
    prob,
    rngs,
    chain_pos,
    chain_val,
    chain_best_val,
    traces,
    active,
    remaining,
    temp,
    sigma,
):
    stepped = []
    for c in active[:remaining]:
        chain_pos[c], chain_val[c], chain_best_val[c] = _step_chain(
            prob, rngs[c], chain_pos[c], chain_val[c], chain_best_val[c], temp, sigma
        )
        traces[c].append(chain_pos[c].copy())
        stepped.append(c)
    _append_budgeted_round(traces, chain_pos, stepped)
    return len(stepped)


def mcmc_sa_budgeted(
    prob,
    seed,
    n_epochs,
    n_chains,
    epoch_budget,
    k_min=30,
    k_check=20,
    rhat_threshold=1.2,
    sigma=None,
    t_init=5.0,
    sparse=False,
    straggler_top_k=0,
):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")

    if sigma is None:
        sigma = _auto_sigma(prob)
    design_low, design_high = _design_bounds(prob)
    starts = _low_discrepancy_starts(
        prob.low, prob.high, n_chains, seed, design_low, design_high
    )
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [prob.fn(p) for p in chain_pos]
    chain_best_val = list(chain_val)
    n_calls = n_chains
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        traces = [[] for _ in range(n_chains)]
        epoch_calls = 0

        min_rounds = min(k_min, epoch_budget // n_chains)
        for _ in range(min_rounds):
            epoch_calls += _budgeted_step_round(
                prob,
                rngs,
                chain_pos,
                chain_val,
                chain_best_val,
                traces,
                list(range(n_chains)),
                epoch_budget - epoch_calls,
                temp,
                sigma,
            )
        while epoch_calls < epoch_budget:
            rhat = gelman_rubin_max(traces)
            if rhat <= rhat_threshold:
                break
            if sparse and 0 < straggler_top_k < n_chains:
                active = _straggler_indices(chain_pos, straggler_top_k)
            else:
                active = list(range(n_chains))
            for _ in range(k_check):
                if epoch_calls >= epoch_budget:
                    break
                epoch_calls += _budgeted_step_round(
                    prob,
                    rngs,
                    chain_pos,
                    chain_val,
                    chain_best_val,
                    traces,
                    active,
                    epoch_budget - epoch_calls,
                    temp,
                    sigma,
                )
        n_calls += epoch_calls
    return min(chain_best_val), n_calls


def _geometric_ladder(t_cold, t_hot, n_chains):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if n_chains == 1:
        return np.array([t_cold], dtype=np.float64)
    if t_cold <= 0 or t_hot <= t_cold:
        raise ValueError("temperature ladder requires 0 < t_cold < t_hot")
    ratios = np.linspace(0.0, 1.0, n_chains)
    return t_cold * (t_hot / t_cold) ** ratios


def _pt_swap_accept_prob(f_i, t_i, f_j, t_j):
    log_alpha = (1.0 / t_i - 1.0 / t_j) * (f_i - f_j)
    if log_alpha >= 0.0:
        return 1.0
    return float(np.exp(max(log_alpha, -745.0)))


def pt_sa_budgeted(
    prob,
    seed,
    n_epochs,
    n_chains,
    epoch_budget,
    swap_period=5,
    sigma=None,
    t_init=5.0,
    t_hot_multiplier=4.0,
    return_diagnostics=False,
):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")
    if swap_period < 1:
        raise ValueError("swap_period must be positive")
    if t_hot_multiplier <= 1.0:
        raise ValueError("t_hot_multiplier must exceed one")

    if sigma is None:
        sigma = _auto_sigma(prob)
    design_low, design_high = _design_bounds(prob)
    starts = _low_discrepancy_starts(
        prob.low, prob.high, n_chains, seed, design_low, design_high
    )
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    swap_rng = np.random.default_rng(seed + n_chains + 1)
    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [prob.fn(p) for p in chain_pos]
    chain_best_val = list(chain_val)
    best_val = min(chain_best_val)
    n_calls = n_chains
    swap_attempts = 0
    swap_accepts = 0

    for epoch in range(n_epochs):
        cold_temp = log_cool(t_init, 2.0, epoch, np.float64)
        temps = _geometric_ladder(cold_temp, cold_temp * t_hot_multiplier, n_chains)
        epoch_calls = 0
        rounds = 0
        while epoch_calls < epoch_budget:
            for c in range(n_chains):
                if epoch_calls >= epoch_budget:
                    break
                chain_pos[c], chain_val[c], chain_best_val[c] = _step_chain(
                    prob,
                    rngs[c],
                    chain_pos[c],
                    chain_val[c],
                    chain_best_val[c],
                    temps[c],
                    sigma,
                )
                n_calls += 1
                epoch_calls += 1
                if chain_best_val[c] < best_val:
                    best_val = chain_best_val[c]
            rounds += 1
            if n_chains > 1 and rounds % swap_period == 0:
                i = int(swap_rng.integers(0, n_chains - 1))
                alpha = _pt_swap_accept_prob(
                    chain_val[i], temps[i], chain_val[i + 1], temps[i + 1]
                )
                swap_attempts += 1
                if swap_rng.random() < alpha:
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts += 1

    if return_diagnostics:
        return (
            best_val,
            n_calls,
            {
                "swap_attempts": swap_attempts,
                "swap_accepts": swap_accepts,
            },
        )
    return best_val, n_calls


def _auto_chain_count(prob, max_fevals):
    dim = int(getattr(prob, "dim", len(prob.low)))
    budget_limited = max(2, min(4, max_fevals // 64))
    dim_limited = max(2, min(4, int(np.ceil(np.sqrt(max(dim, 1))))))
    return max(1, min(budget_limited, dim_limited, max_fevals))


def _auto_sigma(prob):
    """Per-coordinate Gaussian proposal scale for random-walk Metropolis.

    The optimal RWM scale falls as 1/sqrt(dim): with a fixed per-coordinate
    sigma the total step magnitude grows as sigma*sqrt(dim), so acceptance
    collapses to zero in high dimension and the chain freezes. Scaling the
    per-coordinate sigma by the box diagonal divided by dim keeps the total
    step magnitude at O(box) across dimensions.
    """
    low = np.asarray(prob.low, dtype=np.float64)
    high = np.asarray(prob.high, dtype=np.float64)
    dim = max(len(low), 1)
    diag = float(np.linalg.norm(high - low))
    if not np.isfinite(diag) or diag <= 0.0:
        diag = float(np.sqrt(dim))
    per_coord_width = diag / np.sqrt(dim)
    sigma = 0.25 * diag / dim  # = 0.25 * per_coord_width / sqrt(dim)
    return float(np.clip(sigma, 1e-6, per_coord_width))


def _auto_initial_temperature(chain_val):
    vals = np.asarray(chain_val, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return 1.0
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.median(np.abs(finite)))
    return float(np.clip(scale, 1e-6, 5.0))


class _BudgetExhausted(RuntimeError):
    pass


class _BudgetedObjective:
    def __init__(self, prob, max_fevals):
        if max_fevals < 1:
            raise ValueError("max_fevals must be positive")
        self.prob = prob
        self.max_fevals = int(max_fevals)
        self.n_calls = 0
        self.best_val = float("inf")
        self.best_pos = None

    def __call__(self, x):
        if self.n_calls >= self.max_fevals:
            raise _BudgetExhausted
        x = np.clip(
            np.asarray(x, dtype=np.float64).reshape(-1),
            self.prob.low,
            self.prob.high,
        )
        value = float(self.prob.fn(x))
        self.n_calls += 1
        if np.isfinite(value) and value < self.best_val:
            self.best_val = value
            self.best_pos = x.copy()
        return value

    def result(self, fallback=float("inf")):
        if np.isfinite(self.best_val):
            return self.best_val, self.n_calls
        return float(fallback), self.n_calls


def _scipy_bounds(prob):
    design_low, design_high = _design_bounds(prob)
    return list(zip(design_low.tolist(), design_high.tolist()))


def _scipy_start(prob, seed):
    design_low, design_high = _design_bounds(prob)
    return _low_discrepancy_starts(
        prob.low, prob.high, 1, seed, design_low, design_high
    )[0]


def scipy_lbfgsb(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    bounds = _scipy_bounds(prob)
    x0 = _scipy_start(prob, seed)
    fallback = float("inf")
    try:
        res = optimize.minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": int(max_fevals), "maxiter": int(max_fevals)},
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_de(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    dim = max(int(getattr(prob, "dim", len(prob.low))), 1)
    popsize = max(3, min(15, int(max_fevals) // max(2 * dim, 1)))
    generation = max(popsize * dim, 1)
    maxiter = max(1, int(max_fevals) // generation)
    fallback = float("inf")
    try:
        res = optimize.differential_evolution(
            obj,
            _scipy_bounds(prob),
            maxiter=maxiter,
            popsize=popsize,
            polish=False,
            init="sobol",
            tol=0.0,
            atol=0.0,
            rng=seed,
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_dual_annealing(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    fallback = float("inf")
    try:
        res = optimize.dual_annealing(
            obj,
            _scipy_bounds(prob),
            maxfun=int(max_fevals),
            rng=seed,
            x0=_scipy_start(prob, seed),
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_basinhopping(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    dim = max(int(getattr(prob, "dim", len(prob.low))), 1)
    local_budget = max(1, int(max_fevals) // max(4, dim))
    niter = max(1, int(max_fevals) // max(local_budget, 1) - 1)
    fallback = float("inf")
    try:
        res = optimize.basinhopping(
            obj,
            _scipy_start(prob, seed),
            niter=niter,
            stepsize=_auto_sigma(prob),
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": _scipy_bounds(prob),
                "options": {"maxfun": local_budget, "maxiter": local_budget},
            },
            rng=seed,
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_direct(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    fallback = float("inf")
    try:
        res = optimize.direct(
            obj,
            _scipy_bounds(prob),
            maxfun=int(max_fevals),
            maxiter=int(max_fevals),
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_shgo(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    dim = max(int(getattr(prob, "dim", len(prob.low))), 1)
    n = max(dim + 1, min(int(max_fevals), 2 * dim + 1))
    fallback = float("inf")
    try:
        res = optimize.shgo(
            obj,
            _scipy_bounds(prob),
            n=n,
            iters=max(1, int(max_fevals) // max(n, 1)),
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": _scipy_bounds(prob),
                "options": {"maxfun": int(max_fevals), "maxiter": int(max_fevals)},
            },
            sampling_method="sobol",
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def scipy_cobyqa(prob, seed, max_fevals):
    from scipy import optimize

    obj = _BudgetedObjective(prob, max_fevals)
    fallback = float("inf")
    try:
        res = optimize.minimize(
            obj,
            _scipy_start(prob, seed),
            method="COBYQA",
            bounds=_scipy_bounds(prob),
            options={
                "maxfev": int(max_fevals),
                "maxiter": int(max_fevals),
                "scale": True,
            },
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def pdfo_bobyqa(prob, seed, max_fevals):
    import pdfo

    obj = _BudgetedObjective(prob, max_fevals)
    fallback = float("inf")
    try:
        res = pdfo.pdfo(
            obj,
            _scipy_start(prob, seed),
            method="bobyqa",
            bounds=np.asarray(_scipy_bounds(prob), dtype=np.float64),
            options={
                "maxfev": int(max_fevals),
                "quiet": True,
                "scale": True,
            },
        )
        fallback = float(getattr(res, "fun", fallback))
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


def pdfo_bobyqa_available():
    try:
        import pdfo.gethuge  # noqa: F401
    except Exception:
        return False
    return True


def cma_es(prob, seed, max_fevals):
    import cma

    obj = _BudgetedObjective(prob, max_fevals)
    low, high = _design_bounds(prob)
    fallback = float("inf")
    try:
        _xbest, _es = cma.fmin2(
            obj,
            _scipy_start(prob, seed),
            _auto_sigma(prob),
            options={
                "bounds": [low.tolist(), high.tolist()],
                "maxfevals": int(max_fevals),
                "seed": int(seed),
                "verbose": -9,
            },
        )
        fallback = obj.best_val
    except _BudgetExhausted:
        pass
    return obj.result(fallback)


SCIPY_DRIVERS = {
    "scipy_lbfgsb": scipy_lbfgsb,
    "scipy_de": scipy_de,
    "scipy_dual_annealing": scipy_dual_annealing,
    "scipy_basinhopping": scipy_basinhopping,
    "scipy_direct": scipy_direct,
    "scipy_shgo": scipy_shgo,
    "scipy_cobyqa": scipy_cobyqa,
    "pdfo_bobyqa": pdfo_bobyqa,
    "cma_es": cma_es,
}


def bayesian_mixing_sa(prob, seed, max_fevals, return_diagnostics=False):
    if max_fevals < 1:
        raise ValueError("max_fevals must be positive")

    n_chains = _auto_chain_count(prob, max_fevals)
    design_low, design_high = _design_bounds(prob)
    starts = _low_discrepancy_starts(
        prob.low, prob.high, n_chains, seed, design_low, design_high
    )
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    controller_rng = np.random.default_rng(seed + 10_007)
    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [prob.fn(p) for p in chain_pos]
    chain_best_val = list(chain_val)
    n_calls = n_chains
    best_val = min(chain_best_val)
    if n_calls >= max_fevals:
        if return_diagnostics:
            return (
                best_val,
                n_calls,
                {
                    "n_chains": n_chains,
                    "swap_attempts": 0,
                    "swap_accepts": 0,
                    "posterior_accept_mean": 0.5,
                    "posterior_improve_mean": 0.5,
                    "proposal_counts": [0] * n_chains,
                },
            )
        return best_val, n_calls

    base_sigma = _auto_sigma(prob)
    log_sigma = np.full(n_chains, np.log(base_sigma), dtype=np.float64)
    t_init = _auto_initial_temperature(chain_val)
    improve_alpha = np.ones(n_chains, dtype=np.float64)
    improve_beta = np.ones(n_chains, dtype=np.float64)
    improve_alpha[0] = 4.0
    if n_chains > 1:
        improve_beta[1:] = 4.0
    accept_alpha = np.ones(n_chains, dtype=np.float64)
    accept_beta = np.ones(n_chains, dtype=np.float64)
    swap_alpha = np.ones(max(n_chains - 1, 1), dtype=np.float64)
    swap_beta = np.ones(max(n_chains - 1, 1), dtype=np.float64)
    ladder_log_span = np.log(4.0)
    swap_attempts = 0
    swap_accepts = 0
    proposals_since_swap = 0
    proposal_counts = np.zeros(n_chains, dtype=np.int64)
    proposal_budget = max_fevals - n_calls
    incumbent_chain = 0

    while n_calls < max_fevals:
        progress = (n_calls - n_chains) / max(proposal_budget, 1)
        cold_temp = max(t_init / np.log(2.0 + 20.0 * progress), 1e-12)
        temps = _geometric_ladder(
            cold_temp,
            cold_temp * float(np.exp(ladder_log_span)),
            n_chains,
        )
        best_chain = incumbent_chain
        utility = controller_rng.beta(improve_alpha, improve_beta)
        challenger_idx = int(np.argmax(utility))
        if (
            challenger_idx != best_chain
            and utility[challenger_idx] > utility[best_chain] + 0.05
        ):
            chain_idx = challenger_idx
        else:
            chain_idx = best_chain
        sigma = float(np.exp(log_sigma[chain_idx]))
        temp = cold_temp if chain_idx == best_chain else temps[chain_idx]
        (
            chain_pos[chain_idx],
            chain_val[chain_idx],
            chain_best_val[chain_idx],
            accepted,
            _improved,
        ) = _step_chain_observed(
            prob,
            rngs[chain_idx],
            chain_pos[chain_idx],
            chain_val[chain_idx],
            chain_best_val[chain_idx],
            temp,
            sigma,
        )
        n_calls += 1
        proposals_since_swap += 1
        proposal_counts[chain_idx] += 1

        accept_alpha[chain_idx] += 1.0 if accepted else 0.0
        accept_beta[chain_idx] += 0.0 if accepted else 1.0
        global_improved = chain_best_val[chain_idx] < best_val
        if global_improved:
            improve_alpha[chain_idx] += 1.0
        else:
            improve_beta[chain_idx] += 1.0
        accept_mean = accept_alpha[chain_idx] / (
            accept_alpha[chain_idx] + accept_beta[chain_idx]
        )
        log_sigma[chain_idx] += 0.05 * (accept_mean - TARGET_ACCEPT_RATE)
        log_sigma[chain_idx] = np.clip(
            log_sigma[chain_idx],
            np.log(base_sigma / 32.0),
            np.log(base_sigma * 32.0),
        )
        if global_improved:
            best_val = chain_best_val[chain_idx]
            incumbent_chain = chain_idx

        if n_chains > 1 and proposals_since_swap >= n_chains:
            pair_scores = controller_rng.beta(swap_alpha, swap_beta)
            pair_idx = int(np.argmin(pair_scores))
            alpha = _pt_swap_accept_prob(
                chain_val[pair_idx],
                temps[pair_idx],
                chain_val[pair_idx + 1],
                temps[pair_idx + 1],
            )
            swap_attempts += 1
            accepted_swap = bool(controller_rng.random() < alpha)
            if accepted_swap:
                chain_pos[pair_idx], chain_pos[pair_idx + 1] = (
                    chain_pos[pair_idx + 1],
                    chain_pos[pair_idx],
                )
                chain_val[pair_idx], chain_val[pair_idx + 1] = (
                    chain_val[pair_idx + 1],
                    chain_val[pair_idx],
                )
                swap_accepts += 1
                swap_alpha[pair_idx] += 1.0
            else:
                swap_beta[pair_idx] += 1.0
            swap_mean = swap_alpha[pair_idx] / (
                swap_alpha[pair_idx] + swap_beta[pair_idx]
            )
            ladder_log_span += 0.05 * (swap_mean - TARGET_SWAP_RATE)
            ladder_log_span = float(np.clip(ladder_log_span, np.log(1.5), np.log(16.0)))
            proposals_since_swap = 0

    if return_diagnostics:
        return (
            best_val,
            n_calls,
            {
                "n_chains": n_chains,
                "swap_attempts": swap_attempts,
                "swap_accepts": swap_accepts,
                "posterior_accept_mean": float(
                    np.mean(accept_alpha / (accept_alpha + accept_beta))
                ),
                "posterior_improve_mean": float(
                    np.mean(improve_alpha / (improve_alpha + improve_beta))
                ),
                "proposal_counts": proposal_counts.tolist(),
            },
        )
    return best_val, n_calls


DRIVERS = [
    "classical",
    "mcmc_sa",
    "mcmc_sa_sparse",
    "mcmc_sa_budgeted",
    "mcmc_sa_sparse_budgeted",
    "pt_sa_budgeted",
    "bayesian_mixing_sa",
    "additive_indep",
    "scipy_lbfgsb",
    "scipy_de",
    "scipy_dual_annealing",
    "scipy_basinhopping",
    "scipy_direct",
    "scipy_shgo",
    "scipy_cobyqa",
    "pdfo_bobyqa",
    "cma_es",
    "bgsa",
    "bgsa_metad",
    "bgsa_pt_metad",
    "bgsa_auto",
]


def _rust_hmc_omelyan_grad_calls(n_trajectories, l_steps):
    """Gradient calls made by Rust Omelyan HMC trajectories."""
    return int(n_trajectories) * (1 + 2 * int(l_steps))


def _rust_hmc_native_grad_work_units(n_trajectories, l_steps, total_accepted=0):
    """Objective-equivalent work units for Rust Omelyan HMC with native gradients."""
    n_trajectories = int(n_trajectories)
    return (
        1
        + n_trajectories
        + int(total_accepted)
        + _rust_hmc_omelyan_grad_calls(n_trajectories, l_steps)
    )


def _rust_hmc_fd_work_units(dim, n_trajectories, l_steps, total_accepted=0):
    """Objective-call work units for Rust Omelyan HMC with finite differences."""
    n_trajectories = int(n_trajectories)
    grad_cost = int(dim) + 1
    return (
        1
        + n_trajectories
        + int(total_accepted)
        + grad_cost * _rust_hmc_omelyan_grad_calls(n_trajectories, l_steps)
    )


def _rust_hmc_max_work_units_per_trajectory(dim, l_steps, grad_kind):
    grad_cost = 1 if grad_kind == "native" else int(dim) + 1
    return 2 + grad_cost * (1 + 2 * int(l_steps))


def _rust_hmc_steps_per_epoch_budget(epoch_budget, dim, l_steps, grad_kind):
    if epoch_budget <= 0:
        raise ValueError("epoch_budget must be positive")
    per_trajectory = _rust_hmc_max_work_units_per_trajectory(dim, l_steps, grad_kind)
    return max(1, int(epoch_budget) // per_trajectory)


def _pt_hmc_inner_steps_per_epoch_budget(epoch_budget, n_chains, dim, l_steps, grad_kind):
    if epoch_budget <= 0:
        raise ValueError("epoch_budget must be positive")
    per_inner = max(1, int(n_chains)) * _rust_hmc_max_work_units_per_trajectory(
        dim, l_steps, grad_kind
    )
    return max(1, int(epoch_budget) // per_inner)


def _ensemble_candidate_epoch_budget(epoch_budget, n_candidates):
    if epoch_budget <= 0:
        raise ValueError("epoch_budget must be positive")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    return max(1, int(epoch_budget) // int(n_candidates))


def _bgsa_pilot_budget(n_epochs, k_per_epoch, n_chains):
    pilot_steps = max(5, min(20, int(k_per_epoch) // 10))
    return {
        "n_pilot": max(4, min(8, int(n_epochs) // 2)),
        "pilot_steps": pilot_steps,
        "n_rw_pilot": max(4, min(6, int(n_chains))),
        "rw_steps": pilot_steps,
        "n_scout": 4,
    }


def _cutest_gradient(prob):
    native_grad = getattr(prob, "grad", None)
    if callable(native_grad):

        def _native_grad(x):
            return np.asarray(native_grad(x), dtype=np.float64).reshape(-1)

        return _native_grad, "native"

    return _finite_difference_gradient(prob), "finite-difference"


def _finite_difference_gradient(prob):
    def _fd_grad(x):
        x = np.asarray(x, dtype=np.float64)
        f0 = prob.fn(x)
        g = np.zeros_like(x)
        for i in range(len(x)):
            x1 = x.copy()
            x1[i] += FINITE_DIFFERENCE_GRAD_STEP
            g[i] = (prob.fn(x1) - f0) / FINITE_DIFFERENCE_GRAD_STEP
        return g

    return _fd_grad


def _run_cutest_rust_hmc(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_epochs,
    epoch_budget,
    t_map,
    e_map,
    L_map,
    q_map,
    best_pilot_pos,
):
    bounds_dim = int(np.asarray(prob.low).size)
    l_steps = max(1, int(L_map))
    q_hmc = min(float(q_map), float(np.nextafter(1.0 + 2.0 / bounds_dim, 1.0)))
    hmc_steps_per_epoch = _rust_hmc_steps_per_epoch_budget(
        epoch_budget, bounds_dim, l_steps, grad_kind
    )
    x0 = _hmc_initial_position(best_pilot_pos, bounds_dim)
    history = anneal_module.run_hmc(
        prob.fn,
        grad_fn,
        prob.low.astype(np.float64),
        prob.high.astype(np.float64),
        t_init=float(t_map),
        epsilon=float(e_map),
        l_steps=l_steps,
        q=q_hmc,
        n_epochs=int(n_epochs),
        steps_per_epoch=hmc_steps_per_epoch,
        seed=int(seed),
        x0=x0,
    )
    n_trajectories = int(n_epochs) * hmc_steps_per_epoch
    total_accepted = int(getattr(history, "total_accepted", 0))
    if grad_kind == "native":
        work_units = _rust_hmc_native_grad_work_units(
            n_trajectories, l_steps, total_accepted=total_accepted
        )
    else:
        work_units = _rust_hmc_fd_work_units(
            bounds_dim, n_trajectories, l_steps, total_accepted=total_accepted
        )
    return float(history.best_val), work_units


def _polish_work_units(dim, grad_kind, n_evals, n_grads):
    grad_cost = 1 if grad_kind == "native" else int(dim) + 1
    return int(n_evals) + grad_cost * int(n_grads)


def _run_cutest_rust_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    best_pilot_pos,
    max_fevals,
):
    bounds_dim = int(np.asarray(prob.low).size)
    x0 = _hmc_initial_position(best_pilot_pos, bounds_dim)
    if x0 is None or max_fevals < 1:
        return float("inf"), 0
    design_low, design_high = _design_bounds(prob)
    result = anneal_module.polish(
        prob.fn,
        grad_fn,
        design_low,
        design_high,
        x0,
        max_fevals=int(max_fevals),
    )
    work_units = _polish_work_units(
        bounds_dim,
        grad_kind,
        result.get("n_evals", 0),
        result.get("n_grads", 0),
    )
    return float(result["best_val"]), work_units


def _polish_values_agree_to_roundoff(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False
    scale = max(1.0, float(np.max(np.abs(arr))))
    tolerance = np.sqrt(np.finfo(np.float64).eps) * scale
    return float(np.max(arr) - np.min(arr)) <= tolerance


def _polish_best_dominates_sample(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return False
    best = float(np.min(arr))
    median = float(np.median(arr))
    if best <= 0.0:
        scale = max(1.0, abs(best), abs(median))
        return median - best >= np.sqrt(np.finfo(np.float64).eps) * scale
    return best * np.sqrt(float(arr.size)) <= median


def _polish_bulk_dominates_worst_tail(values):
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size < 3 or not np.all(np.isfinite(arr)):
        return False
    bulk_worst = float(arr[-2])
    tail = float(arr[-1])
    if bulk_worst <= 0.0:
        scale = max(1.0, abs(bulk_worst), abs(tail))
        return tail - bulk_worst >= np.sqrt(np.finfo(np.float64).eps) * scale
    return bulk_worst * np.sqrt(float(arr.size)) <= tail


def _run_cutest_multistart_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_starts,
    max_fevals_per_start,
):
    if not hasattr(anneal_module, "polish"):
        return None
    if n_starts < 1 or max_fevals_per_start < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    try:
        starts = _low_discrepancy_starts(
            prob.low,
            prob.high,
            n_starts,
            seed,
            design_low,
            design_high,
        )
    except Exception:
        return None
    outcomes = []
    total_work = 0
    for start in starts:
        best_val, work_units = _run_cutest_rust_polish(
            anneal_module,
            prob,
            grad_fn,
            grad_kind,
            start,
            max_fevals_per_start,
        )
        outcomes.append(float(best_val))
        total_work += int(work_units)
    if not outcomes:
        return None
    return min(outcomes), total_work, outcomes


def _run_cutest_best_start_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_starts,
    max_fevals,
):
    if not hasattr(anneal_module, "polish"):
        return None
    if n_starts < 1 or max_fevals < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    try:
        starts = _low_discrepancy_starts(
            prob.low,
            prob.high,
            n_starts,
            seed,
            design_low,
            design_high,
        )
    except Exception:
        return None
    screened = []
    for start in starts:
        value = float(prob.fn(start))
        if np.isfinite(value):
            screened.append((value, start))
    if not screened:
        return None
    screened.sort(key=lambda item: item[0])
    if screened[0][0] > 0.0:
        cutoff = screened[0][0] * np.sqrt(float(len(starts)))
        candidates = [item for item in screened if item[0] <= cutoff]
    else:
        candidates = [screened[0]]
    best_val = float("inf")
    total_work = len(starts)
    for _value, start in candidates:
        candidate_val, work_units = _run_cutest_rust_polish(
            anneal_module,
            prob,
            grad_fn,
            grad_kind,
            start,
            max_fevals,
        )
        total_work += int(work_units)
        if candidate_val < best_val:
            best_val = candidate_val
    return best_val, total_work


def _run_cutest_raw_best_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_starts,
    max_fevals,
):
    if not hasattr(anneal_module, "polish"):
        return None
    if n_starts < 1 or max_fevals < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    try:
        starts = _low_discrepancy_starts(
            prob.low,
            prob.high,
            n_starts,
            seed,
            design_low,
            design_high,
        )
    except Exception:
        return None
    screened = []
    for start in starts:
        value = float(prob.fn(start))
        if np.isfinite(value):
            screened.append((value, start))
    if not screened:
        return None
    screened.sort(key=lambda item: item[0])
    best_val, work_units = _run_cutest_rust_polish(
        anneal_module,
        prob,
        grad_fn,
        grad_kind,
        screened[0][1],
        max_fevals,
    )
    return best_val, len(starts) + int(work_units)


def _run_cutest_qmc_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_starts,
    max_fevals_per_start,
    top_k=0,
):
    if not hasattr(anneal_module, "qmc_polish"):
        return None
    if n_starts < 1 or max_fevals_per_start < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    result = anneal_module.qmc_polish(
        prob.fn,
        grad_fn,
        design_low,
        design_high,
        int(n_starts),
        int(max_fevals_per_start),
        seed=int(seed),
        top_k=int(top_k),
    )
    work_units = _polish_work_units(
        int(np.asarray(prob.low).size),
        grad_kind,
        result.get("n_evals", 0),
        result.get("n_grads", 0),
    )
    return float(result["best_val"]), work_units


def _has_declared_cutest_bounds(prob):
    return bool(getattr(prob, "has_cutest_bounds", False))


def _has_finite_design_box(prob):
    low, high = _design_bounds(prob)
    return (
        low.shape == high.shape
        and low.size > 0
        and np.all(np.isfinite(low))
        and np.all(np.isfinite(high))
        and np.all(high > low)
    )


def _bounded_polish_dimension_is_covered(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(dim) <= int(n_chains) * int(n_chains)


def _bounded_polish_top_k(n_chains):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return max(1, int(np.ceil(np.sqrt(float(n_chains)))))


def _native_qmc_dense_dimension_is_covered(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(dim) <= int(n_chains) * _bounded_polish_top_k(n_chains)


def _native_qmc_polish_start_count(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if _bounded_polish_dimension_is_covered(dim, n_chains):
        return int(n_chains) + int(dim)
    return int(n_chains) + int(np.ceil(np.sqrt(float(dim))))


def _native_qmc_box_start_count(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(dim) * int(n_chains)


def _native_qmc_box_top_k(n_chains):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(n_chains)


def _cutest_objective_degree(prob):
    raw = getattr(prob, "objective_degree", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _native_qmc_middle_bound_supported(prob):
    degree = _cutest_objective_degree(prob)
    return degree is not None and degree > 1


def _native_qmc_box_stage_specs(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    dim = int(dim)
    n_chains = int(n_chains)
    coverage = [n_chains, 2 * n_chains]
    if dim <= n_chains * n_chains:
        coverage.append(n_chains * n_chains)
    seen = set()
    specs = []
    for multiplier in coverage:
        n_starts = dim * multiplier
        top_values = [n_chains]
        if multiplier > n_chains:
            top_values.append(0)
        if multiplier == n_chains * n_chains:
            top_values.append(dim)
        for top_k in top_values:
            key = (n_starts, top_k)
            if key in seen:
                continue
            seen.add(key)
            specs.append((n_starts, top_k))
    return tuple(specs)


def _native_qmc_polish_budget(epoch_budget, dim, n_starts):
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    return int(epoch_budget) + int(dim) * int(n_starts)


def _covered_local_polish_budget(epoch_budget, n_chains):
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(epoch_budget) * int(n_chains)


def _covered_bound_polish_budget(epoch_budget, dim, n_chains):
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    finite_difference_sweep = int(dim) + 1
    local_budget = int(epoch_budget) // COVERED_BOUND_POLISH_BUDGET_DIVISOR
    return max(1, local_budget - finite_difference_sweep)


def _hmc_initial_position(best_pilot_pos, dim: int) -> np.ndarray | None:
    if best_pilot_pos is None:
        return None
    x0 = np.asarray(best_pilot_pos, dtype=np.float64)
    if x0.shape != (int(dim),) or not np.all(np.isfinite(x0)):
        return None
    return np.ascontiguousarray(x0)


def _metad_cv_supported(prob) -> bool:
    return int(np.asarray(prob.low).size) >= 2 and int(np.asarray(prob.high).size) >= 2


def _run_cutest_dominant_multistart_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_chains,
    k_per_epoch,
):
    auto_multistart_polish = _run_cutest_multistart_polish(
        anneal_module,
        prob,
        grad_fn,
        grad_kind,
        seed,
        int(n_chains),
        int(k_per_epoch),
    )
    if auto_multistart_polish is None:
        return None
    polish_bv, polish_calls, polish_values = auto_multistart_polish
    if (
        _polish_values_agree_to_roundoff(polish_values)
        or _polish_best_dominates_sample(polish_values)
        or _polish_bulk_dominates_worst_tail(polish_values)
    ):
        return polish_bv, polish_calls
    return None


def _run_cutest_native_qmc_box_schedule(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_chains,
    k_per_epoch,
):
    if grad_kind != "native" or not _has_finite_design_box(prob):
        return None
    best_val = None
    total_work = 0
    for n_starts, top_k in _native_qmc_box_stage_specs(prob.dim, n_chains):
        result = _run_cutest_qmc_polish(
            anneal_module,
            prob,
            grad_fn,
            grad_kind,
            seed,
            n_starts,
            _native_qmc_polish_budget(k_per_epoch, prob.dim, n_starts),
            top_k=top_k,
        )
        if result is None:
            continue
        value, work_units = result
        total_work += int(work_units)
        if best_val is None or value < best_val:
            best_val = value
    if best_val is None:
        return None
    return best_val, total_work


def _qmc_differential_population_size(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return max(4, 2 * int(dim) * int(n_chains))


def _run_cutest_qmc_differential_search(
    prob,
    seed,
    n_chains,
    max_fevals_per_chain,
):
    dim = int(prob.dim)
    if dim < 1 or n_chains < 1 or max_fevals_per_chain < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    pop_size = _qmc_differential_population_size(dim, n_chains)
    best_val = float("inf")
    best_pos = None
    total_calls = 0
    for chain in range(int(n_chains)):
        chain_seed = int(seed) + int(chain)
        rng = np.random.default_rng(chain_seed)
        pop = np.ascontiguousarray(
            _low_discrepancy_starts(
                prob.low,
                prob.high,
                pop_size,
                chain_seed,
                design_low,
                design_high,
            )
        )
        values = np.asarray([float(prob.fn(x)) for x in pop], dtype=np.float64)
        calls = int(pop_size)
        finite = np.where(np.isfinite(values))[0]
        if finite.size:
            idx = int(finite[np.argmin(values[finite])])
            if values[idx] < best_val:
                best_val = float(values[idx])
                best_pos = pop[idx].copy()
        while calls < int(max_fevals_per_chain):
            for idx in range(pop_size):
                choices = [
                    candidate for candidate in range(pop_size) if candidate != idx
                ]
                a, b, c = rng.choice(choices, 3, replace=False)
                mutant = np.clip(
                    pop[a]
                    + QMC_DIFFERENTIAL_MUTATION_WEIGHT * (pop[b] - pop[c]),
                    design_low,
                    design_high,
                )
                cross = rng.random(dim) < QMC_DIFFERENTIAL_CROSSOVER_RATE
                if not np.any(cross):
                    cross[int(rng.integers(dim))] = True
                trial = np.where(cross, mutant, pop[idx])
                value = float(prob.fn(trial))
                calls += 1
                if np.isfinite(value) and (
                    not np.isfinite(values[idx]) or value < values[idx]
                ):
                    pop[idx] = trial
                    values[idx] = value
                    if value < best_val:
                        best_val = value
                        best_pos = trial.copy()
                if calls >= int(max_fevals_per_chain):
                    break
        total_calls += calls
    if best_pos is None or not np.isfinite(best_val):
        return None
    return best_val, total_calls


def _shifted_qmc_start_count(dim, n_chains):
    if dim < 1:
        raise ValueError("dim must be positive")
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(dim) * int(n_chains) * int(n_chains) * int(n_chains)


def _shifted_qmc_top_k(n_chains):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    return int(n_chains) * int(n_chains)


def _run_cutest_shifted_qmc_polish(
    anneal_module,
    prob,
    grad_fn,
    grad_kind,
    seed,
    n_chains,
    k_per_epoch,
):
    if not hasattr(anneal_module, "polish"):
        return None
    dim = int(prob.dim)
    if dim < 1 or n_chains < 1 or k_per_epoch < 1:
        return None
    design_low, design_high = _design_bounds(prob)
    width = design_high - design_low
    if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
        return None
    n_points = _shifted_qmc_start_count(dim, n_chains)
    starts = _low_discrepancy_starts(
        prob.low,
        prob.high,
        n_points,
        0,
        design_low,
        design_high,
    )
    unit = (starts - design_low) / width
    shift = np.random.default_rng(int(seed)).random(dim)
    shifted = design_low + width * np.mod(unit + shift, 1.0)
    screened = []
    for start in shifted:
        value = float(prob.fn(start))
        if np.isfinite(value):
            screened.append((value, np.ascontiguousarray(start, dtype=np.float64)))
    if not screened:
        return None
    screened.sort(key=lambda item: item[0])
    best_val = screened[0][0]
    total_work = int(n_points)
    for _value, start in screened[: _shifted_qmc_top_k(n_chains)]:
        result = anneal_module.polish(
            prob.fn,
            grad_fn,
            design_low,
            design_high,
            start,
            max_fevals=int(k_per_epoch) * int(n_chains),
        )
        total_work += _polish_work_units(
            dim,
            grad_kind,
            result.get("n_evals", 0),
            result.get("n_grads", 0),
        )
        value = float(result["best_val"])
        if np.isfinite(value) and value < best_val:
            best_val = value
    return best_val, total_work


def _combine_candidate_results(*candidates):
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        return None
    best_val = min(value for value, _work in valid)
    total_work = sum(int(work) for _value, work in valid)
    return best_val, total_work


def _bgsa_run(prob, seed, n_epochs, k_per_epoch, n_chains, driver):
    """Run a v0.5 bGSA driver on a CUTEst problem. Reuses demo_bgsa's
    pilot + driver functions; we monkey-patch OBJ_FN/LOW/HIGH/OBJ_GRAD
    to point at the CUTEst problem so the existing driver wrappers
    work without modification."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import demo_bgsa as d

    # Save / patch globals.
    saved = (
        d.OBJ_FN,
        d.OBJ_GRAD,
        d.LOW,
        d.HIGH,
        getattr(d, "DESIGN_LOW", None),
        getattr(d, "DESIGN_HIGH", None),
    )
    try:
        d.OBJ_FN = prob.fn
        grad_fn, grad_kind = _cutest_gradient(prob)
        d.OBJ_GRAD = grad_fn
        d.LOW = prob.low.astype(np.float64)
        d.HIGH = prob.high.astype(np.float64)
        d.DESIGN_LOW = getattr(prob, "design_low", prob.low).astype(np.float64)
        d.DESIGN_HIGH = getattr(prob, "design_high", prob.high).astype(np.float64)
        auto_best_start_polish = None
        auto_multistart_polish = None
        if driver == "bgsa_auto":
            import anneal

            core_qmc_available = hasattr(anneal, "qmc_polish")
            if core_qmc_available and grad_kind == "native" and _has_finite_design_box(
                prob
            ) and not _has_declared_cutest_bounds(prob):
                auto_best_start_polish = _run_cutest_native_qmc_box_schedule(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    int(n_chains),
                    int(k_per_epoch),
                )
                if auto_best_start_polish is not None:
                    if int(prob.dim) <= int(n_chains):
                        auto_shifted_qmc_polish = _run_cutest_shifted_qmc_polish(
                            anneal,
                            prob,
                            grad_fn,
                            grad_kind,
                            seed,
                            int(n_chains),
                            int(k_per_epoch),
                        )
                        auto_differential_search = _run_cutest_qmc_differential_search(
                            prob,
                            seed,
                            int(n_chains),
                            1 + int(n_epochs) * int(k_per_epoch),
                        )
                        auto_best_start_polish = _combine_candidate_results(
                            auto_best_start_polish,
                            auto_shifted_qmc_polish,
                            auto_differential_search,
                        ) or auto_best_start_polish
            if int(prob.dim) <= int(n_chains) and not (
                core_qmc_available
                and grad_kind == "native"
                and _has_finite_design_box(prob)
            ):
                local_screen_starts = int(n_chains) + int(prob.dim)
                auto_best_start_polish = _run_cutest_qmc_polish(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    local_screen_starts,
                    _covered_local_polish_budget(k_per_epoch, n_chains),
                    top_k=0,
                )
                if auto_best_start_polish is None:
                    auto_best_start_polish = _run_cutest_best_start_polish(
                        anneal,
                        prob,
                        grad_fn,
                        grad_kind,
                        seed,
                        local_screen_starts,
                        _covered_local_polish_budget(k_per_epoch, n_chains),
                    )
                if auto_best_start_polish is not None:
                    return auto_best_start_polish
            elif _has_declared_cutest_bounds(
                prob
            ) and _bounded_polish_dimension_is_covered(prob.dim, n_chains):
                if (
                    grad_kind == "native"
                    and _native_qmc_dense_dimension_is_covered(prob.dim, n_chains)
                ):
                    auto_best_start_polish = _run_cutest_native_qmc_box_schedule(
                        anneal,
                        prob,
                        grad_fn,
                        grad_kind,
                        seed,
                        int(n_chains),
                        int(k_per_epoch),
                    )
                elif grad_kind == "native" and _native_qmc_middle_bound_supported(
                    prob
                ):
                    auto_best_start_polish = _run_cutest_native_qmc_box_schedule(
                        anneal,
                        prob,
                        grad_fn,
                        grad_kind,
                        seed,
                        int(n_chains),
                        int(k_per_epoch),
                    )
                if auto_best_start_polish is None:
                    auto_best_start_polish = _run_cutest_qmc_polish(
                        anneal,
                        prob,
                        _finite_difference_gradient(prob),
                        "finite-difference",
                        seed,
                        int(n_chains),
                        _covered_bound_polish_budget(k_per_epoch, prob.dim, n_chains),
                        top_k=_bounded_polish_top_k(n_chains),
                    )
                if auto_best_start_polish is None:
                    auto_best_start_polish = _run_cutest_raw_best_polish(
                        anneal,
                        prob,
                        _finite_difference_gradient(prob),
                        "finite-difference",
                        seed,
                        int(n_chains),
                        _covered_bound_polish_budget(k_per_epoch, prob.dim, n_chains),
                    )
            elif _has_declared_cutest_bounds(prob) and grad_kind == "native":
                auto_best_start_polish = _run_cutest_native_qmc_box_schedule(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    int(n_chains),
                    int(k_per_epoch),
                )
                if auto_best_start_polish is None:
                    auto_multistart_polish = _run_cutest_dominant_multistart_polish(
                        anneal,
                        prob,
                        grad_fn,
                        grad_kind,
                        seed,
                        int(n_chains),
                        int(k_per_epoch),
                    )
                    if auto_multistart_polish is not None:
                        return auto_multistart_polish
            elif (
                _has_finite_design_box(prob)
                and grad_kind == "native"
                and auto_best_start_polish is None
            ):
                auto_best_start_polish = _run_cutest_native_qmc_box_schedule(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    int(n_chains),
                    int(k_per_epoch),
                )
                if auto_best_start_polish is None:
                    auto_multistart_polish = _run_cutest_dominant_multistart_polish(
                        anneal,
                        prob,
                        grad_fn,
                        grad_kind,
                        seed,
                        int(n_chains),
                        int(k_per_epoch),
                    )
                    if auto_multistart_polish is not None:
                        return auto_multistart_polish
            else:
                auto_multistart_polish = _run_cutest_dominant_multistart_polish(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    int(n_chains),
                    int(k_per_epoch),
                )
                if auto_multistart_polish is not None:
                    return auto_multistart_polish
        # Run the pilot.
        pilot_budget = _bgsa_pilot_budget(n_epochs, k_per_epoch, n_chains)
        out = d.run_pilot(
            seed,
            pilot_budget["n_pilot"],
            pilot_budget["pilot_steps"],
            dim=prob.dim,
            n_rw_pilot=pilot_budget["n_rw_pilot"],
            rw_steps=pilot_budget["rw_steps"],
            n_scout=pilot_budget["n_scout"],
        )
        (
            t_map,
            e_map,
            L_map,
            q_map,
            sigma_map,
            best_pilot_pos,
            pilot_calls,
            t_hot,
            t_rw_map,
            features,
        ) = out
        if driver == "bgsa":
            import anneal

            best_val, work_units = _run_cutest_rust_hmc(
                anneal,
                prob,
                grad_fn,
                grad_kind,
                seed,
                n_epochs,
                k_per_epoch,
                t_map,
                e_map,
                L_map,
                q_map,
                best_pilot_pos,
            )
            return best_val, pilot_calls + work_units
        if driver == "bgsa_metad":
            if not _metad_cv_supported(prob):
                import anneal

                best_val, work_units = _run_cutest_rust_hmc(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    n_epochs,
                    k_per_epoch,
                    t_map,
                    e_map,
                    L_map,
                    q_map,
                    best_pilot_pos,
                )
                return best_val, pilot_calls + work_units
            bv, nc, _, _, _, _ = d.bgsa_metad(
                seed,
                n_epochs,
                k_per_epoch,
                t_rw_map,
                e_map,
                L_map,
                q_map,
                pilot_calls,
                sigma_rw=sigma_map,
                best_pilot_pos=best_pilot_pos,
            )
            return bv, nc
        if driver == "bgsa_pt_metad":
            if not _metad_cv_supported(prob):
                import anneal

                best_val, work_units = _run_cutest_rust_hmc(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    seed,
                    n_epochs,
                    k_per_epoch,
                    t_map,
                    e_map,
                    L_map,
                    q_map,
                    best_pilot_pos,
                )
                return best_val, pilot_calls + work_units
            bv, nc, _, _, _, _, _, _, _ = d.bgsa_pt_metad(
                seed,
                n_epochs,
                n_chains,
                t_rw_map,
                e_map,
                L_map,
                q_map,
                pilot_calls,
                k_inner=20,
                k_swap=5,
                sigma_rw=sigma_map,
                t_hot=t_hot,
            )
            return bv, nc
        if driver == "bgsa_auto":
            import anneal

            candidate_budget = _ensemble_candidate_epoch_budget(k_per_epoch, 4)
            hmc_bv, hmc_calls = _run_cutest_rust_hmc(
                anneal,
                prob,
                grad_fn,
                grad_kind,
                seed,
                n_epochs,
                candidate_budget,
                t_map,
                e_map,
                L_map,
                q_map,
                best_pilot_pos,
            )
            hybrid_inner = _pt_hmc_inner_steps_per_epoch_budget(
                candidate_budget, n_chains, prob.dim, max(1, int(L_map)), grad_kind
            )
            hybrid_bv, hybrid_calls, _, _, _, _, _, _ = d.bgsa_pt_hybrid_v2(
                seed + 3,
                n_epochs,
                n_chains,
                t_map,
                e_map,
                L_map,
                q_map,
                pilot_calls=0,
                k_inner=hybrid_inner,
                k_swap=max(1, min(5, hybrid_inner)),
                t_hot=t_hot,
            )
            outcomes = [
                (hmc_bv, hmc_calls),
                (hybrid_bv, hybrid_calls),
            ]
            if auto_best_start_polish is not None:
                polish_bv, polish_calls = auto_best_start_polish
                outcomes.append((polish_bv, polish_calls))
            if auto_multistart_polish is not None:
                polish_bv, polish_calls, _polish_values = auto_multistart_polish
                outcomes.append((polish_bv, polish_calls))
            if hasattr(anneal, "polish"):
                polish_bv, polish_calls = _run_cutest_rust_polish(
                    anneal,
                    prob,
                    grad_fn,
                    grad_kind,
                    best_pilot_pos,
                    k_per_epoch,
                )
                outcomes.append((polish_bv, polish_calls))
            for mix_seed in (seed, seed + 4):
                mix_bv, mix_calls = bayesian_mixing_sa(
                    prob,
                    mix_seed,
                    1 + int(n_epochs) * int(k_per_epoch),
                )
                outcomes.append((mix_bv, mix_calls))
            if _metad_cv_supported(prob):
                metad_bv, metad_calls, _, _, _, _ = d.bgsa_metad(
                    seed + 1,
                    n_epochs,
                    candidate_budget,
                    t_rw_map,
                    e_map,
                    L_map,
                    q_map,
                    pilot_calls=0,
                    sigma_rw=sigma_map,
                    best_pilot_pos=best_pilot_pos,
                )
                pt_inner = max(1, candidate_budget // max(1, int(n_chains)))
                pt_bv, pt_calls, _, _, _, _, _, _, _ = d.bgsa_pt_metad(
                    seed + 2,
                    n_epochs,
                    n_chains,
                    t_rw_map,
                    e_map,
                    L_map,
                    q_map,
                    pilot_calls=0,
                    k_inner=pt_inner,
                    k_swap=max(1, min(5, pt_inner)),
                    sigma_rw=sigma_map,
                    t_hot=t_hot,
                )
                outcomes.extend([(metad_bv, metad_calls), (pt_bv, pt_calls)])
            return min(value for value, _calls in outcomes), pilot_calls + sum(
                calls for _value, calls in outcomes
            )
        raise ValueError(f"Unknown bGSA driver: {driver}")
    finally:
        (
            d.OBJ_FN,
            d.OBJ_GRAD,
            d.LOW,
            d.HIGH,
            d.DESIGN_LOW,
            d.DESIGN_HIGH,
        ) = saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cutest_benchmarks.csv")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--k-fixed", type=int, default=200)
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--k-min", type=int, default=30)
    p.add_argument("--k-check", type=int, default=20)
    p.add_argument("--k-max", type=int, default=200)
    p.add_argument("--rhat-threshold", type=float, default=1.2)
    p.add_argument("--straggler-top-k", type=int, default=2)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("Loading CUTEst manifest...")
    problems = load_default_manifest()
    print(
        f"Loaded {len(problems)} problems. Running {args.seeds} seeds x "
        f"{len(DRIVERS)} drivers = {args.seeds * len(DRIVERS) * len(problems)} cells."
    )

    rows = []
    t_start = time.perf_counter()
    for prob in problems:
        f0 = prob.fn((prob.low + prob.high) / 2)
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            bv, nc = classical_sa(prob, seed, args.n_epochs, args.k_fixed)
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="classical",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            bv, nc = mcmc_sa(
                prob,
                seed,
                args.n_epochs,
                args.n_chains,
                args.k_min,
                args.k_check,
                args.k_max,
                args.rhat_threshold,
            )
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="mcmc_sa",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            bv, nc = mcmc_sa(
                prob,
                seed,
                args.n_epochs,
                args.n_chains,
                args.k_min,
                args.k_check,
                args.k_max,
                args.rhat_threshold,
                sparse=True,
                straggler_top_k=args.straggler_top_k,
            )
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="mcmc_sa_sparse",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            bv, nc = mcmc_sa_budgeted(
                prob,
                seed,
                args.n_epochs,
                args.n_chains,
                args.k_fixed,
                args.k_min,
                args.k_check,
                args.rhat_threshold,
            )
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="mcmc_sa_budgeted",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            bv, nc = mcmc_sa_budgeted(
                prob,
                seed,
                args.n_epochs,
                args.n_chains,
                args.k_fixed,
                args.k_min,
                args.k_check,
                args.rhat_threshold,
                sparse=True,
                straggler_top_k=args.straggler_top_k,
            )
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="mcmc_sa_sparse_budgeted",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            bv, nc = pt_sa_budgeted(
                prob, seed, args.n_epochs, args.n_chains, args.k_fixed
            )
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="pt_sa_budgeted",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            t0 = time.perf_counter()
            max_fevals = 1 + args.n_epochs * args.k_fixed
            bv, nc = bayesian_mixing_sa(prob, seed, max_fevals)
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="bayesian_mixing_sa",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                )
            )

            # Rank-1 (mean-field) independence-sampler SA: the separable
            # surrogate as a samplable tempered density, in the Rust core.
            t0 = time.perf_counter()
            try:
                import anneal as _anneal_mod

                add_budget = 1 + args.n_epochs * args.k_fixed
                # Use the finite design box, not the raw CUTEst bounds: an
                # unconstrained problem reports +/-1e20, and sampling the
                # surrogate over that span would feed extreme points to the
                # Fortran objective.
                add_low, add_high = _design_bounds(prob)
                add_out = _anneal_mod.additive_independence(
                    prob.fn,
                    np.asarray(add_low, dtype=np.float64),
                    np.asarray(add_high, dtype=np.float64),
                    int(add_budget),
                    seed=int(seed),
                )
                bv, nc = float(add_out["best_val"]), int(add_out["n_evals"])
            except Exception as exc:
                print(
                    f"    additive_indep failed on {prob.name} seed {seed}: "
                    f"{type(exc).__name__}: {exc}"
                )
                bv, nc = float("nan"), 0
            wt = time.perf_counter() - t0
            rows.append(
                dict(
                    problem=prob.name,
                    dim=prob.dim,
                    driver="additive_indep",
                    seed=seed,
                    fevals=nc,
                    best_val=bv,
                    wall_time_s=wt,
                    f_x0=f0,
                    solved=int(
                        math.isfinite(bv)
                        and (bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0)
                    ),
                )
            )

            # External field at budget parity: scipy global optimisers, CMA-ES,
            # and PDFO, each capped at the same work-unit budget as the SA/MCMC
            # drivers via _BudgetedObjective. A missing optional package or a
            # solver error is recorded as a non-improving cell so the sweep
            # never dies on one driver.
            field_budget = 1 + args.n_epochs * args.k_fixed
            for ext_name, ext_fn in SCIPY_DRIVERS.items():
                t0 = time.perf_counter()
                try:
                    bv, nc = ext_fn(prob, seed, field_budget)
                    field_ok = True
                except Exception as exc:
                    print(
                        f"    {ext_name} failed on {prob.name} seed {seed}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    bv, nc, field_ok = float("nan"), 0, False
                wt = time.perf_counter() - t0
                rows.append(
                    dict(
                        problem=prob.name,
                        dim=prob.dim,
                        driver=ext_name,
                        seed=seed,
                        fevals=nc,
                        best_val=bv,
                        wall_time_s=wt,
                        f_x0=f0,
                        solved=int(
                            field_ok
                            and (bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0)
                        ),
                    )
                )

            # v0.5 bGSA stack on the same CUTEst problem.
            for bgsa_drv in ["bgsa", "bgsa_metad", "bgsa_pt_metad", "bgsa_auto"]:
                try:
                    t0 = time.perf_counter()
                    bv, nc = _bgsa_run(
                        prob, seed, args.n_epochs, args.k_fixed, args.n_chains, bgsa_drv
                    )
                    wt = time.perf_counter() - t0
                    rows.append(
                        dict(
                            problem=prob.name,
                            dim=prob.dim,
                            driver=bgsa_drv,
                            seed=seed,
                            fevals=nc,
                            best_val=bv,
                            wall_time_s=wt,
                            f_x0=f0,
                            solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0),
                        )
                    )
                except Exception as exc:
                    # Don't kill the whole sweep on a single driver failure;
                    # mark the cell as failed and move on.
                    print(
                        f"    {bgsa_drv} failed on {prob.name} seed {seed}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    rows.append(
                        dict(
                            problem=prob.name,
                            dim=prob.dim,
                            driver=bgsa_drv,
                            seed=seed,
                            fevals=0,
                            best_val=float("nan"),
                            wall_time_s=0.0,
                            f_x0=f0,
                            solved=0,
                        )
                    )
        elapsed = time.perf_counter() - t_start
        print(f"  done {prob.name:<10} (n={prob.dim:>3}) -- elapsed {elapsed:.1f}s")
        # Checkpoint after every problem so a later hard crash (e.g. a C-level
        # segfault in an external solver) never discards the completed rows.
        _write_cutest_rows(args.out, rows)

    _write_cutest_rows(args.out, rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    for driver in DRIVERS:
        sub = [r for r in rows if r["driver"] == driver]
        solved = sum(r["solved"] for r in sub)
        mean_fevals = np.mean([r["fevals"] for r in sub])
        print(
            f"  {driver:<16}: solved {solved}/{len(sub)} cells, mean fevals = {mean_fevals:.0f}"
        )


if __name__ == "__main__":
    sys.exit(main())
