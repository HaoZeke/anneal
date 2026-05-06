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


def classical_sa(prob, seed, n_epochs, k_fixed, sigma=0.5, t_init=5.0):
    rng = np.random.default_rng(seed)
    cur_pos = rng.uniform(prob.low, prob.high).astype(np.float64)
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
    sigma=0.5,
    t_init=5.0,
    sparse=False,
    straggler_top_k=0,
):
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    chain_pos = [r.uniform(prob.low, prob.high).astype(np.float64) for r in rngs]
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
    sigma=0.5,
    t_init=5.0,
    sparse=False,
    straggler_top_k=0,
):
    if n_chains < 1:
        raise ValueError("n_chains must be positive")
    if epoch_budget < 1:
        raise ValueError("epoch_budget must be positive")

    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    chain_pos = [r.uniform(prob.low, prob.high).astype(np.float64) for r in rngs]
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
    sigma=0.5,
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

    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    swap_rng = np.random.default_rng(seed + n_chains + 1)
    chain_pos = [r.uniform(prob.low, prob.high).astype(np.float64) for r in rngs]
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
    low = np.asarray(prob.low, dtype=np.float64)
    high = np.asarray(prob.high, dtype=np.float64)
    dim = max(len(low), 1)
    span = np.linalg.norm(high - low) / np.sqrt(dim)
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    return float(np.clip(0.25 * span, 0.05, 0.5))


def _auto_initial_temperature(chain_val):
    vals = np.asarray(chain_val, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size < 2:
        return 1.0
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.median(np.abs(finite)))
    return float(np.clip(scale, 1e-6, 5.0))


def bayesian_mixing_sa(prob, seed, max_fevals, return_diagnostics=False):
    if max_fevals < 1:
        raise ValueError("max_fevals must be positive")

    n_chains = _auto_chain_count(prob, max_fevals)
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    controller_rng = np.random.default_rng(seed + 10_007)
    chain_pos = [r.uniform(prob.low, prob.high).astype(np.float64) for r in rngs]
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
    "bgsa",
    "bgsa_metad",
    "bgsa_pt_metad",
    "bgsa_auto",
]


def _rust_hmc_fd_work_units(dim, n_trajectories, l_steps):
    """Objective-call work units for Rust Omelyan HMC with finite differences."""
    return 1 + int(n_trajectories) * (1 + 3 * int(l_steps) * (int(dim) + 1))


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
    saved = (d.OBJ_FN, d.OBJ_GRAD, d.LOW, d.HIGH)
    try:
        d.OBJ_FN = prob.fn
        # CUTEst problems have no analytic gradient available cheaply; use
        # finite differences.
        eps = 1e-6

        def _fd_grad(x):
            x = np.asarray(x, dtype=np.float64)
            f0 = prob.fn(x)
            g = np.zeros_like(x)
            for i in range(len(x)):
                x1 = x.copy()
                x1[i] += eps
                g[i] = (prob.fn(x1) - f0) / eps
            return g

        d.OBJ_GRAD = _fd_grad
        d.LOW = prob.low.astype(np.float64)
        d.HIGH = prob.high.astype(np.float64)
        # Run the pilot.
        out = d.run_pilot(
            seed,
            max(8, min(16, n_epochs)),
            max(40, min(80, k_per_epoch // 2)),
            dim=prob.dim,
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

            l_steps = max(1, int(L_map))
            history = anneal.run_hmc(
                prob.fn,
                _fd_grad,
                prob.low.astype(np.float64),
                prob.high.astype(np.float64),
                t_init=float(t_map),
                epsilon=float(e_map),
                l_steps=l_steps,
                q=float(q_map),
                n_epochs=int(n_epochs),
                steps_per_epoch=int(k_per_epoch),
                seed=int(seed),
                x0=np.asarray(best_pilot_pos, dtype=np.float64),
            )
            n_trajectories = int(n_epochs) * int(k_per_epoch)
            work_units = _rust_hmc_fd_work_units(prob.dim, n_trajectories, l_steps)
            return float(history.best_val), pilot_calls + work_units
        if driver == "bgsa_metad":
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
            )
            return bv, nc
        if driver == "bgsa_pt_metad":
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
            bv, nc, _chosen = d.bgsa_auto(
                seed,
                n_epochs,
                k_per_epoch,
                n_chains,
                t_map,
                e_map,
                L_map,
                q_map,
                t_rw_map,
                sigma_map,
                t_hot,
                features,
                best_pilot_pos,
                pilot_calls,
            )
            return bv, nc
        raise ValueError(f"Unknown bGSA driver: {driver}")
    finally:
        d.OBJ_FN, d.OBJ_GRAD, d.LOW, d.HIGH = saved


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
    print(f"Loading CUTEst manifest...")
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

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "dim",
                "driver",
                "seed",
                "fevals",
                "best_val",
                "wall_time_s",
                "f_x0",
                "solved",
            ],
        )
        w.writeheader()
        w.writerows(rows)
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
