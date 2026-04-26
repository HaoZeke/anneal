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
    gaussian_propose,
    log_cool,
    metropolis_accept_prob,
)


def gelman_rubin_max(traces):
    m = len(traces)
    if m < 2:
        return float("inf")
    n = len(traces[0])
    if n < 2:
        return float("inf")
    dim = traces[0][0].shape[0]
    max_rhat = 0.0
    for d in range(dim):
        means = np.array([np.mean([x[d] for x in chain]) for chain in traces])
        vars_ = np.array([np.var([x[d] for x in chain], ddof=1) for chain in traces])
        if np.any(vars_ <= 0):
            continue
        theta_bar = means.mean()
        b = (n / (m - 1.0)) * np.sum((means - theta_bar) ** 2)
        w = vars_.mean()
        var_hat = ((n - 1.0) / n) * w + b / n
        rhat = np.sqrt(var_hat / w)
        if rhat > max_rhat:
            max_rhat = rhat
    return max_rhat


def _straggler_indices(chain_pos, top_k):
    if top_k <= 0 or top_k >= len(chain_pos):
        return list(range(len(chain_pos)))
    pooled = np.mean(chain_pos, axis=0)
    dists = [(i, np.linalg.norm(p - pooled)) for i, p in enumerate(chain_pos)]
    dists.sort(key=lambda x: -x[1])
    return [i for i, _ in dists[:top_k]]


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


def mcmc_sa(prob, seed, n_epochs, n_chains, k_min, k_check, k_max,
            rhat_threshold, sigma=0.5, t_init=5.0,
            sparse=False, straggler_top_k=0):
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
            for _ in range(k_check):
                for c in active:
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
                for c in range(n_chains):
                    if c not in active:
                        traces[c].append(chain_pos[c].copy())
            total_steps += k_check
            rhat = gelman_rubin_max(traces)
    return min(chain_best_val), n_calls


DRIVERS = ["classical", "mcmc_sa", "mcmc_sa_sparse",
           "bgsa", "bgsa_metad", "bgsa_pt_metad", "bgsa_auto"]


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
        out = d.run_pilot(seed, max(8, min(16, n_epochs)),
                          max(40, min(80, k_per_epoch // 2)),
                          dim=prob.dim)
        (t_map, e_map, L_map, q_map, sigma_map,
         best_pilot_pos, pilot_calls, t_hot, t_rw_map, features) = out
        if driver == "bgsa":
            bv, nc, _ = d.hmc_sa(seed, n_epochs, k_per_epoch,
                                  t_map, e_map, L_map,
                                  x0=best_pilot_pos, q=q_map)
            return bv, pilot_calls + nc
        if driver == "bgsa_metad":
            bv, nc, _, _, _, _ = d.bgsa_metad(
                seed, n_epochs, k_per_epoch,
                t_rw_map, e_map, L_map, q_map, pilot_calls,
                sigma_rw=sigma_map)
            return bv, nc
        if driver == "bgsa_pt_metad":
            bv, nc, _, _, _, _, _, _, _ = d.bgsa_pt_metad(
                seed, n_epochs, n_chains,
                t_rw_map, e_map, L_map, q_map, pilot_calls,
                k_inner=20, k_swap=5, sigma_rw=sigma_map, t_hot=t_hot)
            return bv, nc
        if driver == "bgsa_auto":
            bv, nc, _chosen = d.bgsa_auto(
                seed, n_epochs, k_per_epoch, n_chains,
                t_map, e_map, L_map, q_map, t_rw_map, sigma_map,
                t_hot, features, best_pilot_pos, pilot_calls)
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
    print(f"Loaded {len(problems)} problems. Running {args.seeds} seeds x 3 drivers = "
          f"{args.seeds * 3 * len(problems)} cells.")

    rows = []
    t_start = time.perf_counter()
    for prob in problems:
        f0 = prob.fn((prob.low + prob.high) / 2)
        for seed in range(args.seeds):
            t0 = time.perf_counter()
            bv, nc = classical_sa(prob, seed, args.n_epochs, args.k_fixed)
            wt = time.perf_counter() - t0
            rows.append(dict(problem=prob.name, dim=prob.dim, driver="classical",
                             seed=seed, fevals=nc, best_val=bv,
                             wall_time_s=wt, f_x0=f0,
                             solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0)))

            t0 = time.perf_counter()
            bv, nc = mcmc_sa(prob, seed, args.n_epochs, args.n_chains,
                             args.k_min, args.k_check, args.k_max,
                             args.rhat_threshold)
            wt = time.perf_counter() - t0
            rows.append(dict(problem=prob.name, dim=prob.dim, driver="mcmc_sa",
                             seed=seed, fevals=nc, best_val=bv,
                             wall_time_s=wt, f_x0=f0,
                             solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0)))

            t0 = time.perf_counter()
            bv, nc = mcmc_sa(prob, seed, args.n_epochs, args.n_chains,
                             args.k_min, args.k_check, args.k_max,
                             args.rhat_threshold, sparse=True,
                             straggler_top_k=args.straggler_top_k)
            wt = time.perf_counter() - t0
            rows.append(dict(problem=prob.name, dim=prob.dim, driver="mcmc_sa_sparse",
                             seed=seed, fevals=nc, best_val=bv,
                             wall_time_s=wt, f_x0=f0,
                             solved=int(bv < 0.95 * f0 if f0 > 0 else bv < 1.05 * f0)))

            # v0.5 bGSA stack on the same CUTEst problem.
            for bgsa_drv in ["bgsa", "bgsa_metad", "bgsa_pt_metad", "bgsa_auto"]:
                try:
                    t0 = time.perf_counter()
                    bv, nc = _bgsa_run(prob, seed, args.n_epochs,
                                        args.k_fixed, args.n_chains,
                                        bgsa_drv)
                    wt = time.perf_counter() - t0
                    rows.append(dict(problem=prob.name, dim=prob.dim,
                                     driver=bgsa_drv,
                                     seed=seed, fevals=nc, best_val=bv,
                                     wall_time_s=wt, f_x0=f0,
                                     solved=int(bv < 0.95 * f0 if f0 > 0
                                                else bv < 1.05 * f0)))
                except Exception as exc:
                    # Don't kill the whole sweep on a single driver failure;
                    # mark the cell as failed and move on.
                    print(f"    {bgsa_drv} failed on {prob.name} seed {seed}: "
                          f"{type(exc).__name__}: {exc}")
                    rows.append(dict(problem=prob.name, dim=prob.dim,
                                     driver=bgsa_drv,
                                     seed=seed, fevals=0, best_val=float("nan"),
                                     wall_time_s=0.0, f_x0=f0, solved=0))
        elapsed = time.perf_counter() - t_start
        print(f"  done {prob.name:<10} (n={prob.dim:>3}) -- elapsed {elapsed:.1f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "dim", "driver", "seed",
                                          "fevals", "best_val", "wall_time_s",
                                          "f_x0", "solved"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    for driver in DRIVERS:
        sub = [r for r in rows if r["driver"] == driver]
        solved = sum(r["solved"] for r in sub)
        mean_fevals = np.mean([r["fevals"] for r in sub])
        print(f"  {driver:<16}: solved {solved}/{len(sub)} cells, mean fevals = {mean_fevals:.0f}")


if __name__ == "__main__":
    sys.exit(main())
