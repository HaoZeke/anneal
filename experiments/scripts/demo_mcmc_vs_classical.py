"""Demonstration: MCMC-SA (multi-chain + Gelman-Rubin termination) vs
classical SA (fixed inner-loop K) on Rastrigin 5D, a standard hard
multimodal benchmark.

The MCMC-SA driver should:
  - find a comparable or better optimum
  - use fewer total proposals on easy epochs (where chains converge fast)
  - use MORE proposals on hard epochs (where chains diverge), which is
    the point: budget allocation tracks the actual difficulty of the
    landscape rather than a hard-coded K.

Reports a CSV row per (driver, seed) with total fevals, best_val,
wall_time, and the per-epoch step trace for MCMC-SA. Pixi task
`mcmc-demo` wires it up under the verify profile."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from experiments.shared.runner import (
    gelman_rubin_max,
    gaussian_propose,
    log_cool,
    metropolis_accept_prob,
)


def rastrigin_5d(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


LOW = np.full(5, -5.12)
HIGH = np.full(5, 5.12)
F_STAR = 0.0  # Rastrigin global minimum at the origin


def classical_sa(seed, n_epochs, k_fixed, t_init=5.0, sigma=0.5):
    rng = np.random.default_rng(seed)
    cur_pos = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_val = rastrigin_5d(cur_pos)
    best_val = cur_val
    best_pos = cur_pos.copy()
    n_calls = 1
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        for _ in range(k_fixed):
            proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
            proposal_val = rastrigin_5d(proposal)
            n_calls += 1
            delta = proposal_val - cur_val
            p = metropolis_accept_prob(delta, temp, np.float64)
            u = rng.random()
            if u < p:
                cur_pos = proposal
                cur_val = proposal_val
                if proposal_val < best_val:
                    best_val = proposal_val
                    best_pos = proposal.copy()
    return best_val, n_calls


def _straggler_indices(chain_pos, top_k):
    if top_k <= 0 or top_k >= len(chain_pos):
        return list(range(len(chain_pos)))
    pooled = np.mean(chain_pos, axis=0)
    dists = [(i, np.linalg.norm(p - pooled)) for i, p in enumerate(chain_pos)]
    dists.sort(key=lambda x: -x[1])
    return [i for i, _ in dists[:top_k]]


def mcmc_sa(seed, n_epochs, n_chains, k_min, k_check, k_max,
            rhat_threshold, t_init=5.0, sigma=0.5,
            sparse_straggler_only=False, straggler_top_k=0):
    rngs = [np.random.default_rng(seed + c) for c in range(n_chains)]
    chain_pos = [r.uniform(LOW, HIGH).astype(np.float64) for r in rngs]
    chain_val = [rastrigin_5d(p) for p in chain_pos]
    chain_best_val = list(chain_val)
    n_calls = n_chains
    epoch_steps = []
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        traces = [[] for _ in range(n_chains)]
        # Phase 1: k_min steps minimum
        for _ in range(k_min):
            for c in range(n_chains):
                proposal = gaussian_propose(rngs[c], chain_pos[c], sigma, np.float64)
                proposal_val = rastrigin_5d(proposal)
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
        # Phase 2: keep stepping until convergence or k_max.
        # In sparse mode, step only the stragglers per batch.
        while rhat > rhat_threshold and total_steps < k_max:
            if sparse_straggler_only and 0 < straggler_top_k < n_chains:
                active = _straggler_indices(chain_pos, straggler_top_k)
            else:
                active = list(range(n_chains))
            for _ in range(k_check):
                for c in active:
                    proposal = gaussian_propose(rngs[c], chain_pos[c], sigma, np.float64)
                    proposal_val = rastrigin_5d(proposal)
                    n_calls += 1
                    delta = proposal_val - chain_val[c]
                    p = metropolis_accept_prob(delta, temp, np.float64)
                    if rngs[c].random() < p:
                        chain_pos[c] = proposal
                        chain_val[c] = proposal_val
                        if proposal_val < chain_best_val[c]:
                            chain_best_val[c] = proposal_val
                    traces[c].append(chain_pos[c].copy())
                # Frozen chains record their unchanged position so the
                # Rhat computation stays length-aligned.
                for c in range(n_chains):
                    if c not in active:
                        traces[c].append(chain_pos[c].copy())
            total_steps += k_check
            rhat = gelman_rubin_max(traces)
        epoch_steps.append(total_steps)
    return min(chain_best_val), n_calls, epoch_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/mcmc_demo.csv")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--k-fixed", type=int, default=200,
                   help="Classical SA inner-loop K.")
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--k-min", type=int, default=30)
    p.add_argument("--k-check", type=int, default=20)
    p.add_argument("--k-max", type=int, default=400)
    p.add_argument("--rhat-threshold", type=float, default=1.2)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"Rastrigin 5D, F* = {F_STAR}, {args.seeds} seeds, {args.n_epochs} epochs:\n")
    print(f"  Classical SA: K = {args.k_fixed} per epoch (fixed)")
    print(f"  MCMC-SA:      {args.n_chains} chains, k_min/check/max = "
          f"{args.k_min}/{args.k_check}/{args.k_max}, Rhat <= {args.rhat_threshold}\n")

    for seed in range(args.seeds):
        t0 = time.perf_counter()
        bv_c, ncalls_c = classical_sa(seed, args.n_epochs, args.k_fixed)
        wt_c = time.perf_counter() - t0
        rows.append(dict(driver="classical", seed=seed, best_val=bv_c,
                         fevals=ncalls_c, wall_time_s=wt_c, mean_steps_per_epoch=args.k_fixed))

        t0 = time.perf_counter()
        bv_m, ncalls_m, steps_m = mcmc_sa(
            seed, args.n_epochs, args.n_chains,
            args.k_min, args.k_check, args.k_max, args.rhat_threshold,
        )
        wt_m = time.perf_counter() - t0
        rows.append(dict(driver="mcmc_sa", seed=seed, best_val=bv_m,
                         fevals=ncalls_m, wall_time_s=wt_m,
                         mean_steps_per_epoch=float(np.mean(steps_m))))

        t0 = time.perf_counter()
        bv_s, ncalls_s, steps_s = mcmc_sa(
            seed, args.n_epochs, args.n_chains,
            args.k_min, args.k_check, args.k_max, args.rhat_threshold,
            sparse_straggler_only=True, straggler_top_k=2,
        )
        wt_s = time.perf_counter() - t0
        rows.append(dict(driver="mcmc_sa_sparse", seed=seed, best_val=bv_s,
                         fevals=ncalls_s, wall_time_s=wt_s,
                         mean_steps_per_epoch=float(np.mean(steps_s))))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["driver", "seed", "best_val",
                                          "fevals", "wall_time_s",
                                          "mean_steps_per_epoch"])
        w.writeheader()
        w.writerows(rows)

    classical = [r for r in rows if r["driver"] == "classical"]
    mcmc = [r for r in rows if r["driver"] == "mcmc_sa"]
    sparse = [r for r in rows if r["driver"] == "mcmc_sa_sparse"]
    for label, group in (("Classical (fixed K)    ", classical),
                         ("MCMC-SA (dense Rhat)   ", mcmc),
                         ("MCMC-SA (sparse skip)  ", sparse)):
        print(f"  {label}: best_val mean = {np.mean([r['best_val'] for r in group]):7.3f}  "
              f"std = {np.std([r['best_val'] for r in group]):6.3f}  "
              f"fevals = {np.mean([r['fevals'] for r in group]):7.0f}  "
              f"steps/epoch = {np.mean([r['mean_steps_per_epoch'] for r in group]):6.1f}")
    print()
    feval_savings = (
        1.0 - np.mean([r["fevals"] for r in sparse]) / np.mean([r["fevals"] for r in mcmc])
    ) * 100
    print(f"Sparse-skip fevals saving vs dense MCMC-SA: {feval_savings:+.1f}%")


if __name__ == "__main__":
    sys.exit(main())
