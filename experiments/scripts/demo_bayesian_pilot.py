"""Demonstration: Bayesian-pilot SA (Method A) vs hand-tuned grid SA.

Pipeline:
  1. Pilot phase: draw N_p hyperparameter samples (T_0, sigma) from a
     log-Normal prior, run a short SA chain at each, record empirical
     acceptance rate.
  2. Laplace fit: minimise the negative log posterior over (log T_0,
     log sigma) given the pilot observations + the Roberts/Rosenthal
     0.234 target acceptance rate. Returns MAP + posterior SDs.
  3. Production: run a long SA at the MAP hyperparameters.

Baseline: a 5x5 grid search over (T_0 in {0.5, 1, 2, 5, 10}) x (sigma in
{0.1, 0.3, 0.5, 1.0, 2.0}). For each grid cell, run a short SA. Pick
the cell with the best best-val. Use that cell for the production run.

Both methods get the same total compute budget: N_p pilot evaluations +
production evaluations.

Reports:
  - MAP hyperparameters from Laplace
  - Best grid hyperparameters
  - Production best_val for each
  - Per-seed paired comparison
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from experiments.shared.runner import (
    gaussian_propose,
    log_cool,
    metropolis_accept_prob,
)


def rastrigin_5d(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


LOW = np.full(5, -5.12)
HIGH = np.full(5, 5.12)
TARGET_ACCEPT_RATE = 0.234


def short_chain(seed, t_init, sigma, n_steps):
    """Returns (best_val, accept_rate, final_pos, n_calls)."""
    rng = np.random.default_rng(seed)
    cur_pos = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_val = rastrigin_5d(cur_pos)
    best_val = cur_val
    accepts = 0
    n_calls = 1
    for step in range(n_steps):
        temp = log_cool(t_init, 2.0, step // 10, np.float64)
        proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
        proposal = np.clip(proposal, LOW, HIGH)
        proposal_val = rastrigin_5d(proposal)
        n_calls += 1
        delta = proposal_val - cur_val
        p = metropolis_accept_prob(delta, temp, np.float64)
        if rng.random() < p:
            cur_pos = proposal
            cur_val = proposal_val
            if proposal_val < best_val:
                best_val = proposal_val
            accepts += 1
    return best_val, accepts / n_steps, cur_pos, n_calls


def production_run(seed, t_init, sigma, n_epochs, k_per_epoch, x0=None):
    """Long SA run at the chosen (t_init, sigma). Optionally warm-starts
    from `x0` instead of a uniform draw."""
    rng = np.random.default_rng(seed + 10_000)
    if x0 is not None:
        cur_pos = np.asarray(x0, dtype=np.float64).copy()
    else:
        cur_pos = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_val = rastrigin_5d(cur_pos)
    best_val = cur_val
    best_pos = cur_pos.copy()
    n_calls = 1
    for epoch in range(n_epochs):
        temp = log_cool(t_init, 2.0, epoch, np.float64)
        for _ in range(k_per_epoch):
            proposal = gaussian_propose(rng, cur_pos, sigma, np.float64)
            proposal = np.clip(proposal, LOW, HIGH)
            proposal_val = rastrigin_5d(proposal)
            n_calls += 1
            delta = proposal_val - cur_val
            p = metropolis_accept_prob(delta, temp, np.float64)
            if rng.random() < p:
                cur_pos = proposal
                cur_val = proposal_val
                if proposal_val < best_val:
                    best_val = proposal_val
                    best_pos = proposal.copy()
    return best_val, n_calls, best_pos


def sample_log_normal(mean, sd, rng):
    return float(np.exp(mean + sd * rng.standard_normal()))


def neg_log_posterior(log_t, log_s, obs, best_val_ref=None):
    """Bayesian negative log-posterior on (log T_init, log sigma).

    Three terms:
      (1) Prior: log-Normal on each.
      (2) Acceptance-rate likelihood: weighted Gaussian on logit(a) vs
          logit(0.234) (Roberts/Rosenthal target).
      (3) Improvement likelihood: weighted Gaussian rewarding pilot chains
          whose best_val is far below the best_val of the worst pilot chain.
          Without this term, we are blind to which hyperparameters
          actually optimised the objective. With it, the posterior
          concentrates on hyperparameters that BOTH mix well (term 2)
          AND find low values (term 3) -- which grid search picks
          implicitly via its argmin-over-cells rule.
    """
    prior_t_mean, prior_t_sd = 0.0, 1.0
    prior_s_mean, prior_s_sd = -0.693, 0.7
    prior_term = 0.5 * (
        ((log_t - prior_t_mean) / prior_t_sd) ** 2
        + ((log_s - prior_s_mean) / prior_s_sd) ** 2
    )

    if best_val_ref is None:
        best_val_ref = max(o["best_val"] for o in obs)
    bv_range = best_val_ref - min(o["best_val"] for o in obs) + 1e-12

    total_w = 0.0
    weighted_dev_accept = 0.0
    weighted_dev_improve = 0.0
    logit_target = np.log(TARGET_ACCEPT_RATE / (1.0 - TARGET_ACCEPT_RATE))
    for o in obs:
        dx = log_t - np.log(o["t_init"])
        dy = log_s - np.log(o["sigma"])
        d2 = dx * dx + dy * dy
        w = np.exp(-0.5 * d2 / 0.5)
        total_w += w
        a = max(min(o["accept_rate"], 1.0 - 1e-6), 1e-6)
        logit_r = np.log(a / (1.0 - a))
        weighted_dev_accept += w * (logit_r - logit_target) ** 2
        # Improvement deviation: 0 = same as best pilot chain; 1 = same as worst.
        # Lower is better, so subtract from 1 to make "good" = 0 deviation.
        norm_improve = (best_val_ref - o["best_val"]) / bv_range
        weighted_dev_improve += w * (1.0 - norm_improve) ** 2

    if total_w > 0:
        accept_term = 0.5 * weighted_dev_accept / total_w / 0.25  # sigma_a = 0.5
        improve_term = 0.5 * weighted_dev_improve / total_w / 0.04  # sigma_i = 0.2
    else:
        accept_term = improve_term = 0.0
    return prior_term + accept_term + improve_term


def fit_laplace_grid(obs):
    n = 31
    grid_t = np.linspace(-4.0, 4.0, n)
    grid_s = np.linspace(-4.0 * 0.7 + (-0.693), 4.0 * 0.7 + (-0.693), n)
    best_val_ref = max(o["best_val"] for o in obs)
    best = (0.0, -0.693)
    best_nll = neg_log_posterior(0.0, -0.693, obs, best_val_ref)
    for log_t in grid_t:
        for log_s in grid_s:
            nll = neg_log_posterior(log_t, log_s, obs, best_val_ref)
            if nll < best_nll:
                best_nll = nll
                best = (log_t, log_s)
    return float(np.exp(best[0])), float(np.exp(best[1])), best_nll


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/bayesian_pilot_demo.csv")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-pilot", type=int, default=12)
    p.add_argument("--pilot-steps", type=int, default=400)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--k-per-epoch", type=int, default=200)
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Rastrigin 5D Bayesian-pilot demo, {args.seeds} seeds")
    print(f"  Pilot:      {args.n_pilot} chains x {args.pilot_steps} steps")
    print(f"  Production: {args.n_epochs} epochs x {args.k_per_epoch} steps each")
    print()

    grid_t = [0.5, 1.0, 2.0, 5.0, 10.0]
    grid_s = [0.1, 0.3, 0.5, 1.0, 2.0]
    rows = []
    for seed in range(args.seeds):
        # ---- Bayesian pilot path ----
        rng = np.random.default_rng(seed)
        pilot_obs = []
        pilot_calls = 0
        best_pilot_pos = None
        best_pilot_val = float("inf")
        for k in range(args.n_pilot):
            t = sample_log_normal(0.0, 1.0, rng)
            s = sample_log_normal(-0.693, 0.7, rng)
            bv, ar, fpos, nc = short_chain(seed * 1000 + k, t, s, args.pilot_steps)
            pilot_obs.append({"t_init": t, "sigma": s, "accept_rate": ar,
                              "best_val": bv, "final_pos": fpos})
            pilot_calls += nc
            if bv < best_pilot_val:
                best_pilot_val = bv
                best_pilot_pos = fpos
        t_map, s_map, _ = fit_laplace_grid(pilot_obs)

        t0 = time.perf_counter()
        bv_bayes, prod_calls, _ = production_run(seed, t_map, s_map,
                                                 args.n_epochs, args.k_per_epoch,
                                                 x0=best_pilot_pos)
        wt_bayes = time.perf_counter() - t0
        rows.append(dict(seed=seed, method="bayesian_pilot",
                         t_init=t_map, sigma=s_map,
                         pilot_calls=pilot_calls, prod_calls=prod_calls,
                         total_calls=pilot_calls + prod_calls,
                         best_val=bv_bayes, wall_time_s=wt_bayes))

        # ---- Hand-tuned grid baseline ----
        k_per_cell = max(1, (args.n_pilot * args.pilot_steps) // 25)
        grid_calls = 0
        best_grid_bv = float("inf")
        best_grid_t, best_grid_s = grid_t[0], grid_s[0]
        best_grid_pos = None
        for tg in grid_t:
            for sg in grid_s:
                bv, ar, fpos, nc = short_chain(seed * 2000 + hash((tg, sg)) % 1000,
                                               tg, sg, k_per_cell)
                grid_calls += nc
                if bv < best_grid_bv:
                    best_grid_bv = bv
                    best_grid_t, best_grid_s = tg, sg
                    best_grid_pos = fpos

        t0 = time.perf_counter()
        bv_grid, prod_calls_grid, _ = production_run(seed, best_grid_t, best_grid_s,
                                                     args.n_epochs, args.k_per_epoch,
                                                     x0=best_grid_pos)
        wt_grid = time.perf_counter() - t0
        rows.append(dict(seed=seed, method="grid_search",
                         t_init=best_grid_t, sigma=best_grid_s,
                         pilot_calls=grid_calls, prod_calls=prod_calls_grid,
                         total_calls=grid_calls + prod_calls_grid,
                         best_val=bv_grid, wall_time_s=wt_grid))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "method", "t_init", "sigma",
                                          "pilot_calls", "prod_calls",
                                          "total_calls", "best_val", "wall_time_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}\n")

    bayes = [r for r in rows if r["method"] == "bayesian_pilot"]
    grid = [r for r in rows if r["method"] == "grid_search"]
    print(f"Bayesian pilot: best_val mean = {np.mean([r['best_val'] for r in bayes]):.3f}  "
          f"std = {np.std([r['best_val'] for r in bayes]):.3f}  "
          f"total_calls mean = {np.mean([r['total_calls'] for r in bayes]):.0f}  "
          f"avg t_init = {np.mean([r['t_init'] for r in bayes]):.2f}  "
          f"avg sigma = {np.mean([r['sigma'] for r in bayes]):.3f}")
    print(f"Grid search:    best_val mean = {np.mean([r['best_val'] for r in grid]):.3f}  "
          f"std = {np.std([r['best_val'] for r in grid]):.3f}  "
          f"total_calls mean = {np.mean([r['total_calls'] for r in grid]):.0f}  "
          f"avg t_init = {np.mean([r['t_init'] for r in grid]):.2f}  "
          f"avg sigma = {np.mean([r['sigma'] for r in grid]):.3f}")

    paired_diff = [bayes[i]["best_val"] - grid[i]["best_val"]
                   for i in range(min(len(bayes), len(grid)))]
    print(f"\nPaired (bayes - grid) best_val: mean = {np.mean(paired_diff):.3f}  "
          f"(< 0 means Bayesian wins)  ratio fevals = "
          f"{np.mean([r['total_calls'] for r in bayes]) / np.mean([r['total_calls'] for r in grid]):.2f}x")


if __name__ == "__main__":
    sys.exit(main())
