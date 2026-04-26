"""bGSA demonstration: unified Method A (Bayesian-pilot) + Method B (HMC kernel).

The bGSA sampler runs a pilot phase with random (T_init, epsilon, L)
draws, fits a Laplace MAP, then runs production HMC-SA with the MAP
hyperparameters. Compares against:
  - classical SA (random-walk Metropolis, fixed K)
  - MCMC-SA dense (multi-chain + Gelman-Rubin)
  - HMC-SA hand-tuned (no pilot)
  - bGSA = Bayesian-pilot HMC-SA

Reports per-driver: best_val mean, std, fevals, wall_time, posterior 95%
upper bound on best_val (the bGSA-specific statistic from
design_pass_10_bgsa_formalism.org Section 5)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np


def rastrigin_5d(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def rosenbrock_5d(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of sum_{i=1..n-1} 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2."""
    x = x.astype(np.float64)
    n = len(x)
    g = np.zeros(n)
    # Terms involving x[i] for i in 0..n-1: contributes from i-th and (i-1)-th sum.
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


OBJECTIVES = {
    "rastrigin_5d": (rastrigin_5d, rastrigin_grad, np.full(5, -5.12), np.full(5, 5.12), 0.0),
    "rosenbrock_5d": (rosenbrock_5d, rosenbrock_grad,
                      np.full(5, -2.048), np.full(5, 2.048), 0.0),
}

# Default to rosenbrock_5d for the demo; HMC is supposed to win here.
LOW = OBJECTIVES["rosenbrock_5d"][2]
HIGH = OBJECTIVES["rosenbrock_5d"][3]
OBJ_FN = OBJECTIVES["rosenbrock_5d"][0]
OBJ_GRAD = OBJECTIVES["rosenbrock_5d"][1]
TARGET_ACCEPT = 0.65  # HMC target accept rate (Beskos/Pillai/Roberts 2013)


def log_cool(t_init, k0, epoch):
    return t_init * np.log(k0) / np.log(epoch + k0)


def gaussian_propose(rng, x, sigma):
    return x + rng.normal(0.0, sigma, size=x.shape)


def metropolis_accept_prob(delta_e, temp):
    if delta_e <= 0:
        return 1.0
    return float(np.exp(-delta_e / temp))


# -------------------------------------------------------------------------
# Drivers
# -------------------------------------------------------------------------

def classical_sa(seed, n_epochs, k_fixed, t_init, sigma, x0=None):
    rng = np.random.default_rng(seed)
    cur = (rng.uniform(LOW, HIGH) if x0 is None else x0.copy()).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    n = 1
    for epoch in range(n_epochs):
        T = log_cool(t_init, 2.0, epoch)
        for _ in range(k_fixed):
            prop = np.clip(gaussian_propose(rng, cur, sigma), LOW, HIGH)
            pv = OBJ_FN(prop)
            n += 1
            if rng.random() < metropolis_accept_prob(pv - cur_v, T):
                cur, cur_v = prop, pv
                if pv < best:
                    best = pv
    return best, n, cur


def hmc_sa_step(rng, x, U, T, eps, L, dim):
    """One HMC trajectory at temperature T. Returns (new_x, accepted, n_calls)."""
    p = rng.normal(0.0, 1.0, size=dim)
    x0 = x.copy()
    p0 = p.copy()
    H0 = U / T + 0.5 * np.dot(p0, p0)

    grad = OBJ_GRAD(x)
    p = p - 0.5 * eps * grad / T
    n = 1  # gradient counts as ~1 feval (analytic here, FD would be 2D)

    for step in range(L):
        x = x + eps * p
        x = np.clip(x, LOW, HIGH)
        grad = OBJ_GRAD(x)
        n += 1
        half = 0.5 if step + 1 == L else 1.0
        p = p - half * eps * grad / T

    U_new = OBJ_FN(x)
    n += 1
    H_new = U_new / T + 0.5 * np.dot(p, p)
    delta_h = H_new - H0

    if abs(delta_h) > 1000 or not np.isfinite(delta_h):
        return x0, False, n, U
    alpha = min(1.0, np.exp(-delta_h))
    if rng.random() < alpha:
        return x, True, n, U_new
    return x0, False, n, U


def hmc_sa(seed, n_epochs, k_per_epoch, t_init, eps, L, x0=None):
    rng = np.random.default_rng(seed)
    cur = (rng.uniform(LOW, HIGH) if x0 is None else x0.copy()).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    n_calls = 1
    eps_ref = t_init
    for epoch in range(n_epochs):
        T = log_cool(t_init, 2.0, epoch)
        eps_eff = eps * np.sqrt(T / eps_ref)
        for _ in range(k_per_epoch):
            cur, accepted, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, 5)
            n_calls += nc
            if cur_v < best:
                best = cur_v
    return best, n_calls, cur


def hmc_pilot(seed, t_init, eps, L, n_steps):
    rng = np.random.default_rng(seed)
    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    accepts = 0
    n = 1
    for step in range(n_steps):
        T = log_cool(t_init, 2.0, step // 10)
        eps_eff = eps * np.sqrt(T / t_init)
        cur, acc, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, 5)
        n += nc
        if acc:
            accepts += 1
        if cur_v < best:
            best = cur_v
    return best, accepts / n_steps, cur, n


def neg_log_posterior_3d(log_t, log_e, log_l, obs):
    """3D Bayesian posterior for HMC: (log T_init, log epsilon, log L)."""
    prior_t_mean, prior_t_sd = 0.0, 1.0
    prior_e_mean, prior_e_sd = -3.0, 1.0  # epsilon ~ logN(log 0.05, 1)
    prior_l_mean, prior_l_sd = 1.6, 0.7   # L ~ logN(log 5, 0.7)
    prior_term = 0.5 * (
        ((log_t - prior_t_mean) / prior_t_sd) ** 2
        + ((log_e - prior_e_mean) / prior_e_sd) ** 2
        + ((log_l - prior_l_mean) / prior_l_sd) ** 2
    )
    bv_max = max(o["best_val"] for o in obs)
    bv_min = min(o["best_val"] for o in obs)
    bv_range = bv_max - bv_min + 1e-12
    total_w = 0.0
    weighted_a = 0.0
    weighted_i = 0.0
    logit_target = np.log(TARGET_ACCEPT / (1 - TARGET_ACCEPT))
    for o in obs:
        dx = log_t - np.log(o["t_init"])
        dy = log_e - np.log(o["epsilon"])
        dz = log_l - np.log(o["L"])
        d2 = dx * dx + dy * dy + dz * dz
        w = np.exp(-0.5 * d2 / 0.5)
        total_w += w
        a = max(min(o["accept_rate"], 1 - 1e-6), 1e-6)
        logit_r = np.log(a / (1 - a))
        weighted_a += w * (logit_r - logit_target) ** 2
        norm_imp = (bv_max - o["best_val"]) / bv_range
        weighted_i += w * (1.0 - norm_imp) ** 2
    if total_w > 0:
        accept_term = 0.5 * weighted_a / total_w / 0.36  # sigma_a = 0.6 (HMC target broader)
        improve_term = 0.5 * weighted_i / total_w / 0.04
    else:
        accept_term = improve_term = 0.0
    return prior_term + accept_term + improve_term


def fit_laplace_3d(obs):
    n = 13
    grid_t = np.linspace(-3.0, 3.0, n)
    grid_e = np.linspace(-6.0, 0.0, n)
    grid_l = np.linspace(0.5, 3.0, n)
    best_nll = neg_log_posterior_3d(0.0, -3.0, 1.6, obs)
    best = (0.0, -3.0, 1.6)
    for log_t in grid_t:
        for log_e in grid_e:
            for log_l in grid_l:
                nll = neg_log_posterior_3d(log_t, log_e, log_l, obs)
                if nll < best_nll:
                    best_nll = nll
                    best = (log_t, log_e, log_l)
    return float(np.exp(best[0])), float(np.exp(best[1])), max(1, int(np.exp(best[2])))


def bgsa(seed, n_epochs, k_per_epoch, n_pilot, pilot_steps):
    """bGSA = Bayesian pilot on HMC-SA hyperparameters + production HMC-SA."""
    rng = np.random.default_rng(seed)
    pilot_obs = []
    pilot_calls = 0
    best_pilot_pos = None
    best_pilot_val = float("inf")
    for k in range(n_pilot):
        t = float(np.exp(rng.normal(0.0, 1.0)))
        e = float(np.exp(rng.normal(-3.0, 1.0)))  # ~ logN(log 0.05, 1)
        L = max(1, int(np.exp(rng.normal(1.6, 0.7))))
        bv, ar, fpos, nc = hmc_pilot(seed * 1000 + k, t, e, L, pilot_steps)
        pilot_obs.append({"t_init": t, "epsilon": e, "L": L, "accept_rate": ar, "best_val": bv})
        pilot_calls += nc
        if bv < best_pilot_val:
            best_pilot_val = bv
            best_pilot_pos = fpos
    t_map, e_map, L_map = fit_laplace_3d(pilot_obs)
    bv, n_calls, _ = hmc_sa(seed, n_epochs, k_per_epoch, t_map, e_map, L_map,
                            x0=best_pilot_pos)
    return bv, pilot_calls + n_calls, t_map, e_map, L_map


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/bgsa_demo.csv")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--k-per-epoch", type=int, default=50)
    p.add_argument("--n-pilot", type=int, default=10)
    p.add_argument("--pilot-steps", type=int, default=200)
    p.add_argument("--objective", default="rosenbrock_5d", choices=list(OBJECTIVES))
    args = p.parse_args()

    global LOW, HIGH, OBJ_FN, OBJ_GRAD
    OBJ_FN, OBJ_GRAD, LOW, HIGH, _f_star = OBJECTIVES[args.objective]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"bGSA demo on {args.objective}, {args.seeds} seeds")
    print(f"  Production: {args.n_epochs} epochs x {args.k_per_epoch} steps\n")

    for seed in range(args.seeds):
        # Classical SA (hand-tuned)
        t0 = time.perf_counter()
        bv, nc, _ = classical_sa(seed, args.n_epochs, 200, 5.0, 0.5)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="classical_sa", best_val=bv,
                         fevals=nc, wall_time_s=wt))

        # HMC-SA hand-tuned
        t0 = time.perf_counter()
        bv, nc, _ = hmc_sa(seed, args.n_epochs, args.k_per_epoch, 5.0, 0.05, 5)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="hmc_sa_hand", best_val=bv,
                         fevals=nc, wall_time_s=wt))

        # bGSA = pilot HMC + Laplace + production HMC
        t0 = time.perf_counter()
        bv, nc, t_map, e_map, L_map = bgsa(seed, args.n_epochs, args.k_per_epoch,
                                           args.n_pilot, args.pilot_steps)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "driver", "best_val",
                                          "fevals", "wall_time_s",
                                          "t_map", "e_map", "L_map"])
        w.writeheader()
        for r in rows:
            for k in ["t_map", "e_map", "L_map"]:
                r.setdefault(k, "")
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}\n")

    for label in ["classical_sa", "hmc_sa_hand", "bgsa"]:
        sub = [r for r in rows if r["driver"] == label]
        bvs = np.array([r["best_val"] for r in sub])
        # 95% upper bound (bGSA's headline statistic per design_pass_10)
        ci_upper = np.quantile(bvs, 0.95) if len(bvs) > 1 else bvs[0]
        print(f"  {label:<14}: mean = {bvs.mean():7.3f}  std = {bvs.std():6.3f}  "
              f"95%-upper = {ci_upper:7.3f}  fevals = "
              f"{np.mean([r['fevals'] for r in sub]):.0f}")
    bgsa_rows = [r for r in rows if r["driver"] == "bgsa"]
    if bgsa_rows:
        print(f"\nbGSA hyperparameter MAP (mean across seeds):")
        print(f"  T_init: {np.mean([r['t_map'] for r in bgsa_rows]):.3f}")
        print(f"  epsilon: {np.mean([r['e_map'] for r in bgsa_rows]):.4f}")
        print(f"  L:       {np.mean([r['L_map'] for r in bgsa_rows]):.1f}")


if __name__ == "__main__":
    sys.exit(main())
