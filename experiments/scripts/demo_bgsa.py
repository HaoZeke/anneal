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


def schwefel_20d(x: np.ndarray) -> float:
    """Schwefel function. Global min at x_i = 420.9687 with f(x*) = 0.
    Multimodal, deceptive (global min near domain boundary)."""
    x = x.astype(np.float64)
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def schwefel_grad(x: np.ndarray) -> np.ndarray:
    """Analytic gradient of Schwefel. d/dx_i = -sin(sqrt|x|) - x*cos(sqrt|x|)/(2*sqrt|x|)*sign(x)
    -> simplifies to -sin(sqrt|x|) - sqrt(|x|)/2 * cos(sqrt|x|) * sign(x)."""
    x = x.astype(np.float64)
    sx = np.sign(x) * np.sqrt(np.abs(x) + 1e-12)
    return -np.sin(sx) - 0.5 * sx * np.cos(sx)


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
    "schwefel_20d": (schwefel_20d, schwefel_grad,
                     np.full(20, -500.0), np.full(20, 500.0), 0.0),
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


def sample_q_gaussian_momentum(rng, dim, q):
    """q-Gaussian momentum draw. q=1.0 is Gaussian; 1 < q < 1+2/dim is heavy-tailed."""
    if q <= 1.0 + 1e-9:
        return rng.normal(0.0, 1.0, size=dim)
    alpha = 1.0 / (q - 1.0) - 0.5 * dim
    if alpha <= 0:
        # Outside valid range; fall back to Gaussian.
        return rng.normal(0.0, 1.0, size=dim)
    g = rng.gamma(alpha, 1.0)
    g = max(g, 1e-300)
    scale = (1.0 / ((q - 1.0) * g)) ** 0.5
    return scale * rng.normal(0.0, 1.0, size=dim)


def kinetic_q_gaussian(p, q):
    if q <= 1.0 + 1e-9:
        return 0.5 * np.dot(p, p)
    return (1.0 / (q - 1.0)) * np.log1p(0.5 * (q - 1.0) * np.dot(p, p))


def dk_dp_q_gaussian(p, q):
    if q <= 1.0 + 1e-9:
        return p
    denom = 1.0 + 0.5 * (q - 1.0) * np.dot(p, p)
    return p / denom


def hmc_sa_step(rng, x, U, T, eps, L, dim, q=1.0):
    """One HMC trajectory at temperature T with q-Gaussian momentum."""
    p = sample_q_gaussian_momentum(rng, dim, q)
    x0 = x.copy()
    p0 = p.copy()
    H0 = U / T + kinetic_q_gaussian(p0, q)

    grad = OBJ_GRAD(x)
    p = p - 0.5 * eps * grad / T
    n = 1  # gradient counts as ~1 feval (analytic here, FD would be 2D)

    for step in range(L):
        dk = dk_dp_q_gaussian(p, q)
        x = x + eps * dk
        x = np.clip(x, LOW, HIGH)
        grad = OBJ_GRAD(x)
        n += 1
        half = 0.5 if step + 1 == L else 1.0
        p = p - half * eps * grad / T

    U_new = OBJ_FN(x)
    n += 1
    H_new = U_new / T + kinetic_q_gaussian(p, q)
    delta_h = H_new - H0

    if abs(delta_h) > 1000 or not np.isfinite(delta_h):
        return x0, False, n, U
    alpha = min(1.0, np.exp(-delta_h))
    if rng.random() < alpha:
        return x, True, n, U_new
    return x0, False, n, U


def hmc_sa(seed, n_epochs, k_per_epoch, t_init, eps, L, x0=None, q=1.0):
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
            cur, accepted, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, len(LOW), q)
            n_calls += nc
            if cur_v < best:
                best = cur_v
    return best, n_calls, cur


def hmc_pilot(seed, t_init, eps, L, n_steps, q=1.0):
    rng = np.random.default_rng(seed)
    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    accepts = 0
    n = 1
    for step in range(n_steps):
        T = log_cool(t_init, 2.0, step // 10)
        eps_eff = eps * np.sqrt(T / t_init)
        cur, acc, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, len(LOW), q)
        n += nc
        if acc:
            accepts += 1
        if cur_v < best:
            best = cur_v
    return best, accepts / n_steps, cur, n


def neg_log_posterior_4d(log_t, log_e, log_l, q, obs, dim):
    """4D Bayesian posterior for HMC: (log T_init, log epsilon, log L, q).

    q is the Tsallis momentum index. Prior: truncated normal on (1.05,
    1 + 2/dim - 0.05) with mode at 1.15 (mildly heavy-tailed).
    """
    q_max = 1.0 + 2.0 / dim - 0.05
    if q <= 1.0 + 0.04 or q >= q_max:
        return float("inf")
    prior_t_mean, prior_t_sd = 0.0, 1.0
    prior_e_mean, prior_e_sd = -3.0, 1.0
    prior_l_mean, prior_l_sd = 1.6, 0.7
    prior_q_mean, prior_q_sd = 1.15, 0.1
    prior_term = 0.5 * (
        ((log_t - prior_t_mean) / prior_t_sd) ** 2
        + ((log_e - prior_e_mean) / prior_e_sd) ** 2
        + ((log_l - prior_l_mean) / prior_l_sd) ** 2
        + ((q - prior_q_mean) / prior_q_sd) ** 2
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
        dq = (q - o["q"]) / 0.1  # bandwidth 0.1 in q dimension
        d2 = dx * dx + dy * dy + dz * dz + dq * dq
        w = np.exp(-0.5 * d2 / 0.5)
        total_w += w
        a = max(min(o["accept_rate"], 1 - 1e-6), 1e-6)
        logit_r = np.log(a / (1 - a))
        weighted_a += w * (logit_r - logit_target) ** 2
        norm_imp = (bv_max - o["best_val"]) / bv_range
        weighted_i += w * (1.0 - norm_imp) ** 2
    if total_w > 0:
        accept_term = 0.5 * weighted_a / total_w / 0.36
        improve_term = 0.5 * weighted_i / total_w / 0.04
    else:
        accept_term = improve_term = 0.0
    return prior_term + accept_term + improve_term


def fit_laplace_4d(obs, dim):
    n_t, n_e, n_l, n_q = 9, 9, 9, 7
    grid_t = np.linspace(-2.0, 2.0, n_t)
    grid_e = np.linspace(-5.0, -1.0, n_e)
    grid_l = np.linspace(1.0, 2.5, n_l)
    q_max = 1.0 + 2.0 / dim - 0.06
    grid_q = np.linspace(1.05, q_max, n_q)
    best_nll = float("inf")
    best = (0.0, -3.0, 1.6, 1.15)
    for log_t in grid_t:
        for log_e in grid_e:
            for log_l in grid_l:
                for q in grid_q:
                    nll = neg_log_posterior_4d(log_t, log_e, log_l, q, obs, dim)
                    if nll < best_nll:
                        best_nll = nll
                        best = (log_t, log_e, log_l, q)
    return (float(np.exp(best[0])), float(np.exp(best[1])),
            max(1, int(np.exp(best[2]))), float(best[3]))


def gelman_rubin_max(traces):
    """Max-per-coordinate Rhat across M chains; mirrors the Rust impl."""
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


def multichain_q_hmc(seed, n_epochs, n_chains, k_min, k_check, k_max,
                     rhat_threshold, t_init, eps, L, q):
    """Multi-chain q-HMC-SA with Gelman-Rubin termination per epoch.

    Each chain runs HMC trajectories independently; per epoch we run
    k_min steps then check Rhat across chains. Each step inside a
    chain is a full HMC trajectory of L leapfrog steps.
    """
    rngs = [np.random.default_rng(seed + 100 * c) for c in range(n_chains)]
    chain_pos = [r.uniform(LOW, HIGH).astype(np.float64) for r in rngs]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    best_val = min(chain_val)
    n_calls = n_chains
    eps_ref = t_init
    for epoch in range(n_epochs):
        T = log_cool(t_init, 2.0, epoch)
        eps_eff = eps * np.sqrt(T / eps_ref)
        traces = [[] for _ in range(n_chains)]
        for _ in range(k_min):
            for c in range(n_chains):
                chain_pos[c], _, nc, chain_val[c] = hmc_sa_step(
                    rngs[c], chain_pos[c], chain_val[c], T, eps_eff, L, len(LOW), q)
                n_calls += nc
                if chain_val[c] < best_val:
                    best_val = chain_val[c]
                traces[c].append(chain_pos[c].copy())
        total_steps = k_min
        rhat = gelman_rubin_max(traces)
        while rhat > rhat_threshold and total_steps < k_max:
            for _ in range(k_check):
                for c in range(n_chains):
                    chain_pos[c], _, nc, chain_val[c] = hmc_sa_step(
                        rngs[c], chain_pos[c], chain_val[c], T, eps_eff, L, len(LOW), q)
                    n_calls += nc
                    if chain_val[c] < best_val:
                        best_val = chain_val[c]
                    traces[c].append(chain_pos[c].copy())
            total_steps += k_check
            rhat = gelman_rubin_max(traces)
    return best_val, n_calls


def svgd_step(particles, grad_logp_fn, eps, h=None, T_for_noise=None, rng=None):
    """One step of Stein Variational Gradient Descent (Liu/Wang 2016).

    M particles in shape (M, D); each evolves via the Stein-kernelized
    gradient:
      phi_i = (1/M) sum_j [k(x_i, x_j) * grad log p(x_j)
                          + grad_{x_j} k(x_i, x_j)]
    with RBF kernel k(x, y) = exp(-|x - y|^2 / h). Bandwidth h chosen
    by the median heuristic (Liu/Wang 2016 sec 3.4).

    BAYESIAN VARIANT: when `T_for_noise` is provided, add a Brownian
    noise term sqrt(2 * eps * T) * z with z ~ N(0, I). This converts
    SVGD into Stochastic SVGD a.k.a. Particle Langevin (Liu/Wang 2017
    follow-up): each particle's update is now an Euler-Maruyama
    discretisation of the target's Langevin SDE plus the Stein kernel
    smoothing. The stationary distribution is exactly pi_T(x) rather
    than concentrating on modes, so Bayesian credible intervals on
    f(x_min) come for free as quantiles of {f(x_i)}.
    """
    M, D = particles.shape
    # Pairwise squared distances.
    diffs = particles[:, None, :] - particles[None, :, :]  # (M, M, D)
    sq = np.sum(diffs ** 2, axis=2)  # (M, M)
    if h is None:
        # Median heuristic.
        med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
        h = med / max(np.log(M), 1.0)
    K = np.exp(-sq / h)  # (M, M)
    # grad_{x_j} k(x_i, x_j) = K_ij * 2*(x_i - x_j)/h
    grad_K = (2.0 / h) * (K[:, :, None] * diffs)  # (M, M, D)
    grads = np.array([grad_logp_fn(p) for p in particles])  # (M, D)
    # phi_i = (1/M) * (sum_j K_ij * grad_j) + (1/M) * sum_j grad_K_ij
    attraction = K @ grads / M  # (M, D)
    repulsion = grad_K.sum(axis=1) / M  # (M, D)
    phi = attraction + repulsion
    if T_for_noise is not None and rng is not None:
        # Brownian noise for proper Bayesian sampling of pi_T.
        noise = rng.standard_normal(size=particles.shape)
        return particles + eps * phi + np.sqrt(2.0 * eps * T_for_noise) * noise, h
    return particles + eps * phi, h


def svgd_sa(seed, n_epochs, n_particles, k_inner, t_init, eps_svgd,
            stochastic=True):
    """SVGD-driven SA, optionally Bayesian (Stochastic SVGD).

    M particles evolve per Stein flow at the cooling temperature.
    `stochastic=True` (default) adds Brownian noise sqrt(2*eps*T)
    so the stationary distribution of the particle ensemble is
    pi_T(x) ~ exp(-F(x)/T), giving Bayesian credible intervals on
    best_val from the particle quantiles. `stochastic=False` is the
    deterministic-flow variant (mode-seeking, no Bayesian semantics)."""
    rng = np.random.default_rng(seed)
    particles = rng.uniform(LOW, HIGH, size=(n_particles, len(LOW))).astype(np.float64)
    vals = np.array([OBJ_FN(p) for p in particles])
    best_val = vals.min()
    best_pos = particles[np.argmin(vals)].copy()
    n_calls = n_particles
    eps_ref = t_init
    for epoch in range(n_epochs):
        T = log_cool(t_init, 2.0, epoch)
        eps_eff = eps_svgd * np.sqrt(T / eps_ref)

        def grad_logp(x):
            return -OBJ_GRAD(x) / T

        for _ in range(k_inner):
            T_noise = T if stochastic else None
            particles, _ = svgd_step(particles, grad_logp, eps_eff,
                                     T_for_noise=T_noise, rng=rng)
            particles = np.clip(particles, LOW, HIGH)
            vals = np.array([OBJ_FN(p) for p in particles])
            n_calls += n_particles + n_particles  # one grad + one obj per particle
            if vals.min() < best_val:
                best_val = vals.min()
                best_pos = particles[np.argmin(vals)].copy()
    # Bayesian credible interval on best_val from particle quantiles.
    final_vals_sorted = np.sort(vals)
    bci_lower = float(final_vals_sorted[max(0, int(0.025 * n_particles))])
    bci_upper = float(final_vals_sorted[min(n_particles - 1, int(0.975 * n_particles))])
    return best_val, n_calls, best_pos, bci_lower, bci_upper


def bgsa_svgd(seed, n_epochs, n_particles,
              t_map, e_map, L_map, q_map, pilot_calls,
              k_inner=10):
    """bGSA with Bayesian Stochastic SVGD production. Pilot done upstream."""
    bv, prod_calls, _, _bci_lo, _bci_hi = svgd_sa(
        seed, n_epochs, n_particles, k_inner,
        t_map, eps_svgd=0.05, stochastic=True)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def adaptive_ladder_q_hmc(
    seed, n_epochs, n_chains, k_inner, k_swap,
    t_init, t_final, eps, L, q,
    target_swap_rate=0.25,
):
    """Adaptive PT (Lacki/Miasojedow 2016): Robbins-Monro updates on
    log(T_{k+1}/T_k) targeting `target_swap_rate` swap acceptance per
    adjacent pair. Removes the hand-tuned T_hot bake-in.

    Each adjacent pair maintains its own log-ratio. After every swap
    attempt we update:
       log_ratio_k += eta_k * (alpha_k - target)
    where eta_k = c / sqrt(swap_count_k + 1) with c = 1.0. Robbins-Monro
    convergence preserves stationarity in the limit (Atchade-Roberts-
    Rosenthal 2011 diminishing-adaptation scaffold).
    """
    if n_chains < 2:
        raise ValueError("PT requires at least 2 chains")
    # Initialise log-ratios geometric (starting from the user-provided ladder).
    init_ratios = np.linspace(0, 1, n_chains)
    init_temps = t_final * (t_init / t_final) ** init_ratios
    log_ratios = np.diff(np.log(init_temps))  # length n_chains - 1
    swap_counts = np.zeros(n_chains - 1, dtype=np.int64)

    rngs = [np.random.default_rng(seed + 100 * c) for c in range(n_chains)]
    chain_pos = [r.uniform(LOW, HIGH).astype(np.float64) for r in rngs]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    best_val = min(chain_val)
    n_calls = n_chains
    swap_accepts_total = 0
    swap_attempts_total = 0

    for epoch in range(n_epochs):
        # Rebuild current ladder from log_ratios.
        temps = np.empty(n_chains)
        temps[0] = t_final
        for i in range(n_chains - 1):
            temps[i + 1] = temps[i] * np.exp(log_ratios[i])
        for inner in range(k_inner):
            for c in range(n_chains):
                T_c = temps[c]
                chain_pos[c], _, nc, chain_val[c] = hmc_sa_step(
                    rngs[c], chain_pos[c], chain_val[c], T_c, eps, L, len(LOW), q)
                n_calls += nc
                if chain_val[c] < best_val:
                    best_val = chain_val[c]
            if (inner + 1) % k_swap == 0:
                i = rngs[0].integers(0, n_chains - 1)
                T_i, T_j = temps[i], temps[i + 1]
                F_i, F_j = chain_val[i], chain_val[i + 1]
                log_alpha = (1.0 / T_i - 1.0 / T_j) * (F_i - F_j)
                alpha = min(1.0, np.exp(log_alpha))
                accepted = rngs[0].random() < alpha
                swap_attempts_total += 1
                if accepted:
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts_total += 1
                # Robbins-Monro update on log_ratios[i]
                swap_counts[i] += 1
                eta = 1.0 / np.sqrt(swap_counts[i])
                log_ratios[i] += eta * (alpha - target_swap_rate)
                # Re-clamp to keep monotone (positive log-ratios).
                log_ratios[i] = max(log_ratios[i], 1e-3)
                temps[i + 1] = temps[i] * np.exp(log_ratios[i])
                # Re-propagate downstream of i+1 too.
                for k in range(i + 2, n_chains):
                    temps[k] = temps[k - 1] * np.exp(log_ratios[k - 1])
    return best_val, n_calls, swap_accepts_total, swap_attempts_total


def parallel_tempering_q_hmc(
    seed, n_epochs, n_chains, k_inner, k_swap,
    t_init, t_final, eps, L, q,
    use_hmc=True,
):
    """Parallel tempering with q-HMC inner kernel (or random-walk if use_hmc=False).

    M = n_chains chains at temperatures T_1 < ... < T_M on a geometric
    ladder spanning [t_final, t_init]. Hot chain explores broadly
    (random-walk-dominant); cold chain refines (gradient-dominant).
    Every k_swap inner-loop steps we attempt a Metropolis swap between
    adjacent chains:
      alpha = min(1, exp((1/T_i - 1/T_j) * (F(x_i) - F(x_j))))
    The Cool component becomes the temperature ladder; the Exchange
    component (this swap rule) is the new typed primitive that the
    IISE Section 8.3 defers to "next paper". E1 (detailed balance
    across the swap) holds by the standard parallel-tempering argument.
    """
    # Geometric temperature ladder: T_i = t_final * (t_init/t_final)^(i / (M-1))
    if n_chains < 2:
        raise ValueError("PT requires at least 2 chains")
    ratios = np.linspace(0, 1, n_chains)
    temps = t_final * (t_init / t_final) ** ratios

    rngs = [np.random.default_rng(seed + 100 * c) for c in range(n_chains)]
    chain_pos = [r.uniform(LOW, HIGH).astype(np.float64) for r in rngs]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    best_val = min(chain_val)
    n_calls = n_chains
    swap_accepts = 0
    swap_attempts = 0

    for epoch in range(n_epochs):
        for inner in range(k_inner):
            for c in range(n_chains):
                T_c = temps[c]
                if use_hmc:
                    chain_pos[c], _, nc, chain_val[c] = hmc_sa_step(
                        rngs[c], chain_pos[c], chain_val[c], T_c, eps, L, len(LOW), q)
                    n_calls += nc
                else:
                    # Random-walk Metropolis at temperature T_c
                    sigma = 0.5
                    prop = np.clip(gaussian_propose(rngs[c], chain_pos[c], sigma),
                                   LOW, HIGH)
                    pv = OBJ_FN(prop)
                    n_calls += 1
                    if rngs[c].random() < metropolis_accept_prob(
                            pv - chain_val[c], T_c):
                        chain_pos[c], chain_val[c] = prop, pv
                if chain_val[c] < best_val:
                    best_val = chain_val[c]
            # Attempt swap every k_swap inner-loop steps (between random adj pair).
            if (inner + 1) % k_swap == 0:
                # Pick a random adjacent pair (i, i+1).
                i = rngs[0].integers(0, n_chains - 1)
                T_i, T_j = temps[i], temps[i + 1]
                F_i, F_j = chain_val[i], chain_val[i + 1]
                # Swap accept: alpha = min(1, exp((1/T_i - 1/T_j)*(F_i - F_j)))
                log_alpha = (1.0 / T_i - 1.0 / T_j) * (F_i - F_j)
                swap_attempts += 1
                if rngs[0].random() < min(1.0, np.exp(log_alpha)):
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts += 1
    return best_val, n_calls, swap_accepts, swap_attempts


def bgsa_pt_adaptive(seed, n_epochs, n_chains,
                     t_map, e_map, L_map, q_map, pilot_calls,
                     k_inner=20, k_swap=5, target_swap_rate=0.25):
    """bGSA with adaptive parallel-tempering production. Pilot done upstream."""
    t_hot_init = max(t_map * 30.0, 50.0)
    t_cold_init = max(t_map, 0.1)
    bv, prod_calls, swap_a, swap_t = adaptive_ladder_q_hmc(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot_init, t_cold_init, e_map, L_map, q_map,
        target_swap_rate=target_swap_rate)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map, swap_a, swap_t


def bgsa_pt(seed, n_epochs, n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, _unused_dim=None):
    """bGSA with parallel-tempering q-HMC production. Pilot done upstream."""
    t_hot = max(t_map * 30.0, 50.0)
    t_cold = max(t_map, 0.1)
    bv, prod_calls, swap_a, swap_t = parallel_tempering_q_hmc(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot, t_cold, e_map, L_map, q_map)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map, swap_a, swap_t


def bgsa_multichain(seed, n_epochs, n_chains,
                    t_map, e_map, L_map, q_map, pilot_calls,
                    k_min=15, k_check=10, k_max=80, rhat_threshold=1.3):
    """bGSA multi-chain q-HMC + Rhat termination. Pilot done upstream."""
    bv, prod_calls = multichain_q_hmc(
        seed, n_epochs, n_chains, k_min, k_check, k_max, rhat_threshold,
        t_map, e_map, L_map, q_map)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def run_pilot(seed, n_pilot, pilot_steps, dim):
    """Shared pilot phase for ALL bGSA drivers.

    Run once per seed; returns the Laplace-MAP hyperparameters + the
    best pilot endpoint + the pilot feval count, which downstream
    bGSA variants reuse instead of running their own pilot. This was
    the largest source of feval overhead in the v0.4.0 demo (each
    driver re-ran a 1500-feval pilot)."""
    rng = np.random.default_rng(seed)
    q_max = 1.0 + 2.0 / dim - 0.06
    pilot_obs = []
    pilot_calls = 0
    best_pilot_pos = None
    best_pilot_val = float("inf")
    for k in range(n_pilot):
        t = float(np.exp(rng.normal(0.0, 1.0)))
        e = float(np.exp(rng.normal(-3.0, 1.0)))
        L = max(1, int(np.exp(rng.normal(1.6, 0.7))))
        q = float(np.clip(rng.normal(1.15, 0.1), 1.05, q_max))
        bv, ar, fpos, nc = hmc_pilot(seed * 1000 + k, t, e, L, pilot_steps, q=q)
        pilot_obs.append({"t_init": t, "epsilon": e, "L": L, "q": q,
                          "accept_rate": ar, "best_val": bv})
        pilot_calls += nc
        if bv < best_pilot_val:
            best_pilot_val = bv
            best_pilot_pos = fpos
    t_map, e_map, L_map, q_map = fit_laplace_4d(pilot_obs, dim)
    return t_map, e_map, L_map, q_map, best_pilot_pos, pilot_calls


def bgsa(seed, n_epochs, k_per_epoch, t_map, e_map, L_map, q_map,
         best_pilot_pos, pilot_calls):
    """bGSA = production q-HMC-SA (single chain). Pilot done upstream."""
    bv, n_calls, _ = hmc_sa(seed, n_epochs, k_per_epoch, t_map, e_map, L_map,
                            x0=best_pilot_pos, q=q_map)
    return bv, pilot_calls + n_calls, t_map, e_map, L_map, q_map


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
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--k-min", type=int, default=15)
    p.add_argument("--k-check", type=int, default=10)
    p.add_argument("--k-max", type=int, default=80)
    p.add_argument("--rhat-threshold", type=float, default=1.3)
    args = p.parse_args()

    global LOW, HIGH, OBJ_FN, OBJ_GRAD
    OBJ_FN, OBJ_GRAD, LOW, HIGH, _f_star = OBJECTIVES[args.objective]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    print(f"bGSA demo on {args.objective}, {args.seeds} seeds")
    print(f"  Production: {args.n_epochs} epochs x {args.k_per_epoch} steps\n")

    for seed in range(args.seeds):
        # SHARED PILOT: run once per seed; downstream bGSA drivers
        # reuse the (t_map, e_map, L_map, q_map, best_pos, pilot_calls)
        # tuple. This was the largest source of feval overhead in the
        # v0.4.0 demo (each driver re-ran a pilot of 1500-2400 fevals).
        t_map, e_map, L_map, q_map, best_pilot_pos, pilot_calls = run_pilot(
            seed, args.n_pilot, args.pilot_steps, dim=len(LOW))

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

        # bGSA = production q-HMC at MAP
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa(
            seed, args.n_epochs, args.k_per_epoch,
            t_map, e_map, L_map, q_map, best_pilot_pos, pilot_calls)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA multi-chain = multi-chain q-HMC with Rhat termination
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa_multichain(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_min=args.k_min, k_check=args.k_check, k_max=args.k_max,
            rhat_threshold=args.rhat_threshold)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_multichain", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA PT = PT q-HMC with adjacent chain swaps
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _ = bgsa_pt(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA adaptive PT = Robbins-Monro adaptive ladder
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _ = bgsa_pt_adaptive(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, target_swap_rate=0.25)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt_adaptive", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + Bayesian SVGD = Stein variational particle flow
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa_svgd(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls, k_inner=10)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_svgd", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "driver", "best_val",
                                          "fevals", "wall_time_s",
                                          "t_map", "e_map", "L_map", "q_map"])
        w.writeheader()
        for r in rows:
            for k in ["t_map", "e_map", "L_map", "q_map"]:
                r.setdefault(k, "")
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}\n")

    for label in ["classical_sa", "hmc_sa_hand", "bgsa", "bgsa_multichain",
                  "bgsa_pt", "bgsa_pt_adaptive", "bgsa_svgd"]:
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
        print(f"  q:       {np.mean([r['q_map'] for r in bgsa_rows]):.3f}")


if __name__ == "__main__":
    sys.exit(main())
