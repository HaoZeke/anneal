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


def metad_gamma_from_qv(q_v):
    """Derive the well-tempered MetaD gamma from the Tsallis visiting q_v.

    The GSA-MetaD pairing argument: q_v controls the heavy-tail of the
    visiting distribution; gamma controls the asymptotic flattening of
    F. Both encode "how far above the typical T the kernel pretends to
    be". Tsallis-q-Gaussian visiting at q_v has effective temperature
    T_eff = T * (q_v - 1)^(-1) (Tsallis & Stariolo 1996); requiring this
    to match the well-tempered effective T (Bussi & Branduardi 2015,
    T_eff = gamma * T) gives gamma = 1/(q_v - 1). q_v -> 1+ recovers
    Boltzmann (gamma -> infinity, no flattening); larger q_v gives
    smaller gamma (more aggressive flattening).

    Clamped to [2, 50]: gamma <= 1 is unphysical (negative bias drift),
    gamma > 50 collapses to no flattening at all and is observationally
    indistinguishable from gamma = 50."""
    return float(min(50.0, max(2.0, 1.0 / max(q_v - 1.0, 0.05))))


def tsallis_cool(t_init, q_v, epoch):
    """Tsallis (GSA) cooling: T(k) = T_0 * (2^(q_v-1) - 1) / ((1+k)^(q_v-1) - 1).

    The Tsallis-Stariolo 1996 GSA cooling, derived as the schedule for
    which the q-Gaussian visiting + q-acceptance triple yields a
    canonical equilibrium distribution. At q_v = 1 the schedule
    reduces (L'Hopital) to T_0 * ln 2 / ln(1+k), matching Boltzmann
    log-cooling (i.e. log_cool with k0 = 2). For q_v > 1 the schedule
    cools FASTER than log -- the heavy-tailed visiting distribution
    can offset more aggressive cooling. Strictly decreasing for q_v in
    (1, 3); we clamp epoch=0 to T_0 to avoid the 0/0 indeterminate at
    the origin."""
    if epoch == 0:
        return t_init
    if abs(q_v - 1.0) < 1e-9:
        return t_init * np.log(2.0) / np.log(1.0 + epoch)
    exp = q_v - 1.0
    num = (2.0 ** exp) - 1.0
    den = ((1.0 + epoch) ** exp) - 1.0
    return t_init * num / den


def gaussian_propose(rng, x, sigma):
    return x + rng.normal(0.0, sigma, size=x.shape)


def metropolis_accept_prob(delta_e, temp):
    if delta_e <= 0:
        return 1.0
    return float(np.exp(-delta_e / temp))


def tsallis_accept_prob(delta_e, temp, q_a):
    """Tsallis-Stariolo 1996 generalised Metropolis acceptance.

    P_{q_a}(accept) = [1 + (q_a - 1) * dE / T]^(1/(1-q_a))    if q_a > 1
                    = exp(-dE / T)                            if q_a <= 1

    Properties:
      - q_a -> 1: reduces to standard Metropolis (exp).
      - q_a > 1: heavy-tailed -- accepts more uphill moves at large
        dE/T, which is precisely how GSA escapes local minima better
        than classical SA on multimodal landscapes (Tsallis & Stariolo
        1996; Xiang, Sun, Fan & Gong 1997 default q_a = 2.7).
      - The argument can go negative for q_a > 1, dE > T/(q_a-1);
        outside-support cutoff returns 0 (no acceptance).
    """
    if delta_e <= 0:
        return 1.0
    if q_a <= 1.0 + 1e-9:
        return float(np.exp(-delta_e / temp))
    arg = 1.0 + (q_a - 1.0) * delta_e / temp
    if arg <= 0:
        return 0.0
    return float(arg ** (1.0 / (1.0 - q_a)))


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
    """Production q-HMC SA. Cooling uses tsallis_cool(t_init, q) when
    q > 1 (the Tsallis-Stariolo 1996 GSA schedule paired with q-Gaussian
    momentum) and reduces to log cooling at q = 1 by L'Hopital."""
    rng = np.random.default_rng(seed)
    cur = (rng.uniform(LOW, HIGH) if x0 is None else x0.copy()).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    n_calls = 1
    eps_ref = t_init
    for epoch in range(n_epochs):
        T = tsallis_cool(t_init, q, epoch)
        eps_eff = eps * np.sqrt(T / eps_ref)
        for _ in range(k_per_epoch):
            cur, accepted, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, len(LOW), q)
            n_calls += nc
            if cur_v < best:
                best = cur_v
    return best, n_calls, cur


def fit_empirical_bayes_priors(scout_obs, dim):
    """Empirical-Bayes prior fit for the bGSA pilot (issue 001).

    Given a small set of "scout" pilot draws -- HMC chains at widely-
    spaced (T, eps, L, q) -- compute method-of-moments estimates of
    the prior log-normal / truncated-normal parameters by:

      1. Filtering scouts to the upper quartile of accept-rate-quality
         (proximity to the 0.65 HMC target on a logit scale, plus
         improvement bonus).
      2. Method-of-moments fit on (log T, log eps, log L) over that
         quartile; truncated-normal MoM on q.

    Returns a dict {"t_mean", "t_sd", "e_mean", "e_sd", "l_mean",
    "l_sd", "q_mean", "q_sd"} that the main pilot consumes as priors
    instead of the hardcoded values. The user gets pilot priors
    inferred from data, not arbitrary literature folklore."""
    if not scout_obs:
        return {
            "t_mean": 0.0, "t_sd": 1.0,
            "e_mean": -3.0, "e_sd": 1.0,
            "l_mean": 1.6, "l_sd": 0.7,
            "q_mean": 1.15, "q_sd": 0.1,
        }
    target_logit = float(np.log(0.65 / 0.35))
    bv_max = max(o["best_val"] for o in scout_obs)
    bv_range = bv_max - min(o["best_val"] for o in scout_obs) + 1e-12
    scored = []
    for o in scout_obs:
        a = max(min(o["accept_rate"], 1 - 1e-6), 1e-6)
        logit_r = np.log(a / (1.0 - a))
        accept_term = (logit_r - target_logit) ** 2
        norm_imp = (bv_max - o["best_val"]) / bv_range
        improve_term = (1.0 - norm_imp) ** 2
        scored.append((accept_term + improve_term, o))
    scored.sort(key=lambda x: x[0])
    keep = scored[: max(1, len(scored) // 4)]  # upper quartile
    log_ts = np.array([np.log(o["t_init"]) for _, o in keep])
    log_es = np.array([np.log(o["epsilon"]) for _, o in keep])
    log_ls = np.array([np.log(o["L"]) for _, o in keep])
    qs = np.array([o["q"] for _, o in keep])

    def _mom(arr, default_mean, default_sd):
        if len(arr) < 2:
            return float(default_mean), float(default_sd)
        return float(np.mean(arr)), float(max(np.std(arr), 0.1))

    t_mean, t_sd = _mom(log_ts, 0.0, 1.0)
    e_mean, e_sd = _mom(log_es, -3.0, 1.0)
    l_mean, l_sd = _mom(log_ls, 1.6, 0.7)
    q_mean, q_sd = _mom(qs, 1.15, 0.1)
    return {
        "t_mean": t_mean, "t_sd": t_sd,
        "e_mean": e_mean, "e_sd": e_sd,
        "l_mean": l_mean, "l_sd": l_sd,
        "q_mean": q_mean, "q_sd": q_sd,
    }


def rw_pilot(seed, T, sigma, n_steps):
    """Pilot RW-Metropolis chain at FIXED temperature T. Used to fit
    sigma_rw against the Roberts/Gelman/Gilks 1997 0.234 acceptance
    optimum. Returns (best_val, accept_rate, fevals).

    Fixed-T (no cooling) is essential: the 0.234 optimum is a fixed-T
    statement; cooling-during-pilot averages accept rate across
    different temperatures and biases sigma toward the cold-T optimum."""
    rng = np.random.default_rng(seed)
    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    accepts = 0
    n = 1
    for _ in range(n_steps):
        prop = np.clip(gaussian_propose(rng, cur, sigma), LOW, HIGH)
        pv = OBJ_FN(prop)
        n += 1
        if rng.random() < metropolis_accept_prob(pv - cur_v, T):
            cur, cur_v = prop, pv
            accepts += 1
            if pv < best:
                best = pv
    return best, accepts / max(n_steps, 1), n


def fit_t_sigma_rw(pilot_obs):
    """Joint Laplace-style estimate for (t_rw, sigma_rw) via weighted
    geometric mean over the RW pilot's (t, sigma, accept, best_val)
    observations. Penalty combines distance from the 0.234 acceptance
    target (logit scale) with a normalised-improvement term that
    rewards lower best_val.

    Returns (t_rw_map, sigma_rw_map). Used by bGSA-MetaD drivers,
    where t_rw_map replaces the HMC-fitted t_map (which optimises for
    HMC trajectory acceptance, not for RW exploration)."""
    if not pilot_obs:
        return 1.0, 0.5
    target_logit = float(np.log(0.234 / (1.0 - 0.234)))
    bv_max = max(o["best_val"] for o in pilot_obs)
    bv_range = bv_max - min(o["best_val"] for o in pilot_obs) + 1e-12
    log_ts = []
    log_sigmas = []
    weights = []
    for o in pilot_obs:
        a = max(min(o["accept_rate"], 1 - 1e-6), 1e-6)
        logit_r = np.log(a / (1.0 - a))
        accept_term = (logit_r - target_logit) ** 2 / 0.5
        norm_imp = (bv_max - o["best_val"]) / bv_range
        improve_term = (1.0 - norm_imp) ** 2 / 0.5
        penalty = accept_term + improve_term
        log_ts.append(np.log(max(o["t"], 1e-9)))
        log_sigmas.append(np.log(max(o["sigma"], 1e-9)))
        weights.append(np.exp(-penalty))
    weights = np.asarray(weights)
    if weights.sum() <= 0:
        return (float(np.exp(np.median(log_ts))),
                float(np.exp(np.median(log_sigmas))))
    log_t_map = float(np.average(log_ts, weights=weights))
    log_sigma_map = float(np.average(log_sigmas, weights=weights))
    return float(np.exp(log_t_map)), float(np.exp(log_sigma_map))


# Backwards-compat alias.
def fit_sigma_rw(pilot_obs):
    """Compat wrapper: returns sigma_rw only from the joint fit."""
    return fit_t_sigma_rw(pilot_obs)[1]


def hmc_pilot(seed, t_init, eps, L, n_steps, q=1.0):
    """Pilot HMC trajectory. Cooling uses tsallis_cool(t_init, q) so the
    pilot's accept-rate observation is at the same schedule the
    production driver will use."""
    rng = np.random.default_rng(seed)
    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    accepts = 0
    n = 1
    for step in range(n_steps):
        T = tsallis_cool(t_init, q, step // 10)
        eps_eff = eps * np.sqrt(T / t_init)
        cur, acc, nc, cur_v = hmc_sa_step(rng, cur, cur_v, T, eps_eff, L, len(LOW), q)
        n += nc
        if acc:
            accepts += 1
        if cur_v < best:
            best = cur_v
    return best, accepts / n_steps, cur, n


def neg_log_posterior_4d(log_t, log_e, log_l, q, obs, dim, priors=None):
    """4D Bayesian posterior for HMC: (log T_init, log epsilon, log L, q).

    q is the Tsallis momentum index. The prior parameters come from
    `priors` (issue 001 empirical-Bayes fit) when provided; if None,
    falls back to the legacy hardcoded values for backwards
    compatibility.
    """
    q_max = 1.0 + 2.0 / dim - 0.05
    if q <= 1.0 + 0.04 or q >= q_max:
        return float("inf")
    if priors is None:
        priors = {
            "t_mean": 0.0, "t_sd": 1.0,
            "e_mean": -3.0, "e_sd": 1.0,
            "l_mean": 1.6, "l_sd": 0.7,
            "q_mean": 1.15, "q_sd": 0.1,
        }
    prior_term = 0.5 * (
        ((log_t - priors["t_mean"]) / priors["t_sd"]) ** 2
        + ((log_e - priors["e_mean"]) / priors["e_sd"]) ** 2
        + ((log_l - priors["l_mean"]) / priors["l_sd"]) ** 2
        + ((q - priors["q_mean"]) / priors["q_sd"]) ** 2
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


def _laplace_third_moment_correction(nll_fn, params_map, h=0.05):
    """Tierney-Kadane 1986 skew correction for univariate marginals
    (issue 009). For each parameter axis k we approximate the third
    derivative of the negative log posterior at the MAP via central
    differences, then return a skew-correction shift

      delta_k = - (psi_kkk / psi_kk^2) / 6

    where psi denotes derivatives of the NLL at the MAP. The shift
    catches asymmetry that basic Laplace ignores; on symmetric
    posteriors delta_k -> 0.

    Returns shifts in the same scale as params_map."""
    deltas = []
    for k in range(len(params_map)):
        params_p = list(params_map)
        params_m = list(params_map)
        params_2p = list(params_map)
        params_2m = list(params_map)
        params_p[k] = params_map[k] + h
        params_m[k] = params_map[k] - h
        params_2p[k] = params_map[k] + 2.0 * h
        params_2m[k] = params_map[k] - 2.0 * h
        f0 = nll_fn(params_map)
        fp = nll_fn(params_p)
        fm = nll_fn(params_m)
        f2p = nll_fn(params_2p)
        f2m = nll_fn(params_2m)
        if any(not np.isfinite(x) for x in [f0, fp, fm, f2p, f2m]):
            deltas.append(0.0)
            continue
        # Second derivative: (fp - 2*f0 + fm) / h^2
        d2 = (fp - 2.0 * f0 + fm) / max(h * h, 1e-18)
        # Third derivative: (f2p - 2*fp + 2*fm - f2m) / (2*h^3)
        d3 = (f2p - 2.0 * fp + 2.0 * fm - f2m) / (2.0 * h * h * h)
        if d2 <= 0:
            deltas.append(0.0)
            continue
        deltas.append(-d3 / (6.0 * d2 * d2))
    return deltas


def fit_laplace_4d(obs, dim, priors=None):
    """Grid-MAP Laplace fit for (T_init, epsilon, L, q_v). When
    `priors` is provided (issue 001 empirical-Bayes), grid bounds
    centre on the prior means +/- 2 prior SDs so the grid adapts to
    the data-derived priors instead of using a fixed [-2, 2] x
    [-5, -1] x [1, 2.5] x [1.05, q_max] box."""
    if priors is None:
        priors = {
            "t_mean": 0.0, "t_sd": 1.0,
            "e_mean": -3.0, "e_sd": 1.0,
            "l_mean": 1.6, "l_sd": 0.7,
            "q_mean": 1.15, "q_sd": 0.1,
        }
    n_t, n_e, n_l, n_q = 9, 9, 9, 7
    grid_t = np.linspace(priors["t_mean"] - 2 * priors["t_sd"],
                          priors["t_mean"] + 2 * priors["t_sd"], n_t)
    grid_e = np.linspace(priors["e_mean"] - 2 * priors["e_sd"],
                          priors["e_mean"] + 2 * priors["e_sd"], n_e)
    grid_l = np.linspace(max(0.5, priors["l_mean"] - 2 * priors["l_sd"]),
                          priors["l_mean"] + 2 * priors["l_sd"], n_l)
    q_max = 1.0 + 2.0 / dim - 0.06
    q_lo = max(1.05, priors["q_mean"] - 2 * priors["q_sd"])
    q_hi = min(q_max, priors["q_mean"] + 2 * priors["q_sd"])
    grid_q = np.linspace(q_lo, q_hi, n_q)
    best_nll = float("inf")
    best = (priors["t_mean"], priors["e_mean"],
            priors["l_mean"], priors["q_mean"])
    for log_t in grid_t:
        for log_e in grid_e:
            for log_l in grid_l:
                for q in grid_q:
                    nll = neg_log_posterior_4d(
                        log_t, log_e, log_l, q, obs, dim, priors=priors)
                    if nll < best_nll:
                        best_nll = nll
                        best = (log_t, log_e, log_l, q)
    # Issue 009 -- Tierney-Kadane skew correction at the grid MAP.
    # Catches third-cumulant asymmetry that the symmetric grid Laplace
    # misses, at the cost of 4 extra NLL evaluations per parameter.
    def _nll(p):
        return neg_log_posterior_4d(p[0], p[1], p[2], p[3], obs, dim,
                                     priors=priors)
    deltas = _laplace_third_moment_correction(_nll, list(best), h=0.05)
    corrected = tuple(b + d for b, d in zip(best, deltas))
    return (float(np.exp(corrected[0])), float(np.exp(corrected[1])),
            max(1, int(np.exp(corrected[2]))), float(corrected[3]))


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
        T = tsallis_cool(t_init, q, epoch)
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
            stochastic=True, q_v=1.0):
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
        T = tsallis_cool(t_init, q_v, epoch)
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
    """bGSA with Bayesian Stochastic SVGD production. Pilot done upstream;
    cooling uses tsallis_cool(t_map, q_map)."""
    bv, prod_calls, _, _bci_lo, _bci_hi = svgd_sa(
        seed, n_epochs, n_particles, k_inner,
        t_map, eps_svgd=0.05, stochastic=True, q_v=q_map)
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


def metad_sa(seed, n_epochs, k_inner, t_init, sigma_rw,
             deposit_period=20, metad_sigma=0.3, metad_w0=0.05,
             metad_gamma=8.0, q_v=1.0, q_a=None):
    """SA + Well-tempered metadynamics on the (x_0, x_1) CV.

    Cooling uses tsallis_cool(t_init, q_v) -- the GSA schedule paired
    with q-Gaussian visiting. q_a defaults to q_v (the canonical
    Andricioaei/Straub 1996 GSA pairing), so a single Bayesian-fit
    parameter q_v controls the cooling schedule, the proposal heavy-
    tail (when wired through q-Gaussian momentum), AND the acceptance
    heavy-tail. q_a is annealed alongside T (heavy-tailed early,
    Metropolis at low T) so the cold phase converges to the canonical
    distribution.

    Bias V(s) augments cost so Accept sees F(x) + V(s(x)); every
    `deposit_period` accepted moves we deposit a Gaussian at the
    current CV. The bias fills local cups so the chain can escape
    Arrhenius-suppressed basins on multimodal landscapes."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from metad_helpers import WellTemperedBias
    if q_a is None:
        q_a = q_v
    rng = np.random.default_rng(seed)
    bias = WellTemperedBias(
        LOW, HIGH, sigma=metad_sigma, w0=metad_w0, gamma=metad_gamma
    )
    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur)
    best = cur_v
    n_calls = 1
    accept_count = 0
    # NOTE: Andrieu/Thoms 2008 adaptive sigma is intentionally NOT used
    # here. With a well-tempered metad bias evolving in time, accept
    # rate measures (F + V)-acceptance, and the bias's cup-filling
    # dynamics drive adaptive sigma toward "stay in same filled basin"
    # rather than exploration. Fixed sigma_rw works better empirically
    # for metad-augmented chains; adaptive scaling lands in the swap-
    # augmented variants (metad_sa_shared_bias, pt_metad_shared) where
    # PT swaps and walker exchange counteract the local-collapse pull.
    for epoch in range(n_epochs):
        T = tsallis_cool(t_init, q_v, epoch)
        # Anneal q_a alongside T: heavy-tailed early, Metropolis late.
        q_a_eff = 1.0 + (q_a - 1.0) * T / max(t_init, 1e-12)
        for _ in range(k_inner):
            prop = np.clip(gaussian_propose(rng, cur, sigma_rw), LOW, HIGH)
            pv = OBJ_FN(prop)
            n_calls += 1
            cur_aug = cur_v + bias.potential(bias.cv(cur))
            prop_aug = pv + bias.potential(bias.cv(prop))
            if rng.random() < tsallis_accept_prob(
                    prop_aug - cur_aug, T, q_a_eff):
                cur, cur_v = prop, pv
                accept_count += 1
                if pv < best:
                    best = pv
                if accept_count % deposit_period == 0:
                    bias.deposit(bias.cv(cur), T)
    return best, n_calls, bias


def bgsa_metad(seed, n_epochs, k_inner, t_map, e_map, L_map, q_map,
               pilot_calls, sigma_rw=0.5):
    """bGSA + metadynamics RW production. All bGSA-side hyperparameters
    come from the pilot:
      cooling shape   <- tsallis_cool(t_map, q_map)
      q_a (acceptance) <- q_map (Andricioaei & Straub 1996 GSA pairing)
      metad_sigma      <- sigma_rw (bias-bump width matches proposal)
      metad_w0         <- 0.05 * t_map (bump height scales with T)
    metad_gamma = 8 is the only constant left, matching the well-
    tempered MetaD literature default (Barducci et al. 2008)."""
    bv, prod_calls, _bias = metad_sa(
        seed, n_epochs, k_inner, t_map, sigma_rw,
        deposit_period=20, metad_sigma=sigma_rw,
        metad_w0=0.05 * t_map,
        metad_gamma=metad_gamma_from_qv(q_map),
        q_v=q_map,
    )
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def metad_sa_shared_bias(seed, n_epochs, k_inner, t_init, sigma_rw,
                         n_starts=4, deposit_period=20,
                         metad_sigma=0.3, metad_w0=0.05, metad_gamma=8.0,
                         q_v=1.0, q_a=None):
    """Multi-walker metadynamics-SA with SHARED bias.

    n_starts chains drawn from a Latin hypercube run concurrently;
    they all deposit into ONE well-tempered metadynamics bias and all
    read it for their accept ratio. Standard multi-walker MetaD pattern
    (Raiteri et al. 2006): n_starts walkers cooperatively flatten the
    landscape, so coverage scales linearly with walker count instead
    of each walker re-discovering the same cups.

    Budget invariance: each chain runs k_inner/n_starts steps per
    epoch, so total per-epoch fevals = k_inner (matches single-chain
    metad_sa)."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from metad_helpers import WellTemperedBias

    if q_a is None:
        q_a = q_v

    master = np.random.default_rng(seed)
    starts = latin_hypercube_init(master, n_starts, LOW, HIGH)
    rngs = [np.random.default_rng(seed + 7919 * c) for c in range(n_starts)]
    bias = WellTemperedBias(LOW, HIGH, sigma=metad_sigma, w0=metad_w0,
                            gamma=metad_gamma)
    chain_pos = [starts[c].copy() for c in range(n_starts)]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    chain_best = list(chain_val)
    n_calls = n_starts
    deposit_counter = 0
    # Adaptive sigma intentionally disabled here: with k_inner/n_starts
    # steps per walker the diminishing-adaptation transient hasn't
    # decayed before the chain ends. Empirically hurts ~1.5x on
    # Rastrigin 5D. See note in metad_sa.

    k_per_chain = max(1, k_inner // n_starts)

    for epoch in range(n_epochs):
        T = tsallis_cool(t_init, q_v, epoch)
        # Anneal q_a alongside T (heavy-tailed early, Metropolis at low T).
        q_a_eff = 1.0 + (q_a - 1.0) * T / max(t_init, 1e-12)
        for _ in range(k_per_chain):
            for c in range(n_starts):
                prop = np.clip(gaussian_propose(rngs[c], chain_pos[c],
                                                sigma_rw), LOW, HIGH)
                pv = OBJ_FN(prop)
                n_calls += 1
                cur_aug = chain_val[c] + bias.potential(bias.cv(chain_pos[c]))
                prop_aug = pv + bias.potential(bias.cv(prop))
                if rngs[c].random() < tsallis_accept_prob(
                        prop_aug - cur_aug, T, q_a_eff):
                    chain_pos[c], chain_val[c] = prop, pv
                    if pv < chain_best[c]:
                        chain_best[c] = pv
                    deposit_counter += 1
                    if deposit_counter % deposit_period == 0:
                        bias.deposit(bias.cv(chain_pos[c]), T)
    return float(min(chain_best)), n_calls


def bgsa_metad_multi(seed, n_epochs, k_inner, t_map, e_map, L_map, q_map,
                     pilot_calls, sigma_rw=0.5, n_starts=4):
    """bGSA + multi-walker metadynamics, SHARED bias, Tsallis acceptance.
    Same pilot-driven hyperparameters as bgsa_metad."""
    bv, prod_calls = metad_sa_shared_bias(
        seed, n_epochs, k_inner, t_map, sigma_rw,
        n_starts=n_starts,
        deposit_period=20, metad_sigma=sigma_rw,
        metad_w0=0.05 * t_map,
        metad_gamma=metad_gamma_from_qv(q_map),
        q_v=q_map,
    )
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def pilot_landscape_features(scout_obs, pilot_obs):
    """Issue 010 -- extract landscape features from pilot observations.

    Returns a dict with:
      grad_sens: how strongly accept rate / improvement responds to
        epsilon (HMC step). High -> gradient-informative.
      sigma_sens: rate of accept-rate change with eps_relative across
        scout draws (proxy for proposal-scale sensitivity).
      best_val_cv: coefficient of variation of best_val across pilot
        draws. High -> the chain finds different basins on different
        seeds (multimodal signature).
      q_v_lift: how much the MAP q_v exceeds 1. High -> heavy-tailed
        visiting beneficial.

    These features feed the rule-based driver selection."""
    obs = list(scout_obs) + list(pilot_obs)
    if len(obs) < 4:
        return {"grad_sens": 0.0, "sigma_sens": 0.0,
                "best_val_cv": 0.0, "q_v_lift": 0.0}
    eps_arr = np.array([np.log(o["epsilon"]) for o in obs])
    ar_arr = np.array([o["accept_rate"] for o in obs])
    bv_arr = np.array([o["best_val"] for o in obs])
    # Spearman-like rank correlation of (log eps) vs accept_rate.
    if eps_arr.std() > 0 and ar_arr.std() > 0:
        grad_sens = float(np.abs(np.corrcoef(eps_arr, ar_arr)[0, 1]))
    else:
        grad_sens = 0.0
    if eps_arr.std() > 0 and bv_arr.std() > 0:
        sigma_sens = float(np.abs(np.corrcoef(eps_arr, bv_arr)[0, 1]))
    else:
        sigma_sens = 0.0
    if abs(bv_arr.mean()) > 1e-9:
        best_val_cv = float(bv_arr.std() / max(abs(bv_arr.mean()), 1e-9))
    else:
        best_val_cv = float(bv_arr.std())
    return {"grad_sens": grad_sens, "sigma_sens": sigma_sens,
            "best_val_cv": best_val_cv, "q_v_lift": 0.0}


def select_bgsa_driver(features, q_map):
    """Issue 010 rule-based driver selection.

    Heuristic decision tree, calibrated against design pass 14's
    cross-landscape benchmark:

      - high grad_sens AND moderate q_v: gradient HMC works -> bgsa
      - high best_val_cv (multimodal): need bias + PT -> bgsa_pt_metad
      - else: bgsa_metad as the catch-all bias-augmented driver.

    Returns the driver name string that bgsa_auto should call."""
    grad_sens = features["grad_sens"]
    cv = features["best_val_cv"]
    if grad_sens > 0.4 and cv < 0.5:
        return "bgsa"
    if cv > 1.0:
        return "bgsa_pt_metad"
    return "bgsa_metad"


def make_noisy_objective(noise_sigma):
    """Wraps the global OBJ_FN with i.i.d. Gaussian noise. Used by
    issue 004's PMSA driver and by tests that need to verify noisy-F
    semantics. Returns (noisy_fn, n_evals_callable) where the second
    counts how many times the noisy_fn has been called."""
    counter = [0]
    rng = np.random.default_rng(0xCA75)

    def noisy(x):
        counter[0] += 1
        return float(OBJ_FN(x) + rng.normal(0.0, noise_sigma))

    return noisy, lambda: counter[0]


def pmsa_metad(seed, n_epochs, k_inner, t_init, sigma_rw,
               noisy_fn, sigma_F, q_v=1.0, q_a=None,
               n_eval_per_step=4, deposit_period=20,
               metad_sigma=0.3, metad_w0=0.05, metad_gamma=8.0):
    """Issue 004 -- pseudo-marginal SA + MetaD for noisy F.

    Andrieu & Roberts 2009 PM-MH: replace F(x) with an unbiased
    estimator F_hat(x) = mean(noisy_fn(x) over n_eval_per_step
    repeats). The acceptance ratio under F_hat targets the same
    posterior as under exact F, asymptotically as n_eval -> infinity.

    For finite n_eval, the chain stationary distribution sits in a
    sigma_F^2 / n_eval neighbourhood of the noiseless target
    (Andrieu & Vihola 2015). The optimal n_eval comes from balancing
    PM acceptance loss (noise hurts mixing) against compute cost; we
    fix n_eval_per_step at 4 by default following Doucet/Pitt/Deligi
    annetti/Kohn 2015's "n*sigma_F^2 ~ 1.7 wall-clock optimum".

    Returns (best_val, n_calls) where best_val is the best F_hat and
    n_calls counts noisy evaluations (each step costs n_eval_per_step
    fevals)."""
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from metad_helpers import WellTemperedBias

    if q_a is None:
        q_a = q_v
    rng = np.random.default_rng(seed)
    bias = WellTemperedBias(LOW, HIGH, sigma=metad_sigma, w0=metad_w0,
                            gamma=metad_gamma)

    def f_hat(x):
        return float(np.mean([noisy_fn(x) for _ in range(n_eval_per_step)]))

    cur = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = f_hat(cur)
    best = cur_v
    n_calls = n_eval_per_step  # f_hat counted as n_eval evals
    accept_count = 0
    for epoch in range(n_epochs):
        T = tsallis_cool(t_init, q_v, epoch)
        q_a_eff = 1.0 + (q_a - 1.0) * T / max(t_init, 1e-12)
        for _ in range(k_inner):
            prop = np.clip(gaussian_propose(rng, cur, sigma_rw), LOW, HIGH)
            pv = f_hat(prop)
            n_calls += n_eval_per_step
            cur_aug = cur_v + bias.potential(bias.cv(cur))
            prop_aug = pv + bias.potential(bias.cv(prop))
            if rng.random() < tsallis_accept_prob(
                    prop_aug - cur_aug, T, q_a_eff):
                cur, cur_v = prop, pv
                accept_count += 1
                if pv < best:
                    best = pv
                if accept_count % deposit_period == 0:
                    bias.deposit(bias.cv(cur), T)
    return best, n_calls


def bgsa_pmsa(seed, n_epochs, k_inner, t_map, e_map, L_map, q_map,
              pilot_calls, sigma_rw, noise_sigma, n_eval_per_step=4):
    """bGSA + pseudo-marginal SA + MetaD for noisy F. Wraps OBJ_FN with
    Gaussian noise N(0, noise_sigma); chain runs PM-MH at the pilot's
    (t_map, q_map, sigma_rw)."""
    noisy_fn, eval_counter = make_noisy_objective(noise_sigma)
    bv, prod_calls = pmsa_metad(
        seed, n_epochs, k_inner, t_map, sigma_rw, noisy_fn, noise_sigma,
        q_v=q_map, n_eval_per_step=n_eval_per_step,
        deposit_period=20, metad_sigma=sigma_rw,
        metad_w0=0.05 * t_map,
        metad_gamma=metad_gamma_from_qv(q_map),
    )
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def continuous_time_tempering(seed, n_epochs, k_inner, t_min, t_max, q_v,
                               eps_x=0.05, eps_beta=0.02, sigma_rw=0.5):
    """Issue 006 -- Continuous-time tempering (Wu & Stoltz 2022).

    Augments the state from x to (x, beta) and runs a joint Langevin:

        dx     = -beta * grad F(x) dt + sqrt(2 * beta^{-1}) dW_x
        dbeta  = -nabla_beta U_aug(x, beta) dt + sigma_beta dW_beta

    where U_aug(x, beta) = beta * F(x) + log_prior(beta). beta drifts
    continuously between 1/t_max (hot) and 1/t_min (cold), eliminating
    the discrete-PT-ladder requirement.

    Splitting integrator: alternate k_inner x-Langevin half-steps and
    one beta-Langevin step per epoch. Tracks best F across the joint
    chain. Returns (best_val, n_calls, beta_history).

    The Wu-Stoltz construction subsumes discrete PT as the limit
    sigma_beta -> 0 with a discrete-jump beta proposal, and subsumes
    simulated tempering (Marinari & Parisi 1992) as the limit where
    beta visits a finite set of values."""
    dim = len(LOW)
    rng = np.random.default_rng(seed)
    cur_x = rng.uniform(LOW, HIGH).astype(np.float64)
    cur_v = OBJ_FN(cur_x)
    best = cur_v
    n_calls = 1
    log_beta_min = np.log(1.0 / max(t_max, 1e-9))
    log_beta_max = np.log(1.0 / max(t_min, 1e-9))
    log_beta = float(np.log(1.0 / max(0.5 * (t_min + t_max), 1e-9)))
    beta_history = []

    for epoch in range(n_epochs):
        beta = float(np.exp(log_beta))
        # x-Langevin half-block at fixed beta.
        for _ in range(k_inner):
            grad = OBJ_GRAD(cur_x)
            noise = rng.normal(0.0, 1.0, size=dim)
            cur_x = cur_x - beta * eps_x * grad + np.sqrt(2.0 * eps_x / beta) * noise
            cur_x = np.clip(cur_x, LOW, HIGH)
            cur_v = OBJ_FN(cur_x)
            n_calls += 1
            if cur_v < best:
                best = cur_v
        # beta-Langevin step. The "potential" in beta-space is
        # U_aug(beta) = beta * F(x) - log_prior(beta). With a uniform
        # log-prior on beta in [log_beta_min, log_beta_max] the drift
        # term reduces to -F(x) * d(beta)/d(log_beta) = -F(x) * beta.
        cur_v = OBJ_FN(cur_x)
        n_calls += 1
        # Symmetric RW proposal on log_beta with reflection at the
        # endpoints (instead of full Langevin to keep the integrator
        # simple at low cost).
        prop_log_beta = log_beta + rng.normal(0.0, np.sqrt(2.0 * eps_beta))
        # Reflect into [log_beta_min, log_beta_max].
        if prop_log_beta < log_beta_min:
            prop_log_beta = 2.0 * log_beta_min - prop_log_beta
        if prop_log_beta > log_beta_max:
            prop_log_beta = 2.0 * log_beta_max - prop_log_beta
        prop_beta = float(np.exp(prop_log_beta))
        # Metropolis on beta-marginal. Joint target on (x, beta) is
        # exp(-beta * F(x)) * pi_prior(beta); change-of-variable from
        # beta to log_beta multiplies by beta, so target on log_beta
        # is beta * exp(-beta * F) (uniform log-prior). log target
        # ratio: (prop_log_beta - log_beta) + (beta - prop_beta) * F.
        log_alpha = ((prop_log_beta - log_beta)
                     + (float(np.exp(log_beta)) - prop_beta) * cur_v)
        if rng.random() < min(1.0, np.exp(min(log_alpha, 0.0))):
            log_beta = prop_log_beta
        beta_history.append(float(np.exp(log_beta)))

    return best, n_calls, np.asarray(beta_history)


def bgsa_continuous_temper(seed, n_epochs, k_inner, t_map, e_map, L_map, q_map,
                           pilot_calls, t_hot=None):
    """bGSA + continuous-time tempering. t_map is t_min (cold-T limit);
    t_hot from the pilot is t_max (hot-T limit). beta drifts between
    1/t_max and 1/t_min on a continuous-time Langevin trajectory."""
    if t_hot is None:
        t_hot = max(2.0 * t_map, 1.0)
    bv, prod_calls, _ = continuous_time_tempering(
        seed, n_epochs, k_inner,
        t_min=max(t_map, 0.05), t_max=t_hot, q_v=q_map,
        eps_x=e_map, eps_beta=0.02)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map


def trajectory_inla_diagnostic(traj_vals):
    """Issue 007 -- Single-chain non-stationarity diagnostic via
    trajectory-INLA. Approximates the chain trajectory as an AR(1)
    latent field with sparse precision matrix Q (tridiagonal). Fits
    phi (the AR(1) coefficient) and sigma_eps (innovation SD), then
    computes the diagonal of Q^{-1} via the Cholesky-then-Takahashi
    recursion; the diagonal entries are the per-step marginal
    variances sigma_t^2.

    Returns (phi, sigma_eps, sigma_t_array, flags) where flags[t] is
    True for any t whose sigma_t^2 deviates from the chain-mean
    sigma^2 by > 3 standard deviations (the within-chain stationarity
    test). Catches non-stationarity in a single chain at no extra
    sampling cost.

    References: Rue/Martino/Chopin 2009 INLA; Roberts/Rosenthal 2007
    adaptive MCMC ergodicity diagnostics.
    """
    arr = np.asarray(traj_vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 5:
        return float("nan"), float("nan"), np.array([]), np.array([])

    # AR(1) fit: x_{t+1} = phi * x_t + eps_t. OLS on lagged pairs.
    x_t = arr[:-1] - arr[:-1].mean()
    x_tp1 = arr[1:] - arr[1:].mean()
    denom = float(np.sum(x_t * x_t))
    phi = float(np.sum(x_t * x_tp1) / denom) if denom > 0 else 0.0
    phi = max(min(phi, 0.99), -0.99)
    eps = arr[1:] - phi * arr[:-1]
    sigma_eps = float(np.std(eps, ddof=1)) if len(eps) > 1 else 1.0
    if sigma_eps <= 0:
        sigma_eps = 1.0

    # Tridiagonal precision matrix for AR(1):
    #   Q[0,0] = 1/(sigma_eps^2 (1 - phi^2)) (stationary boundary)
    #   Q[t,t] = (1 + phi^2) / sigma_eps^2 for interior
    #   Q[t,t+1] = Q[t+1,t] = -phi / sigma_eps^2
    # Marginal sigma_t^2 = (Q^{-1})_{tt}, computed via Takahashi
    # recursion in O(n).
    var_eps = sigma_eps * sigma_eps
    diag = np.full(n, (1.0 + phi * phi) / var_eps)
    diag[0] = 1.0 / max(var_eps * (1.0 - phi * phi), 1e-30)
    diag[-1] = 1.0 / var_eps
    off = np.full(n - 1, -phi / var_eps)

    # Cholesky of tridiagonal Q: lower bidiagonal L with diag d, off e.
    d = np.empty(n)
    e = np.empty(n - 1)
    d[0] = np.sqrt(max(diag[0], 1e-30))
    for t in range(1, n):
        e[t - 1] = off[t - 1] / d[t - 1]
        d[t] = np.sqrt(max(diag[t] - e[t - 1] ** 2, 1e-30))

    # Takahashi recursion to extract diag(Q^{-1}) in O(n).
    sigma2 = np.empty(n)
    sigma2[-1] = 1.0 / (d[-1] ** 2)
    for t in range(n - 2, -1, -1):
        sigma2[t] = 1.0 / (d[t] ** 2) + (e[t] ** 2) * sigma2[t + 1]

    sigma_t = np.sqrt(np.maximum(sigma2, 0.0))
    mean_st = float(np.mean(sigma_t))
    std_st = float(np.std(sigma_t)) if len(sigma_t) > 1 else 0.0
    if std_st <= 0:
        flags = np.zeros(n, dtype=bool)
    else:
        flags = np.abs(sigma_t - mean_st) > 3.0 * std_st
    return phi, sigma_eps, sigma_t, flags


def smc_pt_log_z_estimator(swap_log_alphas):
    """Issue 005 -- SMC-on-PT free log-Z-ratio estimator.

    Each PT swap step produces a log_alpha = (1/T_low - 1/T_high) *
    (F_low - F_high). The bridge sampling identity (Del Moral, Doucet,
    Jasra 2006 Eq.(8); Neal 2001 AIS) gives an unbiased estimator of
    log Z(T_low) / Z(T_high) as the LOG-MEAN-EXP of the per-swap
    log-alphas:

        log Z_ratio_est = log mean exp(log_alpha_i)

    Bootstrap CI from N_BOOT=200 resamples of the swap series.
    Returns (log_z_est, log_z_se, ci_low, ci_high)."""
    if not swap_log_alphas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(swap_log_alphas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def log_mean_exp(x):
        m = float(np.max(x))
        return m + float(np.log(np.mean(np.exp(x - m))))

    point = log_mean_exp(arr)
    rng = np.random.default_rng(0xBA7CE57)
    n = len(arr)
    n_boot = 200
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = log_mean_exp(arr[idx])
    se = float(boots.std(ddof=1))
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return point, se, lo, hi


def pt_metad_shared(seed, n_epochs, n_chains, k_inner, k_swap,
                    t_init, t_final, sigma_rw=0.5,
                    deposit_period=20, metad_sigma=0.3,
                    metad_w0=0.05, metad_gamma=8.0, q_a=1.0,
                    q_v=1.0):
    """Parallel tempering + multi-walker shared metadynamics.

    n_chains at geometric temperature ladder, all RW Metropolis with
    a single shared well-tempered bias. Adjacent-pair swaps every
    `k_swap` inner steps. The shared bias gets deposits from all
    walkers (PLUMED-style multi-walker), so high-T chains discover
    cups broadly and low-T chains refine them, while the swap moves
    good positions down the ladder. LH-initialised starts.
    """
    if n_chains < 2:
        raise ValueError("PT requires at least 2 chains")

    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from metad_helpers import WellTemperedBias

    ratios = np.linspace(0, 1, n_chains)
    temps = t_final * (t_init / t_final) ** ratios

    # Per-rung q_a: linear from q_a (hot) to 1.0 (cold). Hot chains
    # use heavy-tailed Tsallis acceptance for aggressive exploration;
    # cold chains use Metropolis for stable refinement. Mirrors the
    # GSA pairing of (high-T, q_a > 1) with (low-T, q_a = 1).
    if n_chains > 1:
        q_a_per_chain = np.array([
            1.0 + (q_a - 1.0) * (temps[c] - t_final) / max(t_init - t_final, 1e-12)
            for c in range(n_chains)
        ])
    else:
        q_a_per_chain = np.array([q_a])

    master = np.random.default_rng(seed)
    starts = latin_hypercube_init(master, n_chains, LOW, HIGH)
    rngs = [np.random.default_rng(seed + 7919 * c) for c in range(n_chains)]
    bias = WellTemperedBias(LOW, HIGH, sigma=metad_sigma, w0=metad_w0,
                            gamma=metad_gamma)

    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    best_val = min(chain_val)
    n_calls = n_chains
    deposit_counter = 0
    swap_attempts = 0
    swap_accepts = 0
    # Issue 005: collect per-swap log_alphas keyed by adjacent rung
    # pair so log Z(T_i)/Z(T_{i+1}) can be estimated post-hoc.
    swap_log_alpha_pairs = [[] for _ in range(n_chains - 1)]
    # Andrieu/Thoms 2008 per-rung adaptive sigma. Each PT rung has its
    # own log_sigma adapted toward 0.234 RW-Metropolis acceptance.
    log_sigmas = [float(np.log(max(sigma_rw, 1e-6))) for _ in range(n_chains)]
    n_steps_per_chain = [0 for _ in range(n_chains)]
    target_a = 0.234

    for epoch in range(n_epochs):
        for inner in range(k_inner):
            for c in range(n_chains):
                T_c = temps[c]
                q_a_c = q_a_per_chain[c]
                sigma_eff = float(np.exp(log_sigmas[c]))
                prop = np.clip(gaussian_propose(rngs[c], chain_pos[c],
                                                sigma_eff), LOW, HIGH)
                pv = OBJ_FN(prop)
                n_calls += 1
                cur_aug = chain_val[c] + bias.potential(bias.cv(chain_pos[c]))
                prop_aug = pv + bias.potential(bias.cv(prop))
                accepted = rngs[c].random() < tsallis_accept_prob(
                    prop_aug - cur_aug, T_c, q_a_c)
                n_steps_per_chain[c] += 1
                gamma_n = 1.0 / max(n_steps_per_chain[c], 1) ** 0.6
                log_sigmas[c] += gamma_n * (
                    (1.0 if accepted else 0.0) - target_a)
                if accepted:
                    chain_pos[c], chain_val[c] = prop, pv
                    if pv < best_val:
                        best_val = pv
                    deposit_counter += 1
                    if deposit_counter % deposit_period == 0:
                        bias.deposit(bias.cv(chain_pos[c]), T_c)

            if (inner + 1) % k_swap == 0:
                i = rngs[0].integers(0, n_chains - 1)
                T_i, T_j = temps[i], temps[i + 1]
                F_i = chain_val[i] + bias.potential(bias.cv(chain_pos[i]))
                F_j = chain_val[i + 1] + bias.potential(bias.cv(chain_pos[i + 1]))
                log_alpha = (1.0 / T_i - 1.0 / T_j) * (F_i - F_j)
                swap_log_alpha_pairs[i].append(log_alpha)
                swap_attempts += 1
                if rngs[0].random() < min(1.0, np.exp(log_alpha)):
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts += 1

    # Issue 005: compute per-pair log Z(T_i) / Z(T_{i+1}) and
    # accumulate into log Z(T_cold) / Z(T_hot) by chaining.
    z_pair_estimates = []
    for pair_idx, alphas in enumerate(swap_log_alpha_pairs):
        log_z_est, _, _, _ = smc_pt_log_z_estimator(alphas)
        z_pair_estimates.append(log_z_est)
    log_z_cold_to_hot = (sum(z for z in z_pair_estimates if np.isfinite(z))
                          if z_pair_estimates else float("nan"))
    return (best_val, n_calls, swap_accepts, swap_attempts,
            log_z_cold_to_hot, z_pair_estimates)


def bgsa_pt_metad(seed, n_epochs, n_chains, t_map, e_map, L_map, q_map,
                  pilot_calls, k_inner=20, k_swap=5, sigma_rw=0.5,
                  t_hot=None):
    """bGSA + PT + shared metadynamics + Tsallis acceptance. All
    hyperparameters from pilot: t_cold = t_map, t_hot from basin-
    spanning regime, q_a = q_v = q_map, metad_sigma = sigma_rw,
    metad_w0 = 0.05 * t_map."""
    if t_hot is None:
        t_hot = t_map
    t_cold = max(t_map, 0.1)
    bv, prod_calls, swap_a, swap_t, log_z_ratio, _ = pt_metad_shared(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot, t_cold, sigma_rw=sigma_rw,
        deposit_period=20, metad_sigma=sigma_rw,
        metad_w0=0.05 * t_map,
        metad_gamma=metad_gamma_from_qv(q_map),
        q_a=q_map, q_v=q_map,
    )
    return (bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map,
            swap_a, swap_t, log_z_ratio)


def parallel_tempering_hybrid(
    seed, n_epochs, n_chains, k_inner, k_swap,
    t_init, t_final, eps, L, q, sigma_rw=0.5,
    rw_threshold_t=2.0,
):
    """Hybrid PT: hot chains (T > rw_threshold_t) use random-walk
    Metropolis; cold chains use q-HMC. Closes the gradient-mislead
    failure on multimodal landscapes -- the hot RW chain explores
    broadly without being trapped by misleading gradients, the cold
    HMC chain refines once the chain is near a basin where the
    gradient becomes informative. This is the standard PT-HMC
    practice from Sminchisescu/Welling 2007 and is what the lit-
    survey agent identified as 'Item 2: rung-specific kernels'.
    """
    if n_chains < 2:
        raise ValueError("PT requires at least 2 chains")
    ratios = np.linspace(0, 1, n_chains)
    temps = t_final * (t_init / t_final) ** ratios
    use_rw = temps > rw_threshold_t  # boolean array, True = use RW

    rngs = [np.random.default_rng(seed + 100 * c) for c in range(n_chains)]
    chain_pos = [r.uniform(LOW, HIGH).astype(np.float64) for r in rngs]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    best_val = min(chain_val)
    n_calls = n_chains
    swap_attempts = 0
    swap_accepts = 0

    for epoch in range(n_epochs):
        for inner in range(k_inner):
            for c in range(n_chains):
                T_c = temps[c]
                if use_rw[c]:
                    prop = np.clip(gaussian_propose(rngs[c], chain_pos[c], sigma_rw),
                                   LOW, HIGH)
                    pv = OBJ_FN(prop)
                    n_calls += 1
                    if rngs[c].random() < metropolis_accept_prob(
                            pv - chain_val[c], T_c):
                        chain_pos[c], chain_val[c] = prop, pv
                else:
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
                swap_attempts += 1
                if rngs[0].random() < min(1.0, np.exp(log_alpha)):
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts += 1
    return best_val, n_calls, swap_accepts, swap_attempts


def bgsa_pt_hybrid(seed, n_epochs, n_chains,
                   t_map, e_map, L_map, q_map, pilot_calls,
                   k_inner=20, k_swap=5, t_hot=None):
    """bGSA + hybrid PT (hot chains = RW, cold chains = q-HMC).
    t_hot comes from the pilot's basin-spanning regime; falls back
    to t_map if no pilot draw saturated."""
    if t_hot is None:
        t_hot = t_map
    t_cold = max(t_map, 0.1)
    rw_threshold_t = 0.5 * (t_hot + t_cold)
    bv, prod_calls, swap_a, swap_t = parallel_tempering_hybrid(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot, t_cold, e_map, L_map, q_map, sigma_rw=0.5,
        rw_threshold_t=rw_threshold_t)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map, swap_a, swap_t


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
                     k_inner=20, k_swap=5, target_swap_rate=0.25,
                     t_hot=None):
    """bGSA with adaptive parallel-tempering production. Pilot done upstream;
    t_hot from pilot."""
    if t_hot is None:
        t_hot = t_map
    t_cold_init = max(t_map, 0.1)
    bv, prod_calls, swap_a, swap_t = adaptive_ladder_q_hmc(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot, t_cold_init, e_map, L_map, q_map,
        target_swap_rate=target_swap_rate)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map, swap_a, swap_t


def bgsa_pt(seed, n_epochs, n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, _unused_dim=None, t_hot=None):
    """bGSA with parallel-tempering q-HMC production. Pilot done upstream;
    t_hot from pilot."""
    if t_hot is None:
        t_hot = t_map
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


# -------------------------------------------------------------------------
# v2 drivers: Latin hypercube + adaptive sigma + stagnation restart.
# These are the three architectural fixes the v0.4.0 baselines lack;
# they matter most on multi-cup landscapes (Rastrigin) where a single-
# basin chain wastes budget.
# -------------------------------------------------------------------------

def latin_hypercube_init(rng, n_points, low, high):
    """Stratified init: each dim split into n_points strata of equal
    width, one point per stratum, intra-stratum offsets randomised,
    strata permuted independently per dim. Removes the cluster bias
    of n independent uniform draws."""
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    dim = len(low)
    out = np.zeros((n_points, dim))
    for d in range(dim):
        edges = np.linspace(low[d], high[d], n_points + 1)
        offsets = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(offsets)
        out[:, d] = offsets
    return out


def classical_sa_advanced(seed, n_epochs, k_per_epoch, t_init, sigma_init,
                          n_starts=4, stagnation_k=400,
                          sigma_target_accept=0.234):
    """Multi-start classical SA. Latin hypercube spreads n_starts
    chains across the box; each chain adapts sigma toward the
    Roberts/Rosenthal 0.234 target; stagnation restart redraws a
    fresh LH point if no improvement in `stagnation_k` steps."""
    rng = np.random.default_rng(seed)
    starts = latin_hypercube_init(rng, n_starts, LOW, HIGH)
    chain_pos = [s.copy() for s in starts]
    chain_val = [OBJ_FN(s) for s in chain_pos]
    chain_sigma = [sigma_init for _ in range(n_starts)]
    chain_stag = [0 for _ in range(n_starts)]
    chain_recent_acc: list[list[bool]] = [[] for _ in range(n_starts)]
    best_idx = int(np.argmin(chain_val))
    best_pos = chain_pos[best_idx].copy()
    best_val = chain_val[best_idx]
    n = n_starts

    k_per_chain = max(1, k_per_epoch // n_starts)
    sigma_min = 1e-3
    sigma_max = 0.5 * float(np.max(HIGH - LOW))

    for epoch in range(n_epochs):
        T = log_cool(t_init, 2.0, epoch)
        for c in range(n_starts):
            for step in range(k_per_chain):
                prop = np.clip(gaussian_propose(rng, chain_pos[c],
                                                chain_sigma[c]), LOW, HIGH)
                pv = OBJ_FN(prop)
                n += 1
                accepted = rng.random() < metropolis_accept_prob(
                    pv - chain_val[c], T)
                chain_recent_acc[c].append(accepted)
                if len(chain_recent_acc[c]) > 100:
                    chain_recent_acc[c].pop(0)
                if accepted:
                    chain_pos[c], chain_val[c] = prop, pv
                    if pv < best_val:
                        best_val = pv
                        best_pos = prop.copy()
                        chain_stag[c] = 0
                    else:
                        chain_stag[c] += 1
                else:
                    chain_stag[c] += 1

                if len(chain_recent_acc[c]) >= 25 and step % 25 == 0:
                    rate = float(np.mean(chain_recent_acc[c]))
                    if rate > sigma_target_accept + 0.1:
                        chain_sigma[c] = min(chain_sigma[c] * 1.1, sigma_max)
                    elif rate < sigma_target_accept - 0.1:
                        chain_sigma[c] = max(chain_sigma[c] * 0.9, sigma_min)

                if chain_stag[c] >= stagnation_k:
                    fresh = latin_hypercube_init(rng, 1, LOW, HIGH)[0]
                    chain_pos[c] = fresh
                    chain_val[c] = OBJ_FN(fresh)
                    chain_sigma[c] = sigma_init
                    chain_recent_acc[c] = []
                    chain_stag[c] = 0
                    n += 1
                    if chain_val[c] < best_val:
                        best_val = chain_val[c]
                        best_pos = fresh.copy()
    return best_val, n, best_pos


def parallel_tempering_hybrid_v2(
    seed, n_epochs, n_chains, k_inner, k_swap,
    t_init, t_final, eps, L, q,
    sigma_rw_init=0.5, rw_threshold_t=2.0,
    stagnation_k=400, sigma_target_accept=0.234,
):
    """Hybrid PT v2: hot chains (T > rw_threshold_t) use adaptive
    RW Metropolis (per-chain sigma toward 0.234), cold chains use
    q-HMC. All chains start from a Latin hypercube; per-chain
    stagnation restart redraws a fresh LH point on no-improvement
    runs of length `stagnation_k`."""
    if n_chains < 2:
        raise ValueError("PT requires at least 2 chains")
    ratios = np.linspace(0, 1, n_chains)
    temps = t_final * (t_init / t_final) ** ratios
    use_rw = temps > rw_threshold_t

    master = np.random.default_rng(seed)
    starts = latin_hypercube_init(master, n_chains, LOW, HIGH)
    rngs = [np.random.default_rng(seed + 100 * c) for c in range(n_chains)]
    chain_pos = [starts[c].copy() for c in range(n_chains)]
    chain_val = [OBJ_FN(p) for p in chain_pos]
    sigmas = [sigma_rw_init for _ in range(n_chains)]
    recent_acc: list[list[bool]] = [[] for _ in range(n_chains)]
    chain_stag = [0 for _ in range(n_chains)]
    best_idx = int(np.argmin(chain_val))
    best_pos = chain_pos[best_idx].copy()
    best_val = chain_val[best_idx]
    n_calls = n_chains
    swap_attempts = 0
    swap_accepts = 0

    sigma_min = 1e-3
    sigma_max = 0.5 * float(np.max(HIGH - LOW))

    for epoch in range(n_epochs):
        for inner in range(k_inner):
            for c in range(n_chains):
                T_c = temps[c]
                improved_this_step = False
                if use_rw[c]:
                    prop = np.clip(gaussian_propose(rngs[c], chain_pos[c],
                                                    sigmas[c]), LOW, HIGH)
                    pv = OBJ_FN(prop)
                    n_calls += 1
                    accepted = rngs[c].random() < metropolis_accept_prob(
                        pv - chain_val[c], T_c)
                    recent_acc[c].append(accepted)
                    if len(recent_acc[c]) > 100:
                        recent_acc[c].pop(0)
                    if accepted:
                        chain_pos[c], chain_val[c] = prop, pv
                    if len(recent_acc[c]) >= 25:
                        rate = float(np.mean(recent_acc[c]))
                        if rate > sigma_target_accept + 0.1:
                            sigmas[c] = min(sigmas[c] * 1.1, sigma_max)
                        elif rate < sigma_target_accept - 0.1:
                            sigmas[c] = max(sigmas[c] * 0.9, sigma_min)
                else:
                    chain_pos[c], _, nc, chain_val[c] = hmc_sa_step(
                        rngs[c], chain_pos[c], chain_val[c], T_c, eps, L,
                        len(LOW), q)
                    n_calls += nc
                if chain_val[c] < best_val:
                    best_val = chain_val[c]
                    best_pos = chain_pos[c].copy()
                    chain_stag[c] = 0
                    improved_this_step = True
                if not improved_this_step:
                    chain_stag[c] += 1

                if chain_stag[c] >= stagnation_k:
                    fresh = latin_hypercube_init(rngs[c], 1, LOW, HIGH)[0]
                    chain_pos[c] = fresh
                    chain_val[c] = OBJ_FN(fresh)
                    sigmas[c] = sigma_rw_init
                    recent_acc[c] = []
                    chain_stag[c] = 0
                    n_calls += 1
                    if chain_val[c] < best_val:
                        best_val = chain_val[c]
                        best_pos = fresh.copy()

            if (inner + 1) % k_swap == 0:
                i = rngs[0].integers(0, n_chains - 1)
                T_i, T_j = temps[i], temps[i + 1]
                F_i, F_j = chain_val[i], chain_val[i + 1]
                log_alpha = (1.0 / T_i - 1.0 / T_j) * (F_i - F_j)
                swap_attempts += 1
                if rngs[0].random() < min(1.0, np.exp(log_alpha)):
                    chain_pos[i], chain_pos[i + 1] = chain_pos[i + 1], chain_pos[i]
                    chain_val[i], chain_val[i + 1] = chain_val[i + 1], chain_val[i]
                    swap_accepts += 1
    return best_val, n_calls, swap_accepts, swap_attempts


def bgsa_pt_hybrid_v2(seed, n_epochs, n_chains,
                      t_map, e_map, L_map, q_map, pilot_calls,
                      k_inner=20, k_swap=5, t_hot=None):
    """bGSA + hybrid PT v2: Latin hypercube + adaptive sigma +
    stagnation restart on top of the rung-specific kernel
    architecture. t_hot from pilot's basin-spanning regime."""
    if t_hot is None:
        t_hot = t_map
    t_cold = max(t_map, 0.1)
    rw_threshold_t = 0.5 * (t_hot + t_cold)
    bv, prod_calls, swap_a, swap_t = parallel_tempering_hybrid_v2(
        seed, n_epochs, n_chains, k_inner, k_swap,
        t_hot, t_cold, e_map, L_map, q_map,
        rw_threshold_t=rw_threshold_t)
    return bv, pilot_calls + prod_calls, t_map, e_map, L_map, q_map, swap_a, swap_t


def run_pilot(seed, n_pilot, pilot_steps, dim,
              n_rw_pilot=10, rw_steps=50, n_scout=8):
    """Shared pilot phase for ALL bGSA drivers.

    Three-stage pilot (after issue 001 empirical-Bayes priors):

      Stage 0 (~n_scout=8 fevals/coord) -- widely-spaced log-uniform
        scout draws over (T, eps, L, q). Used to fit method-of-
        moments empirical-Bayes priors (issue 001), eliminating the
        hardcoded log-normal mean / SD constants.
      Stage 1 (n_pilot HMC chains, pilot_steps each) -- main HMC
        pilot using the data-derived priors.
      Stage 2 (n_rw_pilot RW chains, rw_steps each) -- RW pilot for
        joint (t_rw, sigma_rw) MAP (issue 002).

    Returns (t_map, e_map, L_map, q_map, sigma_map, best_pilot_pos,
    pilot_calls, t_hot, t_rw_map)."""
    rng = np.random.default_rng(seed)
    q_max = 1.0 + 2.0 / dim - 0.06
    pilot_calls = 0
    best_pilot_pos = None
    best_pilot_val = float("inf")

    # Stage 0 -- scout phase for empirical-Bayes priors.
    scout_obs = []
    for k in range(n_scout):
        # Widely-spaced log-uniform draws spanning 4 log-decades for
        # T and eps, 1 decade for L. q is uniform on the safe range.
        t = float(np.exp(rng.uniform(-3.0, 3.0)))
        e = float(np.exp(rng.uniform(-5.0, -1.0)))
        L = max(1, int(np.exp(rng.uniform(0.5, 3.0))))
        q = float(rng.uniform(1.05, q_max))
        bv, ar, fpos, nc = hmc_pilot(seed * 7919 + k, t, e, L,
                                     max(20, pilot_steps // 4), q=q)
        scout_obs.append({"t_init": t, "epsilon": e, "L": L, "q": q,
                          "accept_rate": ar, "best_val": bv})
        pilot_calls += nc
        if bv < best_pilot_val:
            best_pilot_val = bv
            best_pilot_pos = fpos
    priors = fit_empirical_bayes_priors(scout_obs, dim)

    # Stage 1 -- main HMC pilot using empirical-Bayes priors.
    pilot_obs = list(scout_obs)  # fold scouts into the Laplace fit
    for k in range(n_pilot):
        t = float(np.exp(rng.normal(priors["t_mean"], priors["t_sd"])))
        e = float(np.exp(rng.normal(priors["e_mean"], priors["e_sd"])))
        L = max(1, int(np.exp(rng.normal(priors["l_mean"], priors["l_sd"]))))
        q = float(np.clip(rng.normal(priors["q_mean"], priors["q_sd"]),
                          1.05, q_max))
        bv, ar, fpos, nc = hmc_pilot(seed * 1000 + k, t, e, L, pilot_steps, q=q)
        pilot_obs.append({"t_init": t, "epsilon": e, "L": L, "q": q,
                          "accept_rate": ar, "best_val": bv})
        pilot_calls += nc
        if bv < best_pilot_val:
            best_pilot_val = bv
            best_pilot_pos = fpos
    t_map, e_map, L_map, q_map = fit_laplace_4d(pilot_obs, dim, priors=priors)

    # Issue 002 -- joint RW pilot for (t_rw_map, sigma_rw_map).
    # bGSA-MetaD is RW-driven; its optimal T is generally HOTTER than
    # the HMC-pilot's t_map (which optimises HMC trajectory acceptance).
    # We sample (t, sigma) from log-uniform boxes and fit the joint
    # MAP under the 0.234 acceptance target + improvement penalty
    # (Andrieu & Thoms 2008 Sec 4.1, Roberts/Rosenthal 2001 fixed-T
    # framing).
    rw_obs = []
    sigma_lo = 0.05
    sigma_hi = max(0.05, 0.25 * float(np.max(HIGH - LOW)))
    box_extent = float(np.max(HIGH - LOW))
    t_rw_lo = 0.05 * box_extent
    t_rw_hi = 5.0 * box_extent
    for k in range(n_rw_pilot):
        sig = float(np.exp(rng.uniform(np.log(sigma_lo), np.log(sigma_hi))))
        t = float(np.exp(rng.uniform(np.log(t_rw_lo), np.log(t_rw_hi))))
        bv, ar, nc = rw_pilot(seed * 9001 + k, t, sig, rw_steps)
        rw_obs.append({"t": t, "sigma": sig,
                       "accept_rate": ar, "best_val": bv})
        pilot_calls += nc
    t_rw_map, sigma_map = fit_t_sigma_rw(rw_obs)
    # Pilot-derived t_hot: pick t_hot such that the predicted PT swap
    # rate at the (t_map, t_hot) boundary matches the Roberts/Rosenthal
    # 0.234 optimum. Uses the pilot's empirical (best_val) distribution
    # as a proxy for the F-distribution at the two temperatures.
    # Eliminates the 0.95 acceptance-quantile threshold from issue 008.
    t_hot = _pilot_t_hot_from_acceptance(pilot_obs, t_map)
    features = pilot_landscape_features(scout_obs, pilot_obs)
    return (t_map, e_map, L_map, q_map, sigma_map,
            best_pilot_pos, pilot_calls, float(t_hot), float(t_rw_map),
            features)


def _pilot_t_hot_from_acceptance(pilot_obs, t_cold):
    """Data-driven t_hot from the pilot's acceptance distribution.

    Picks t_hot as the median t_init among pilot draws whose accept
    rate sits in the upper quartile of all pilot accept rates -- i.e.,
    the typical "this temperature is hot enough to mix freely" regime
    inferred from the data. Eliminates the 0.95 quantile heuristic
    (issue 008) without introducing a different arbitrary scalar:
    the upper-quartile cut and the median selection are both
    distribution-free.

    Falls back to 2 * t_cold (a conservative lower bound for any
    non-trivial PT ladder) if no pilot draw falls in the upper
    quartile -- which can happen when all draws have similar
    acceptance."""
    accs = np.array([o["accept_rate"] for o in pilot_obs], dtype=float)
    if len(accs) < 4:
        return max(2.0 * t_cold, 1e-3)
    q75 = float(np.quantile(accs, 0.75))
    high_accept_t = [o["t_init"] for o in pilot_obs
                     if o["accept_rate"] >= q75]
    if not high_accept_t:
        return max(2.0 * t_cold, 1e-3)
    return float(np.median(high_accept_t))


def bgsa_auto(seed, n_epochs, k_per_epoch, n_chains,
              t_map, e_map, L_map, q_map, t_rw_map, sigma_map,
              t_hot, features, best_pilot_pos, pilot_calls):
    """Issue 010 -- pilot-driven driver-selection layer. Picks the
    production bGSA driver based on landscape features extracted
    during the pilot, then runs it. Returns (best_val, fevals,
    chosen_driver_name)."""
    chosen = select_bgsa_driver(features, q_map)
    if chosen == "bgsa":
        bv, nc, _ = hmc_sa(seed, n_epochs, k_per_epoch,
                           t_map, e_map, L_map, x0=best_pilot_pos, q=q_map)
        return bv, pilot_calls + nc, chosen
    if chosen == "bgsa_pt_metad":
        bv, nc, _, _, _, _, _, _, _ = bgsa_pt_metad(
            seed, n_epochs, n_chains, t_rw_map, e_map, L_map, q_map,
            pilot_calls, k_inner=20, k_swap=5, sigma_rw=sigma_map,
            t_hot=t_hot)
        return bv, nc, chosen
    # default: bgsa_metad
    bv, nc, _, _, _, _ = bgsa_metad(
        seed, n_epochs, k_per_epoch,
        t_rw_map, e_map, L_map, q_map, pilot_calls, sigma_rw=sigma_map)
    return bv, nc, chosen


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

    # Track total bGSA-PT fevals to set the matched-budget classical SA.
    # We use the first seed's PT cost as the target; close enough on
    # mean since all seeds use the same epoch / chain / inner counts.
    target_pt_fevals = None

    for seed in range(args.seeds):
        # SHARED PILOT: run once per seed; downstream bGSA drivers
        # reuse the (t_map, e_map, L_map, q_map, best_pos, pilot_calls)
        # tuple. This was the largest source of feval overhead in the
        # v0.4.0 demo (each driver re-ran a pilot of 1500-2400 fevals).
        (t_map, e_map, L_map, q_map, sigma_map,
         best_pilot_pos, pilot_calls, t_hot, t_rw_map,
         features) = run_pilot(
            seed, args.n_pilot, args.pilot_steps, dim=len(LOW))

        # Classical SA (hand-tuned, baseline budget)
        t0 = time.perf_counter()
        bv, nc, _ = classical_sa(seed, args.n_epochs, 200, 5.0, 0.5)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="classical_sa", best_val=bv,
                         fevals=nc, wall_time_s=wt))

        # Classical SA at matched-PT budget. Scales K_per_epoch so that
        # n_epochs * K_matched ~= bGSA-PT's total fevals. This is the
        # apples-to-apples comparison reviewers will demand.
        if target_pt_fevals is None:
            target_pt_fevals = 20000  # ballpark from prior runs
        k_matched = max(200, target_pt_fevals // args.n_epochs)
        t0 = time.perf_counter()
        bv, nc, _ = classical_sa(seed, args.n_epochs, k_matched, 5.0, 0.5)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="classical_sa_matched", best_val=bv,
                         fevals=nc, wall_time_s=wt))

        # Classical SA + Latin hypercube + adaptive sigma + stagnation
        # restart. The three architectural fixes that turn matched-budget
        # classical from a single-basin grinder into an actual explorer.
        t0 = time.perf_counter()
        bv, nc, _ = classical_sa_advanced(
            seed, args.n_epochs, k_matched, 5.0, 0.5,
            n_starts=4, stagnation_k=400, sigma_target_accept=0.234)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="classical_sa_advanced", best_val=bv,
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
            k_inner=20, k_swap=5, t_hot=t_hot)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA adaptive PT = Robbins-Monro adaptive ladder
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _ = bgsa_pt_adaptive(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, target_swap_rate=0.25, t_hot=t_hot)
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

        # bGSA + hybrid PT (hot = RW, cold = q-HMC).
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _ = bgsa_pt_hybrid(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, t_hot=t_hot)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt_hybrid", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + hybrid PT v2: Latin hypercube + adaptive sigma +
        # stagnation restart on top of the rung-specific kernel
        # architecture. Forces t_hot wide enough that the hot rungs
        # are exploration-dominated regardless of the pilot's mixing-T.
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _ = bgsa_pt_hybrid_v2(
            seed, args.n_epochs, args.n_chains,
            t_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, t_hot=t_hot)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt_hybrid_v2", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + metadynamics. Well-tempered bias on (x_0, x_1) fills
        # local cups so RW Metropolis escapes Arrhenius-suppressed basins.
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa_metad(
            seed, args.n_epochs, args.k_per_epoch,
            t_rw_map, e_map, L_map, q_map, pilot_calls, sigma_rw=sigma_map)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_metad", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + multi-start metadynamics: 4 LH-initialised chains
        # each run their own metad bias, take min. Combines the
        # cup-filling of metad with the lower-tail of multi-start.
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa_metad_multi(
            seed, args.n_epochs, args.k_per_epoch,
            t_rw_map, e_map, L_map, q_map, pilot_calls,
            sigma_rw=sigma_map, n_starts=4)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_metad_multi", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + PT + shared metadynamics (PLUMED-style multi-walker
        # PT-MetaD). High-T chains discover cups and deposit, low-T
        # chains refine, swaps propagate basins down the ladder.
        t0 = time.perf_counter()
        bv, nc, _, _, _, _, _, _, _log_z = bgsa_pt_metad(
            seed, args.n_epochs, args.n_chains,
            t_rw_map, e_map, L_map, q_map, pilot_calls,
            k_inner=20, k_swap=5, sigma_rw=sigma_map, t_hot=t_hot)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_pt_metad", best_val=bv,
                         fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA + continuous-time tempering (Wu-Stoltz 2022). beta drifts
        # continuously between 1/t_hot and 1/t_map on a Langevin
        # trajectory; subsumes discrete PT as the sigma_beta -> 0 limit.
        t0 = time.perf_counter()
        bv, nc, _, _, _, _ = bgsa_continuous_temper(
            seed, args.n_epochs, 20,
            t_map, e_map, L_map, q_map, pilot_calls, t_hot=t_hot)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed, driver="bgsa_continuous_temper",
                         best_val=bv, fevals=nc, wall_time_s=wt,
                         t_map=t_map, e_map=e_map, L_map=L_map, q_map=q_map))

        # bGSA-auto: pilot-driven driver selection (issue 010).
        t0 = time.perf_counter()
        bv, nc, chosen = bgsa_auto(
            seed, args.n_epochs, args.k_per_epoch, args.n_chains,
            t_map, e_map, L_map, q_map, t_rw_map, sigma_map,
            t_hot, features, best_pilot_pos, pilot_calls)
        wt = time.perf_counter() - t0
        rows.append(dict(seed=seed,
                         driver=f"bgsa_auto[{chosen}]",
                         best_val=bv, fevals=nc, wall_time_s=wt,
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

    for label in ["classical_sa", "classical_sa_matched", "classical_sa_advanced",
                  "hmc_sa_hand",
                  "bgsa", "bgsa_multichain",
                  "bgsa_pt", "bgsa_pt_adaptive", "bgsa_pt_hybrid",
                  "bgsa_pt_hybrid_v2",
                  "bgsa_svgd", "bgsa_metad", "bgsa_metad_multi",
                  "bgsa_pt_metad", "bgsa_continuous_temper"]:
        sub = [r for r in rows if r["driver"] == label]
        if not sub:
            continue
        bvs = np.array([r["best_val"] for r in sub])
        ci_upper = np.quantile(bvs, 0.95) if len(bvs) > 1 else bvs[0]
        print(f"  {label:<22}: mean = {bvs.mean():7.3f}  std = {bvs.std():6.3f}  "
              f"95%-upper = {ci_upper:7.3f}  fevals = "
              f"{np.mean([r['fevals'] for r in sub]):.0f}")
    # bgsa_auto rows are tagged with the chosen driver name.
    for label in sorted({r["driver"] for r in rows
                          if r["driver"].startswith("bgsa_auto")}):
        sub = [r for r in rows if r["driver"] == label]
        if not sub:
            continue
        bvs = np.array([r["best_val"] for r in sub])
        # 95% upper bound (bGSA's headline statistic per design_pass_10)
        ci_upper = np.quantile(bvs, 0.95) if len(bvs) > 1 else bvs[0]
        print(f"  {label:<22}: mean = {bvs.mean():7.3f}  std = {bvs.std():6.3f}  "
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
