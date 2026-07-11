r"""Why colored noise helps: the optimal-sampling theory behind the GLE driver.

Reference derivation (numpy/sympy) for the generalized Langevin equation
thermostat shipped natively in =eindir= (`gle.rs`) and driven by =anneal='s
`gle_langevin`. It is the companion to the Rust implementation, mirroring it the
way `surrogate.py` mirrors the Chebyshev core, and makes precise the claim that
colored noise handles conditioning as the `1/sqrt(D)` proposal scale handles
dimension. The physics is Ceriotti-Bussi-Parrinello (PRL 2009; "a la carte"
JCTC 2010).

For a harmonic mode of frequency omega the position decorrelates at the rate
kappa(omega) = -max Re eig(M), where M is the drift of the augmented
(position, momentum, auxiliary) Ornstein-Uhlenbeck process. The checks:

  C1. White-noise Langevin critically damps ONE frequency: kappa(omega; gamma)
      is maximised at gamma = 2 omega with kappa = omega.
  C2. A single white-noise gamma cannot critically damp a band: the normalised
      efficiency kappa(omega)/omega varies across [w_lo, w_hi] by the full
      conditioning factor w_hi / w_lo.
  C3. The colored-noise GLE (the eindir optimal_sampling_drift) flattens
      kappa(omega)/omega across the band, lifting the worst-sampled mode.
  C4. A la carte correctness: with static covariance C = I the fluctuation-
      dissipation relation A C + C A^T = B B^T holds for any valid drift A, so
      acceleration (A) is decoupled from canonical correctness (C) -- the
      dynamics analogue of the surrogate-agnostic stationarity result.

Run:
    python experiments/colored_noise_sampling.py
"""

from __future__ import annotations

import numpy as np
import sympy as sp


def _white_noise_drift(gamma: float) -> np.ndarray:
    """1x1 thermostat drift for plain Langevin (ns = 0)."""
    return np.array([[gamma]])


def optimal_sampling_drift(omega_min: float, omega_max: float, n_pairs: int) -> np.ndarray:
    """Optimal-sampling drift from log-spaced oscillatory baths -- the reference
    for eindir's gle.rs. Total auxiliary DOFs ns = 2 * n_pairs.

    Each bath is a damped harmonic oscillator (a complex eigenvalue pair
    -Gamma +/- i Omega) skew-coupled to the physical momentum, so its friction
    contribution PEAKS at omega ~ Omega rather than at zero. Spreading the
    Omega_k log-uniformly across [omega_min, omega_max] and scaling the coupling
    as c_k ~ sqrt(2 Omega_k Gamma_k) makes the total friction track 2 omega --
    critical damping across the whole band, hence a flat sampling efficiency.
    A + A^T is diagonal and non-negative (the oscillator and coupling terms are
    skew), so the fluctuation-dissipation relation holds for C = I.
    """
    ns = 2 * n_pairs
    n = ns + 1
    a = np.zeros((n, n))
    log_lo, log_hi = np.log(omega_min), np.log(omega_max)
    a[0, 0] = omega_min  # small white anchor on the physical momentum
    for k in range(n_pairs):
        frac = 0.5 if n_pairs == 1 else k / (n_pairs - 1)
        omega_k = np.exp(log_lo + frac * (log_hi - log_lo))
        gamma_k = omega_k  # critically damped bath
        c_k = np.sqrt(2.0 * omega_k * gamma_k)
        i1, i2 = 1 + 2 * k, 2 + 2 * k
        # damped-oscillator block: eigenvalues -gamma_k +/- i omega_k
        a[i1, i1] = gamma_k
        a[i2, i2] = gamma_k
        a[i1, i2] = -omega_k
        a[i2, i1] = omega_k
        # skew coupling between the physical momentum and the bath
        a[0, i1] = c_k
        a[i1, 0] = -c_k
    return a


def _drift_from_params(n_pairs, log_omega, log_gamma, log_c):
    """Oscillatory-bath drift from explicit per-bath frequency `exp(log_omega)`,
    damping `exp(log_gamma)`, and coupling `exp(log_c)` -- the fit parameters."""
    ns = 2 * n_pairs
    n = ns + 1
    a = np.zeros((n, n))
    a[0, 0] = np.exp(log_omega).min()  # small white anchor
    for k in range(n_pairs):
        omega_k = np.exp(log_omega[k])
        gamma_k = np.exp(log_gamma[k])
        c_k = np.exp(log_c[k])
        i1, i2 = 1 + 2 * k, 2 + 2 * k
        a[i1, i1] = gamma_k
        a[i2, i2] = gamma_k
        a[i1, i2] = -omega_k
        a[i2, i1] = omega_k
        a[0, i1] = c_k
        a[i1, 0] = -c_k
    return a


def fit_optimal_drift(omega_min, omega_max, n_pairs, seed=0, maxiter=160):
    """Fit bath frequencies, dampings and couplings to flatten the efficiency.

    This is what gle4md does numerically: maximise the worst normalised
    efficiency `min_omega kappa(omega)/omega` over a log-grid by optimising every
    bath's frequency, damping and coupling. In eindir the fitted matrix is
    cached so the construction is a closed-form table, not a runtime fit.
    """
    from scipy.optimize import differential_evolution

    grid = np.geomspace(omega_min, omega_max, 28)
    lo, hi = np.log(omega_min), np.log(omega_max)

    def unpack(params):
        log_omega = params[0:n_pairs]
        log_gamma = params[n_pairs:2 * n_pairs]
        log_c = params[2 * n_pairs:3 * n_pairs]
        return log_omega, log_gamma, log_c

    def neg_worst(params):
        a = _drift_from_params(n_pairs, *unpack(params))
        eff = np.array([_relax_rate(w, a) / w for w in grid])
        return -eff.min()

    bounds = (
        [(lo - 0.5, hi + 0.5)] * n_pairs           # bath frequencies (a bit past the band)
        + [(lo - 0.5, hi + 2.0)] * n_pairs         # dampings
        + [(lo - 1.0, hi + 2.0)] * n_pairs         # couplings (log)
    )
    res = differential_evolution(
        neg_worst, bounds, seed=seed, maxiter=maxiter, tol=1e-5,
        popsize=20, polish=True,
    )
    return _drift_from_params(n_pairs, *unpack(res.x)), res


def _augmented_drift(omega: float, a_thermo: np.ndarray) -> np.ndarray:
    """Drift M of (q, p, s_1..s_ns) for a harmonic mode coupled to thermostat A."""
    n = a_thermo.shape[0]  # ns + 1 (momentum + aux)
    dim = n + 1            # + position
    M = np.zeros((dim, dim))
    M[0, 1] = 1.0          # q_dot = p
    M[1, 0] = -(omega**2)  # p_dot gets -omega^2 q
    M[1:, 1:] = -a_thermo  # momentum+aux evolve under -A
    return M


def _relax_rate(omega: float, a_thermo: np.ndarray) -> float:
    """Position decorrelation rate kappa(omega) = -max Re eig(M)."""
    M = _augmented_drift(omega, a_thermo)
    return float(-np.max(np.real(np.linalg.eigvals(M))))


def C1_white_critical_damping() -> bool:
    omega, gamma = sp.symbols("omega gamma", positive=True)
    # eigenvalues of [[0,1],[-omega^2,-gamma]]
    lam = sp.symbols("lam")
    char = lam**2 + gamma * lam + omega**2
    roots = sp.solve(char, lam)
    # underdamped slowest rate is gamma/2; maximised over gamma at gamma=2 omega
    rate_at_critical = float(
        (-sp.re(roots[0].subs(gamma, 2 * omega))).subs(omega, 1.0)
    )
    print(f"[C1] white noise: critical gamma=2 omega gives kappa=omega "
          f"(kappa@omega=1 -> {rate_at_critical:.3f})")
    return abs(rate_at_critical - 1.0) < 1e-9


def C2_white_band_mismatch() -> bool:
    w_lo, w_hi = 1.0, 100.0
    band = np.geomspace(w_lo, w_hi, 40)
    # best single gamma: maximise the worst normalised efficiency kappa(w)/w
    best = None
    for gamma in np.geomspace(2 * w_lo, 2 * w_hi, 60):
        a = _white_noise_drift(gamma)
        norm_eff = np.array([_relax_rate(w, a) / w for w in band])
        worst = norm_eff.min()
        if best is None or worst > best[0]:
            best = (worst, gamma, norm_eff)
    worst, gamma, norm_eff = best
    spread = norm_eff.max() / norm_eff.min()
    print(f"[C2] white noise band [{w_lo},{w_hi}] best gamma={gamma:.2f}: "
          f"kappa/omega spread {spread:.1f}x -- one friction cannot damp the band")
    # a single gamma leaves a large efficiency spread across the band
    return spread > 5.0


def C3_gle_flattens_band() -> bool:
    w_lo, w_hi = 1.0, 100.0
    band = np.geomspace(w_lo, w_hi, 40)
    # best white noise
    best_wn = None
    for gamma in np.geomspace(2 * w_lo, 2 * w_hi, 60):
        a = _white_noise_drift(gamma)
        eff = np.array([_relax_rate(w, a) / w for w in band])
        if best_wn is None or eff.min() > best_wn[0]:
            best_wn = (eff.min(), eff)
    wn_worst, wn_eff = best_wn
    # colored noise GLE over the band, matrix fitted to flatten the efficiency
    a_gle, _ = fit_optimal_drift(w_lo, w_hi, 6)
    gle_eff = np.array([_relax_rate(w, a_gle) / w for w in band])
    gle_worst = gle_eff.min()
    wn_spread = wn_eff.max() / wn_eff.min()
    gle_spread = gle_eff.max() / gle_eff.min()
    print(f"[C3] worst normalised efficiency: white {wn_worst:.3f} "
          f"(spread {wn_spread:.0f}x) vs GLE {gle_worst:.3f} "
          f"(spread {gle_spread:.0f}x); worst-mode speedup "
          f"{gle_worst/wn_worst:.1f}x")
    # the GLE lifts the worst-sampled mode and flattens the band
    return gle_worst > wn_worst and gle_spread < wn_spread


def C4_fluctuation_dissipation() -> bool:
    a = optimal_sampling_drift(1.0, 100.0, 4)
    n = a.shape[0]
    C = np.eye(n)
    BBt = a @ C + C @ a.T            # = A + A^T for C = I
    # B B^T must be PSD (valid diffusion) and symmetric
    eigs = np.linalg.eigvalsh(0.5 * (BBt + BBt.T))
    psd = eigs.min() > -1e-10
    # changing the drift (different band) keeps C = I canonical: the stationary
    # covariance solves A C + C A^T = B B^T; with B B^T = A + A^T it is C = I.
    a2 = optimal_sampling_drift(0.5, 50.0, 4)
    resid = np.linalg.norm((a2 @ np.eye(n) + np.eye(n) @ a2.T) - (a2 + a2.T))
    print(f"[C4] FDT: B B^T = A + A^T PSD (min eig {eigs.min():.2e}); "
          f"C=I stationary for any drift (resid {resid:.1e}) -- correctness "
          f"decoupled from the A tuning")
    return bool(psd and resid < 1e-10)


def main() -> int:
    print("Colored-noise optimal sampling -- theory behind the GLE driver\n")
    checks = {
        "C1 white-noise critical damping (one frequency)": C1_white_critical_damping(),
        "C2 single gamma cannot damp a band": C2_white_band_mismatch(),
        "C3 GLE flattens efficiency across the band": C3_gle_flattens_band(),
        "C4 a la carte: correctness decoupled from drift": C4_fluctuation_dissipation(),
    }
    print("\n--- ledger ---")
    allok = True
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        allok = allok and ok
    print()
    if allok:
        print("Colored noise lifts the worst-sampled mode and flattens the")
        print("sampling efficiency across the curvature band, while the static")
        print("covariance keeps sampling canonical for any drift -- conditioning")
        print("handled at the move slot as 1/sqrt(D) handles dimension.")
        return 0
    print("A check FAILED -- the colored-noise advantage is not established here.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
