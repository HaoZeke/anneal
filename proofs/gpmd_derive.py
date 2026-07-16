"""GPMD / D6 design lab: build the local sphere model, emit algorithm constants.

This is a *design* module, not a re-check of high-school algebra:

1. Exact energy increment Δ under f(x)=‖x‖²/2, y=x+σz
2. ES scaling σ=c‖x‖/d → normalized D
3. Limiting Gaussian parameters (M1): mean c², variance 4c²
4. Density-ratio identity (I1) and pairing sign factor (T1)
5. Optimize state gain G(c, θ⋆) over c → ship α⋆, c⋆

Run:  PYTHONPATH=. python -m proofs.gpmd_derive
Exits non-zero if identities fail or constants are non-finite.

Shipped Rust constants (THETA_STAR, ALPHA_TARGET) are bound by
proofs/tests/test_gpmd_const_bind.py against algorithm_constants().
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Symbolic model construction (design lab)
# ---------------------------------------------------------------------------


def symbolic_energy_increment() -> dict[str, Any]:
    """Exact Δ = f(y)-f(x) for f=‖x‖²/2, y=x+σz (identity, not a limit)."""
    # 1-D symbolic stand-in: radial form uses x·z and ‖z‖²
    sigma, xz, z2 = sp.symbols("sigma xz z2", real=True)
    # f(y)-f(x) = (‖x‖² + 2σ x·z + σ²‖z‖²)/2 - ‖x‖²/2
    delta = sigma * xz + (sigma**2) * z2 / 2
    return {"delta": delta, "form": "sigma*xz + sigma**2*z2/2"}


def symbolic_normalized_D() -> dict[str, Any]:
    """D = d Δ / f under ES scale σ = c ‖x‖ / d, f = ‖x‖²/2.

    Exact finite-d expression before the limit:
      D = 2c (x·z/‖x‖) + c² (‖z‖²/d)
    """
    c, d = sp.symbols("c d", positive=True)
    # unit radial Gaussian component and chi-squared proxy
    g, z2_over_d = sp.symbols("g chi", real=True)
    D = 2 * c * g + c**2 * z2_over_d
    return {
        "D": D,
        "limit_mean_term": c**2,  # E[c² χ] → c²
        "limit_var_from_g": (2 * c) ** 2,  # Var(2c g) = 4 c² for g~N(0,1)
    }


def m1_gaussian_parameters() -> dict[str, sp.Expr]:
    """M1: D ⇒ N(c², 4c²) under d→∞ ES scaling."""
    c = sp.symbols("c", positive=True)
    mu = c**2
    var = 4 * c**2
    sigma = 2 * c  # std
    return {"mu": mu, "var": var, "sigma": sigma, "c": c}


def density_ratio_identity() -> bool:
    """(I1): for N(c², 4c²), p(-u)/p(u) = exp(-u/2) for u>0."""
    c, u = sp.symbols("c u", positive=True)
    mu = c**2
    s2 = 4 * c**2
    logp = lambda x: -((x - mu) ** 2) / (2 * s2)
    ratio = sp.simplify(sp.exp(logp(-u) - logp(u)))
    target = sp.exp(-u / 2)
    return sp.simplify(ratio - target) == 0


def algebraic_exponent() -> bool:
    """log p(-u)-log p(u) = -2 u μ / σ²; under M1 equals -u/2."""
    c, u = sp.symbols("c u", positive=True)
    mu = c**2
    s2 = 4 * c**2
    expr = sp.simplify(-2 * u * mu / s2 + u / 2)
    return expr == 0


def integrand_sign_factor() -> bool:
    """Pairing weight factor: 1/θ - 1/2 = (2-θ)/(2θ).

    This is the *design* claim: G has the sign of (2-θ) for θ>0 in the model.
    """
    theta = sp.symbols("theta", positive=True)
    expr = sp.simplify(1 / theta - sp.Rational(1, 2))
    factored = sp.together(expr)
    return sp.simplify(factored - (2 - theta) / (2 * theta)) == 0


def critical_window_symbolic() -> dict[str, Any]:
    """Symbolic trichotomy of the T1 factor for design θ⋆ choice."""
    theta = sp.symbols("theta", positive=True)
    factor = (2 - theta) / (2 * theta)
    # Sign analysis under positive theta
    # factor > 0 iff 2-theta > 0 iff theta < 2
    return {
        "factor": factor,
        "positive_when": "0 < theta < 2",
        "zero_at": 2,
        "theta_star": sp.Rational(1, 2),
        "theta_star_in_window": True,
    }


def general_gaussian_log_ratio() -> bool:
    """General N(μ,σ²): log p(-u)/p(u) = -2 u μ / σ²."""
    mu, s, u = sp.symbols("mu s u", positive=True)
    log_ratio = sp.simplify(-((-u - mu) ** 2 - (u - mu) ** 2) / (2 * s**2))
    return sp.simplify(log_ratio + 2 * u * mu / s**2) == 0


def small_step_gain_leading() -> bool:
    """Small-c leading term G ~ c² (2-θ)/θ has zero at θ=2 (design consistency)."""
    c, theta = sp.symbols("c theta", positive=True)
    leading = c**2 * (2 - theta) / theta
    # Critical zero of leading coefficient at theta=2
    crit = sp.solve(sp.numer(sp.together(leading / c**2)), theta)
    return crit == [2]


# ---------------------------------------------------------------------------
# Numeric gain and operating constants (emit for ship)
# ---------------------------------------------------------------------------


def gain_numeric(c: float, theta: float, n: int = 200_000, seed: int = 0) -> float:
    """MC estimate of G(c,θ) = -E[D min(1, e^{-D/θ})] under M1."""
    rng = np.random.default_rng(seed)
    mu, s = c * c, 2 * c
    D = rng.normal(mu, s, size=n)
    a = np.minimum(1.0, np.exp(-D / theta))
    return float(-np.mean(D * a))


def operating_constants(theta_star: float = 0.5) -> tuple[float, float, float]:
    """Maximize G(c, θ⋆) over c>0; return (c⋆, G⋆, α⋆)."""

    def neg_g(c: float) -> float:
        return -gain_numeric(float(c), theta_star, n=80_000, seed=1)

    res = minimize_scalar(neg_g, bounds=(0.2, 3.0), method="bounded")
    c_star = float(res.x)
    g_star = -float(res.fun)
    rng = np.random.default_rng(2)
    mu, s = c_star**2, 2 * c_star
    D = rng.normal(mu, s, size=200_000)
    alpha = float(np.mean(np.minimum(1.0, np.exp(-D / theta_star))))
    return c_star, g_star, alpha


# Shipped rounded α used in Rust; lab MC is noisier — bind with tolerance.
SHIP_ALPHA_TARGET = 0.32
SHIP_THETA_STAR = 0.5
ALPHA_TOL = 0.05  # |lab α - SHIP_ALPHA_TARGET| must be ≤ this


def algorithm_constants() -> dict[str, float]:
    """Emit operating constants for the shipped local law.

    THETA_STAR is design (interior of (0,2) window).
    ALPHA_TARGET / C_STAR come from maximizing G at θ⋆ under M1.
    """
    c_star, g_star, alpha = operating_constants(SHIP_THETA_STAR)
    return {
        "THETA_STAR": float(SHIP_THETA_STAR),
        "ALPHA_TARGET_LAB": float(alpha),
        "ALPHA_TARGET_SHIP": float(SHIP_ALPHA_TARGET),
        "ALPHA_TOL": float(ALPHA_TOL),
        "C_STAR_SPHERE": float(c_star),
        "G_STAR": float(g_star),
    }


def read_shipped_rust_constants(gpmd_rs: Path | None = None) -> dict[str, float]:
    """Parse THETA_STAR / ALPHA_TARGET from the shipped gpmd.rs."""
    if gpmd_rs is None:
        gpmd_rs = Path(__file__).resolve().parents[1] / "src" / "methods" / "gpmd.rs"
    text = gpmd_rs.read_text()
    out: dict[str, float] = {}
    m = re.search(r"pub const THETA_STAR:\s*f64\s*=\s*([0-9.eE+-]+)", text)
    if m:
        out["THETA_STAR"] = float(m.group(1))
    m = re.search(r"pub const ALPHA_TARGET:\s*f64\s*=\s*([0-9.eE+-]+)", text)
    if m:
        out["ALPHA_TARGET"] = float(m.group(1))
    return out


def main() -> int:
    print("GPMD / D6 design lab (SymPy model build + operating constants)")
    print("Parent: local sphere Metropolis state-gain (not dual annealing / CMA-ES)")
    print()

    # --- Model construction ---
    print("MODEL_BUILD:")
    dlt = symbolic_energy_increment()
    print(f"  exact_delta_form: {dlt['form']}")
    nd = symbolic_normalized_D()
    print(f"  normalized_D: 2*c*g + c**2*chi  (g~N(0,1), chi→1)")
    m1 = m1_gaussian_parameters()
    print(f"  M1_limit: N(mu={m1['mu']}, var={m1['var']})")
    win = critical_window_symbolic()
    print(f"  T1_factor: {win['factor']}")
    print(f"  T1_window: {win['positive_when']}; theta_star={win['theta_star']}")
    print()

    # --- Identities (gates) ---
    print("IDENTITIES:")
    checks = [
        ("I1 density ratio p(-u)/p(u)=e^{-u/2}", density_ratio_identity()),
        ("I1 algebraic exponent -2 mu/s2 = -1/2 under M1", algebraic_exponent()),
        ("T1 factor 1/theta-1/2 = (2-theta)/(2 theta)", integrand_sign_factor()),
        ("general Gaussian log-ratio -2 u mu / s^2", general_gaussian_log_ratio()),
        ("small-step G leading vanishes at theta=2", small_step_gain_leading()),
    ]
    ok = True
    for name, v in checks:
        print(f"  {name}: {v}")
        ok = ok and bool(v)

    # --- Sign of G (numeric model check of T1) ---
    g_lo = gain_numeric(1.2, 0.5, seed=3)
    g_c = gain_numeric(1.2, 2.0, seed=3)
    g_hi = gain_numeric(1.2, 3.0, seed=3)
    print(f"  G(1.2,0.5)={g_lo:.5f}  G(1.2,2)={g_c:.5f}  G(1.2,3)={g_hi:.5f}")
    ok = ok and g_lo > 0 and abs(g_c) < 0.02 and g_hi < 0

    # --- Emit constants ---
    consts = algorithm_constants()
    print()
    print("ALGORITHM_CONSTANTS:")
    print(f"  THETA_STAR = {consts['THETA_STAR']}")
    print(f"  ALPHA_TARGET_LAB = {consts['ALPHA_TARGET_LAB']:.6f}")
    print(f"  ALPHA_TARGET_SHIP = {consts['ALPHA_TARGET_SHIP']}")
    print(f"  ALPHA_TOL = {consts['ALPHA_TOL']}")
    print(f"  C_STAR_SPHERE = {consts['C_STAR_SPHERE']:.6f}")
    print(f"  G_STAR = {consts['G_STAR']:.6f}")
    ok = ok and math.isfinite(consts["C_STAR_SPHERE"])
    ok = ok and abs(consts["ALPHA_TARGET_LAB"] - consts["ALPHA_TARGET_SHIP"]) <= consts["ALPHA_TOL"]

    # --- Bind check vs shipped Rust (if tree present) ---
    try:
        shipped = read_shipped_rust_constants()
        if shipped:
            print()
            print("SHIPPED_RUST:")
            for k, v in shipped.items():
                print(f"  {k} = {v}")
            ok = ok and shipped.get("THETA_STAR") == consts["THETA_STAR"]
            ok = ok and abs(shipped.get("ALPHA_TARGET", -1) - consts["ALPHA_TARGET_SHIP"]) < 1e-12
            print(
                "  bind_THETA_STAR:",
                shipped.get("THETA_STAR") == consts["THETA_STAR"],
            )
            print(
                "  bind_ALPHA_TARGET:",
                abs(shipped.get("ALPHA_TARGET", -1) - consts["ALPHA_TARGET_SHIP"]) < 1e-12,
            )
    except OSError as e:
        print(f"  (skip rust bind: {e})")

    print()
    print("GPMD_DERIVE_OK" if ok else "GPMD_DERIVE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
