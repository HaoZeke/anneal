"""GPMD design lab: D6 closed-form state-gain → shipped local temperature law.

Research method (mirrors proofs/d6_annealed_descent_scaling.py):

  1. Closed-form Gaussian partial expectations for Metropolis state gain
  2. Pairing / density-ratio argument → critical window θ ∈ (0,2)
  3. Maximize G(c,θ) over c with the closed form (not MC as design source)
  4. Emit θ⋆, α⋆(θ⋆), c⋆ for the gap-proportional / am_sa ship path

MC, when used, only cross-checks the closed form. Parent model is D6;
this module chooses the interior operating point θ⋆ = 1/2 and packages
T = θ⋆ · gap / d for the shipped driver.

Run:  PYTHONPATH=. python -m proofs.gpmd_derive
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import sympy as sp

# ---- Reuse the research engine (D6) -----------------------------------------
from proofs.d6_annealed_descent_scaling import (
    WITNESS as D6_WITNESS,
    WITNESS_ANISOTROPIC,
    WITNESS_DESCENT_LIMIT,
    WITNESS_GLOBAL_SIGN,
    WITNESS_PARTIALS,
    WITNESS_THETA_C,
    WITNESS_THETA_C_EXACT,
    alpha_curve,
    critical_theta,
    gain,
    optimize_c,
    theta_c_exactly_two_symbolic,
)

# Design operating point: interior of (0,2), ~91% residual max descent rate.
THETA_STAR = 0.5
# Rounded ship target; closed-form α*(1/2) is ~0.32.
SHIP_ALPHA_TARGET = 0.32
ALPHA_TOL = 0.02


def residual_descent_rate(theta: float = THETA_STAR) -> float:
    """G*(θ) / G*(0): fraction of pure-descent max rate kept at temperature θ."""
    _c0, g0, _a0 = optimize_c(0.0)
    _c, g, _a = optimize_c(theta)
    if g0 <= 0.0:
        return float("nan")
    return g / g0


def operating_at_theta_star(theta: float = THETA_STAR) -> tuple[float, float, float, float]:
    """Closed-form optimize: returns (c*, G*, α*, residual_rate)."""
    c_star, g_star, alpha_star = optimize_c(theta)
    rate = residual_descent_rate(theta)
    return c_star, g_star, alpha_star, rate


def algorithm_constants() -> dict[str, float]:
    """Emit constants for the shipped local law from the closed-form optimum."""
    c_star, g_star, alpha_lab, rate = operating_at_theta_star(THETA_STAR)
    return {
        "THETA_STAR": float(THETA_STAR),
        "ALPHA_TARGET_LAB": float(alpha_lab),
        "ALPHA_TARGET_SHIP": float(SHIP_ALPHA_TARGET),
        "ALPHA_TOL": float(ALPHA_TOL),
        "C_STAR_SPHERE": float(c_star),
        "G_STAR": float(g_star),
        "RESIDUAL_RATE": float(rate),
        "THETA_C": float(critical_theta()),
    }


def read_shipped_rust_constants(gpmd_rs: Path | None = None) -> dict[str, float]:
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


def mc_gain_check(c: float, theta: float, n: int = 200_000, seed: int = 0) -> float:
    """MC estimate of G — validation of closed form only, not design source."""
    import numpy as np

    rng = np.random.default_rng(seed)
    mu, s = c * c, 2.0 * c
    D = rng.normal(mu, s, size=n)
    a = np.minimum(1.0, np.exp(-D / max(theta, 1e-300)))
    return float(-np.mean(D * a))


def closed_form_matches_mc(tol: float = 0.02) -> bool:
    """Closed-form G vs MC at a few interior points (D6-style multi-point check)."""
    ok = True
    for c, th in [(1.2, 0.5), (1.0, 1.0), (1.5, 0.25)]:
        g_cf = gain(c, th)
        g_mc = mc_gain_check(c, th, n=150_000, seed=7)
        ok &= abs(g_cf - g_mc) < tol
    return ok


def density_ratio_under_m1() -> bool:
    """p(-u)/p(u)=e^{-u/2} under M1: same algebra as D6 paired_increment ratio_ok."""
    u, c = sp.symbols("u c", positive=True)
    mu = c**2
    s2 = 4 * c**2
    log_ratio = sp.simplify(-((-u - mu) ** 2 - (u - mu) ** 2) / (2 * s2))
    return sp.simplify(log_ratio + u / 2) == 0


def integrand_sign_factor() -> bool:
    """(1/θ - 1/2) = (2-θ)/(2θ); sign of pairing weight in D6 G integral."""
    theta = sp.symbols("theta", positive=True)
    expr = sp.together(1 / theta - sp.Rational(1, 2))
    return sp.simplify(expr - (2 - theta) / (2 * theta)) == 0


def small_step_gain_leading() -> bool:
    """Delegate to D6 symbolic series: G ~ c²(2-θ)/θ so θ_c=2."""
    return theta_c_exactly_two_symbolic()


def operating_constants(theta_star: float = THETA_STAR) -> tuple[float, float, float]:
    """Alias for D6 optimize_c(theta_star) → (c*, G*, α*)."""
    c, g, a = optimize_c(theta_star)
    return c, g, a


def main() -> int:
    print("GPMD design lab — D6 closed-form research path")
    print("Parent: d6_annealed_descent_scaling (local sphere state gain)")
    print("Claim boundary: local descent law, not dual annealing / CMA-ES")
    print()

    print("D6_RESEARCH_GATES (closed form / symbolic):")
    gates = [
        ("partial_expectations (quadrature vs closed form)", WITNESS_PARTIALS),
        ("descent limit Rechenberg (theta->0)", WITNESS_DESCENT_LIMIT),
        ("critical temperature bracketed ~2", WITNESS_THETA_C),
        ("theta_c=2 symbolic small-step series", WITNESS_THETA_C_EXACT),
        ("global paired-increment sign", WITNESS_GLOBAL_SIGN),
        ("anisotropic T_c formula", WITNESS_ANISOTROPIC),
        ("integrand factor 1/th-1/2=(2-th)/(2 th)", integrand_sign_factor()),
        ("closed-form G matches MC (validation)", closed_form_matches_mc()),
    ]
    ok = True
    for name, v in gates:
        print(f"  {name}: {v}")
        ok = bool(v) and ok
    print(f"  D6_WITNESS_AGGREGATE: {D6_WITNESS}")
    ok = ok and bool(D6_WITNESS)

    print()
    print("OPERATING_POINT (closed-form optimize_c):")
    print(f"  theta_c (bisection on closed-form G*): {critical_theta():.6f}")
    print(f"{'theta':>8} {'c*':>8} {'G*':>9} {'alpha*':>8} {'G*/G*(0)':>10}")
    g0 = optimize_c(0.0)[1]
    for th, c, g, a in alpha_curve([0.0, 0.25, 0.5, 1.0, 1.5]):
        rate = g / g0 if g0 > 0 else float("nan")
        print(f"{th:>8.3g} {c:>8.4f} {g:>9.5f} {a:>8.4f} {rate:>10.4f}")

    c_star, g_star, a_star, rate = operating_at_theta_star(THETA_STAR)
    print()
    print(f"  at theta_star={THETA_STAR}: c*={c_star:.6f} G*={g_star:.6f} "
          f"alpha*={a_star:.6f} residual_rate={rate:.4f}")
    ok = ok and 0.25 < a_star < 0.40 and rate > 0.85

    # Sign check via closed form (not MC-only)
    g_lo = gain(1.2, 0.5)
    g_c = gain(1.2, 2.0)
    g_hi = gain(1.2, 3.0)
    print(f"  closed-form G(1.2,0.5)={g_lo:.5f}  G(1.2,2)={g_c:.5f}  G(1.2,3)={g_hi:.5f}")
    ok = ok and g_lo > 0 and abs(g_c) < 1e-6 and g_hi < 0

    consts = algorithm_constants()
    print()
    print("ALGORITHM_CONSTANTS:")
    print(f"  THETA_STAR = {consts['THETA_STAR']}")
    print(f"  ALPHA_TARGET_LAB = {consts['ALPHA_TARGET_LAB']:.6f}")
    print(f"  ALPHA_TARGET_SHIP = {consts['ALPHA_TARGET_SHIP']}")
    print(f"  ALPHA_TOL = {consts['ALPHA_TOL']}")
    print(f"  C_STAR_SPHERE = {consts['C_STAR_SPHERE']:.6f}")
    print(f"  G_STAR = {consts['G_STAR']:.6f}")
    print(f"  RESIDUAL_RATE = {consts['RESIDUAL_RATE']:.6f}")
    print(f"  THETA_C = {consts['THETA_C']:.6f}")
    ok = ok and abs(consts["ALPHA_TARGET_LAB"] - consts["ALPHA_TARGET_SHIP"]) <= consts["ALPHA_TOL"]

    try:
        shipped = read_shipped_rust_constants()
        if shipped:
            print()
            print("SHIPPED_RUST:")
            for k, v in shipped.items():
                print(f"  {k} = {v}")
            ok = ok and shipped.get("THETA_STAR") == consts["THETA_STAR"]
            ok = ok and abs(shipped.get("ALPHA_TARGET", -1) - consts["ALPHA_TARGET_SHIP"]) < 1e-12
            print("  bind_THETA_STAR:", shipped.get("THETA_STAR") == consts["THETA_STAR"])
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
