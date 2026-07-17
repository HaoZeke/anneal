"""D11: Budget-feasible Metropolis temperature (BFWT).

Problem. D6 requires θ = T d / gap < 2 for positive expected local descent.
D7 requires T ≳ b / ln B for escape of barrier b within remaining budget B.
GPMD packages θ⋆ = 1/2 only (descent residual rate ~0.91) and ignores B,b.
Those three quantities define a window; when nonempty a single T can serve
both roles, when empty no constant T works.

Model.
  gap = max(f - f_best, ε) > 0
  T_hi = 2 * gap / d                 # D6 ceiling (θ = 2)
  T_lo = b_hat / log(B + e)          # D7 floor (e = exp(1))
  T_des = θ⋆ * gap / d               # design interior, θ⋆ = 1/2

Load-bearing law (BFWT).
  if T_lo < T_hi:   T = clamp(T_des, T_lo, T_hi)
  else:             T = T_lo         # window empty → escape temperature

Load-bearing identity.
  window nonempty  ⇔  b_hat * d < 2 * gap * log(B + e)

Checks.
  1. Symbolic: θ⋆=1/2 ⇒ T_des = (1/4) T_hi (always below ceiling; θ_des/θ_hi=1/4).
  2. Recovery: b_hat → 0 ⇒ T → T_des (GPMD recovered).
  3. Clamp algebra on numeric grid: T always in [T_lo, T_hi] when nonempty.
  4. Empty window: T equals T_lo and T_lo ≥ T_hi.
  5. Nonempty criterion matches the identity above on a numeric grid.

Emit ALGORITHM_CONSTANTS for the ship path (θ⋆, e, modes).

Run: PYTHONPATH=. python -m proofs.d11_budget_feasible_temp
"""
from __future__ import annotations

import math

import sympy as sp

THETA_STAR = 0.5
EULER_E = math.e
GAP_EPS = 1e-12


def t_hi(gap: float, dim: int) -> float:
    d = max(dim, 1)
    return 2.0 * max(gap, GAP_EPS) / d


def t_des(gap: float, dim: int, theta_star: float = THETA_STAR) -> float:
    d = max(dim, 1)
    return float(theta_star) * max(gap, GAP_EPS) / d


def t_lo(barrier: float, budget: float) -> float:
    b = max(barrier, 0.0)
    # log(B+e) ≥ 1 so T_lo ≤ b when B≥0
    return b / math.log(max(budget, 0.0) + EULER_E)


def window_nonempty(gap: float, dim: int, barrier: float, budget: float) -> bool:
    """Identity: nonempty iff b*d < 2*gap*log(B+e)."""
    d = max(dim, 1)
    g = max(gap, GAP_EPS)
    return barrier * d < 2.0 * g * math.log(max(budget, 0.0) + EULER_E)


def budget_feasible_temp(
    f: float,
    f_best: float,
    dim: int,
    budget_remaining: float,
    barrier_hat: float,
    theta_star: float = THETA_STAR,
) -> tuple[float, str]:
    """Return (T, mode) with mode in {'design','escape_floor','descent_cap','escape_forced'}."""
    gap = max(f - f_best, GAP_EPS)
    hi = t_hi(gap, dim)
    lo = t_lo(barrier_hat, budget_remaining)
    des = t_des(gap, dim, theta_star)
    if lo < hi:
        if des < lo:
            return lo, "escape_floor"
        if des > hi:
            return hi, "descent_cap"
        return des, "design"
    return lo, "escape_forced"


def design_below_ceiling_symbolic() -> bool:
    """θ⋆=1/2 ⇒ T_des = (1/4) T_hi identically (θ_des / θ_hi = θ⋆/2 = 1/4)."""
    gap, d = sp.symbols("gap d", positive=True)
    t_hi_s = 2 * gap / d
    t_des_s = sp.Rational(1, 2) * gap / d
    # T_des / T_hi = (1/2) / 2 = 1/4
    return sp.simplify(t_des_s - t_hi_s / 4) == 0


def nonempty_identity_symbolic() -> bool:
    """window_nonempty definition matches algebraic inequality."""
    gap, d, b, B = sp.symbols("gap d b B", positive=True)
    e = sp.E
    # b*d < 2*gap*log(B+e)  ⇔  b/log(B+e) < 2*gap/d  ⇔ T_lo < T_hi
    t_lo_s = b / sp.log(B + e)
    t_hi_s = 2 * gap / d
    # Cross-multiply positives: b*d < 2*gap*log(B+e)
    left = b * d
    right = 2 * gap * sp.log(B + e)
    # Check equivalence of T_lo < T_hi with left < right via simplification
    # of (t_hi - t_lo) sign vs (right - left)
    diff_T = sp.simplify(t_hi_s - t_lo_s)
    # clear positive denominators: sign(t_hi-t_lo) = sign(right-left)
    cleared = sp.simplify(sp.together(diff_T) * d * sp.log(B + e))
    # cleared should equal right - left
    return sp.simplify(cleared - (right - left)) == 0


def recovery_zero_barrier() -> bool:
    """b_hat=0 ⇒ T equals design temperature."""
    ok = True
    for gap, d, B in [(4.0, 4, 1000), (1.0, 10, 50), (100.0, 2, 1e6)]:
        T, mode = budget_feasible_temp(gap, 0.0, d, B, 0.0)
        des = t_des(gap, d)
        ok &= mode == "design" and abs(T - des) < 1e-15
    return ok


def clamp_when_nonempty() -> bool:
    ok = True
    for gap, d, B, b in [
        (4.0, 4, 1e4, 0.1),
        (10.0, 5, 1e3, 2.0),
        (1.0, 8, 1e5, 0.5),
        (20.0, 3, 500, 1.0),
    ]:
        if not window_nonempty(gap, d, b, B):
            continue
        T, mode = budget_feasible_temp(gap + 0.0, 0.0, d, B, b)
        lo, hi = t_lo(b, B), t_hi(gap, d)
        ok &= lo - 1e-12 <= T <= hi + 1e-12
        ok &= mode in ("design", "escape_floor", "descent_cap")
    return ok


def empty_window_forces_escape() -> bool:
    ok = True
    # Large barrier, tiny gap, small budget → empty window
    for gap, d, B, b in [(0.1, 50, 10, 100.0), (1.0, 100, 5, 50.0)]:
        assert not window_nonempty(gap, d, b, B)
        T, mode = budget_feasible_temp(gap, 0.0, d, B, b)
        ok &= mode == "escape_forced"
        ok &= abs(T - t_lo(b, B)) < 1e-12
        ok &= T + 1e-12 >= t_hi(gap, d)
    return ok


def nonempty_grid_matches_identity() -> bool:
    ok = True
    for gap in (0.5, 2.0, 10.0):
        for d in (2, 8, 30):
            for B in (10.0, 1e3, 1e6):
                for b in (0.01, 1.0, 5.0, 50.0):
                    wn = window_nonempty(gap, d, b, B)
                    identity = b * d < 2.0 * gap * math.log(B + EULER_E)
                    ok &= wn == identity
                    lo, hi = t_lo(b, B), t_hi(gap, d)
                    ok &= (lo < hi) == identity
    return ok


def main() -> int:
    print("D11 Budget-Feasible Window Temperature (BFWT)")
    print("Parents: D6 descent ceiling + D7 escape floor; not dual_annealing")
    print()
    checks = [
        ("design T_des = T_hi/4 symbolically (θ⋆=1/2)", design_below_ceiling_symbolic()),
        ("T_lo < T_hi ⇔ nonempty identity symbolically", nonempty_identity_symbolic()),
        ("zero barrier recovers design T (GPMD)", recovery_zero_barrier()),
        ("nonempty: T clamped into [T_lo, T_hi]", clamp_when_nonempty()),
        ("empty window: T = T_lo escape_forced", empty_window_forces_escape()),
        ("nonempty grid matches b*d < 2 gap log(B+e)", nonempty_grid_matches_identity()),
    ]
    ok = True
    for name, v in checks:
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print()
    print("ALGORITHM_CONSTANTS:")
    print(f"  THETA_STAR = {THETA_STAR}")
    print(f"  EULER_E = {EULER_E}")
    print(f"  GAP_EPS = {GAP_EPS}")
    print(f"  LAW = clamp(T_des, T_lo, T_hi) if T_lo < T_hi else T_lo")
    print(f"  T_des = THETA_STAR * gap / d")
    print(f"  T_hi  = 2 * gap / d")
    print(f"  T_lo  = barrier_hat / log(B + e)")
    # Example emission
    T, mode = budget_feasible_temp(4.0, 0.0, 4, 1000.0, 0.5)
    print(f"  example f=4,f_best=0,d=4,B=1000,b=0.5 -> T={T:.6g} mode={mode}")
    print("D11_DERIVE_OK" if ok else "D11_DERIVE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
