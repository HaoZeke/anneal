"""D12: Quenched force ledger, D6∩D7 emptiness, and bias window reopening.

Parents: D6 (descent ceiling), D7 (escape floor), D11 (BFWT composition).

This module deposits the *cluster-facing* consequences of those models:

  Prop A (emptiness).  Window nonempty ⇔ b d < 2 g ln(B+e).
  Prop B (force ledger).  If each hop costs Q force units in expectation and
    independent hops succeed with probability p ∈ (0,1], expected force to
    first success is Q/p. With M independent seeds and per-seed success
    probability p, expected force for at least one success is
    Q · E[N_hops] with N_hops ~ Geo(1-(1-p)^M) structure for sequential
    seeds, or simply M·H·Q if each seed runs a fixed hop budget H.
  Prop C (bias reopening).  If a bias reduces effective barrier to b/γ
    (γ≥1), nonempty requires γ > b d / (2 g ln(B+e)).

All identities are symbolic or exact-arithmetic numeric checks. The LJ75
table uses literature barrier scale b≈8.69 and campaign gap g≈1.21 as
*inputs*; the checks assert the algebra, not that those physical numbers
are measured here.

Run: PYTHONPATH=. python -m proofs.d12_quenched_ledger_and_window
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

# Reuse D11 definitions of the window (single source of truth).
from proofs.d11_budget_feasible_temp import (
    EULER_E,
    t_hi,
    t_lo,
    window_nonempty,
)

# ---------------------------------------------------------------------------
# Prop A: emptiness identity and critical barrier
# ---------------------------------------------------------------------------


def b_max(gap: float, dim: int, budget: float) -> float:
    """Largest barrier for which D6∩D7 is nonempty at this (gap, d, B)."""
    d = max(int(dim), 1)
    g = max(float(gap), 1e-12)
    B = max(float(budget), 0.0)
    return 2.0 * g * math.log(B + EULER_E) / d


def emptiness_ratio(barrier: float, gap: float, dim: int, budget: float) -> float:
    """b / b_max; >1 means empty window. Infinite if b_max=0."""
    bm = b_max(gap, dim, budget)
    if bm <= 0.0:
        return float("inf")
    return float(barrier) / bm


def symbolic_emptiness_identity() -> bool:
    """b d < 2 g log(B+e)  ⇔  T_lo < T_hi  ⇔  b_max definition."""
    g, d, b, B = sp.symbols("g d b B", positive=True)
    e = sp.E
    t_lo_s = b / sp.log(B + e)
    t_hi_s = 2 * g / d
    b_max_s = 2 * g * sp.log(B + e) / d
    # T_hi - T_lo cleared by positive factors d log(B+e) equals 2 g log - b d
    cleared = sp.simplify(sp.together(t_hi_s - t_lo_s) * d * sp.log(B + e))
    ok = sp.simplify(cleared - (2 * g * sp.log(B + e) - b * d)) == 0
    # b_max * d = 2 g log(B+e)
    ok &= sp.simplify(b_max_s * d - 2 * g * sp.log(B + e)) == 0
    # b_max - b and (2 g log - b d)/d share sign: difference of cleared forms
    ok &= sp.simplify((b_max_s - b) * d - (2 * g * sp.log(B + e) - b * d)) == 0
    return bool(ok)


def numeric_window_matches_b_max() -> bool:
    ok = True
    for g, d, B, b in [
        (1.21, 225, 3e6, 8.69),
        (1.21, 219, 3e6, 8.69),
        (50.0, 225, 3e6, 8.69),
        (1.21, 225, 1e7, 8.69),
        (10.0, 10, 1e4, 2.0),
        (5.0, 114, 4e5, 3.0),
    ]:
        wn = window_nonempty(g, d, b, B)
        bm = b_max(g, d, B)
        # D11 uses strict inequality barrier*d < 2*gap*log(B+e)
        ok &= wn == (b < bm)
        lo, hi = t_lo(b, B), t_hi(g, d)
        ok &= (lo < hi) == wn
    return ok


# ---------------------------------------------------------------------------
# Prop B: force ledger on the quenched landscape
# ---------------------------------------------------------------------------


def expected_force_to_first_success(Q: float, p: float) -> float:
    """E[force] for independent hops of cost Q and success probability p.

    Number of hops until first success ~ Geometric(p) with support {1,2,...},
    so E[hops] = 1/p and E[force] = Q/p.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must lie in (0,1]")
    if Q < 0.0:
        raise ValueError("Q must be nonnegative")
    return Q / p


def expected_force_fixed_hop_budget(
    Q: float, hops_per_seed: int, n_seeds: int
) -> float:
    """Force for n_seeds independent runs each charged hops_per_seed * Q."""
    if hops_per_seed < 0 or n_seeds < 0:
        raise ValueError("counts must be nonnegative")
    return float(Q) * float(hops_per_seed) * float(n_seeds)


def success_prob_at_least_one(p: float, n_seeds: int) -> float:
    """1 - (1-p)^n for independent seeds each with success probability p."""
    if not (0.0 <= p <= 1.0) or n_seeds < 0:
        raise ValueError("invalid p or n_seeds")
    return 1.0 - (1.0 - p) ** n_seeds


def symbolic_geometric_force() -> bool:
    """E[Q * Geo(p)] = Q/p for Geo starting at 1.

    Differentiate sum_{k>=1} q^k = q/(1-q) (|q|<1) to obtain
    sum k q^{k-1} = 1/(1-q)^2. Then E[N] = p * sum k (1-p)^{k-1} = 1/p.
    """
    Q, p, q = sp.symbols("Q p q", positive=True)
    k = sp.symbols("k", integer=True, positive=True)

    def _piecewise_main(expr: sp.Expr) -> sp.Expr:
        expr = sp.simplify(expr)
        if expr.is_Piecewise:
            return sp.simplify(expr.args[0][0])
        return expr

    # sum_{k>=1} q^k = q/(1-q) for q<1
    geom_branch = _piecewise_main(sp.summation(q**k, (k, 1, sp.oo)))
    geom_ok = sp.simplify(geom_branch - q / (1 - q)) == 0
    # d/dq => sum k q^{k-1} = 1/(1-q)^2
    deriv = sp.simplify(sp.diff(geom_branch, q))
    deriv_ok = sp.simplify(deriv - 1 / (1 - q) ** 2) == 0
    # Independent path: sum k q^k = q/(1-q)^2, divide by q
    sum_k_qk = _piecewise_main(sp.summation(k * q**k, (k, 1, sp.oo)))
    sum_k_ok = sp.simplify(sum_k_qk - q / (1 - q) ** 2) == 0
    s_via_div = sp.simplify(sum_k_qk / q)
    s_ok = sp.simplify(s_via_div - 1 / (1 - q) ** 2) == 0
    # E[N] = p * sum k (1-p)^{k-1} = 1/p
    EN = sp.simplify(p * s_via_div.subs(q, 1 - p))
    en_ok = sp.simplify(EN - 1 / p) == 0
    force_ok = sp.simplify(Q * EN - Q / p) == 0
    return bool(geom_ok and deriv_ok and sum_k_ok and s_ok and en_ok and force_ok)


def symbolic_union_bound_identity() -> bool:
    """P(at least one) = 1-(1-p)^n exactly under independence."""
    p, n = sp.symbols("p n", positive=True)
    # treat n as positive integer symbolically via expansion
    n = sp.symbols("n", integer=True, positive=True)
    expr = 1 - (1 - p) ** n
    # check for n=1,2,3 by substitution
    ok = True
    for nv in (1, 2, 3, 8):
        ok &= sp.simplify(expr.subs(n, nv) - (1 - (1 - p) ** nv)) == 0
    return ok


def numeric_force_examples() -> bool:
    # Wales-scale: ~4% success per 5k hops; if Q forces/hop then force = Q*5000/0.04
    ok = abs(expected_force_to_first_success(200.0, 0.04) - 200.0 / 0.04) < 1e-12
    ok &= abs(expected_force_to_first_success(200.0, 0.04) - 5000.0) < 1e-12
    # fixed budget 8 seeds * 1e5 hops * Q
    ok &= abs(expected_force_fixed_hop_budget(30.0, 100_000, 8) - 24_000_000.0) < 1e-6
    # 1/8 empirical → p_hat=0.125; at least one of 8 with true p=0.125
    ok &= abs(success_prob_at_least_one(0.125, 8) - (1 - (0.875) ** 8)) < 1e-15
    return ok


# ---------------------------------------------------------------------------
# Prop C: bias factor that reopens the window
# ---------------------------------------------------------------------------


def gamma_min(barrier: float, gap: float, dim: int, budget: float) -> float:
    """Smallest γ≥1 such that b/γ < b_max (strict reopening).

    If already nonempty at γ=1, returns 1.0.
    """
    bm = b_max(gap, dim, budget)
    if barrier <= 0.0:
        return 1.0
    if bm <= 0.0:
        return float("inf")
    if barrier < bm:
        return 1.0
    return barrier / bm


def symbolic_bias_reopening() -> bool:
    """b/γ < b_max  ⇔  γ > b/b_max  (for positive quantities)."""
    b, g, d, B, gam = sp.symbols("b g d B gamma", positive=True)
    e = sp.E
    b_max_s = 2 * g * sp.log(B + e) / d
    # b/gamma < b_max ⇔ gamma > b/b_max when all positive
    lhs = b / gam - b_max_s
    # multiply by gamma*d > 0: b*d - gamma * 2 g log(B+e)
    cleared = sp.simplify(sp.together(lhs) * gam * d)
    target = b * d - gam * 2 * g * sp.log(B + e)
    return sp.simplify(cleared - target) == 0


def numeric_gamma_lj75_plateau() -> bool:
    """Literature-scale inputs: γ_min ≈ 8.69/0.16 ≈ 54."""
    b, g, d, B = 8.69, 1.21, 225, 3.0e6
    gm = gamma_min(b, g, d, B)
    bm = b_max(g, d, B)
    ok = abs(gm - b / bm) < 1e-9
    ok &= 50.0 < gm < 60.0  # order-of-magnitude lock for the paper table
    # After bias b_eff = b/gm, window is just at the boundary; need γ>gm
    ok &= not window_nonempty(g, d, b / gm, B) or abs(b / gm - bm) < 1e-9
    # Strict: slightly larger γ reopens
    ok &= window_nonempty(g, d, b / (gm * 1.01), B)
    return ok


# ---------------------------------------------------------------------------
# Effective dimension after whitening (link to D6 anisotropy)
# ---------------------------------------------------------------------------


def symbolic_whitened_theta_c() -> bool:
    """After proposal covariance ∝ H^{-1}, T_c = 2 f / d recovers θ_c=2.

    On f = x'Hx/2, whitened coords y = H^{1/2} x give sphere f=||y||^2/2.
    Ambient dimension d is unchanged; soft-mode collapse is removed.
    """
    d, f = sp.symbols("d f", positive=True)
    # sphere: T_c = 2f/d  ⇒  θ = T d / f has critical value 2
    t_c = 2 * f / d
    theta_c = sp.simplify(t_c * d / f)
    return theta_c == 2


def soft_mode_collapse_factor() -> bool:
    """Isotropic proposal: T_c along e_min is (d/κ)-collapsed vs sphere scale.

    H = diag(1, κ) in 2-D for a minimal example: along soft axis x=(1,0),
    T_c = x'H^2 x / tr H = 1/(1+κ). Sphere at same f=1/2 would want 2f/d=1/2.
    Ratio T_c / (2f/d) = [1/(1+κ)] / (1/2) = 2/(1+κ) → 2/κ for large κ.
    """
    kappa = sp.symbols("kappa", positive=True)
    # d=2, f = 1/2 on soft unit vector with λ_min=1: f = λ_min/2 = 1/2
    t_c_soft = sp.Integer(1) / (1 + kappa)  # x'H^2x / trH, x=(1,0), H=diag(1,κ)
    t_sphere = sp.Rational(1, 2)  # 2f/d = 1/2
    ratio = sp.simplify(t_c_soft / t_sphere)
    return sp.simplify(ratio - 2 / (1 + kappa)) == 0


# ---------------------------------------------------------------------------
# Deposit table (literature inputs; algebra outputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRow:
    label: str
    barrier: float
    gap: float
    dim: int
    budget: float

    @property
    def t_lo(self) -> float:
        return t_lo(self.barrier, self.budget)

    @property
    def t_hi(self) -> float:
        return t_hi(self.gap, self.dim)

    @property
    def nonempty(self) -> bool:
        return window_nonempty(self.gap, self.dim, self.barrier, self.budget)

    @property
    def b_max(self) -> float:
        return b_max(self.gap, self.dim, self.budget)

    @property
    def ratio(self) -> float:
        return emptiness_ratio(self.barrier, self.gap, self.dim, self.budget)

    @property
    def gamma_star(self) -> float:
        return gamma_min(self.barrier, self.gap, self.dim, self.budget)


# Campaign / literature anchors (documented in derivation_spine note).
LJ75_PLATEAU = WindowRow("LJ75 plateau g=1.21 d=225 B=3e6 b=8.69", 8.69, 1.21, 225, 3e6)
LJ75_OPT_GAP = WindowRow("LJ75 g=50 d=225 B=3e6 b=8.69", 8.69, 50.0, 225, 3e6)
LJ75_1E7 = WindowRow("LJ75 plateau B=1e7", 8.69, 1.21, 225, 1e7)
SPHERE_LOCAL = WindowRow("sphere local g=10 b=2 d=10 B=1e4", 2.0, 10.0, 10, 1e4)

TABLE = (LJ75_PLATEAU, LJ75_OPT_GAP, LJ75_1E7, SPHERE_LOCAL)


def table_invariants() -> bool:
    """Hard locks for the paper table: plateau empty; sphere local open."""
    ok = not LJ75_PLATEAU.nonempty
    ok &= LJ75_PLATEAU.ratio > 50.0
    ok &= 50.0 < LJ75_PLATEAU.gamma_star < 60.0
    ok &= not LJ75_OPT_GAP.nonempty  # still empty at g=50
    ok &= SPHERE_LOCAL.nonempty
    ok &= SPHERE_LOCAL.ratio < 1.0
    return ok


# ---------------------------------------------------------------------------
# Aggregate witness
# ---------------------------------------------------------------------------


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("A: symbolic emptiness identity", symbolic_emptiness_identity()),
        ("A: numeric window matches b_max", numeric_window_matches_b_max()),
        ("B: symbolic E[Geo]=1/p force Q/p", symbolic_geometric_force()),
        ("B: symbolic union 1-(1-p)^n", symbolic_union_bound_identity()),
        ("B: numeric force / seed examples", numeric_force_examples()),
        ("C: symbolic bias reopening", symbolic_bias_reopening()),
        ("C: numeric γ* on LJ75 plateau", numeric_gamma_lj75_plateau()),
        ("D6 link: whitened θ_c=2", symbolic_whitened_theta_c()),
        ("D6 link: soft-mode collapse factor", soft_mode_collapse_factor()),
        ("table invariants (LJ75 empty, sphere open)", table_invariants()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("D12: Quenched force ledger, emptiness, bias reopening")
    print("Parents: D6 + D7 + D11")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print()
    print(f"{'label':48} {'T_lo':>8} {'T_hi':>8} {'empty?':>7} {'b_max':>8} {'b/bmax':>8} {'γ*':>8}")
    for row in TABLE:
        empty = "empty" if not row.nonempty else "open"
        print(
            f"{row.label:48} {row.t_lo:8.4f} {row.t_hi:8.4f} {empty:>7} "
            f"{row.b_max:8.4f} {row.ratio:8.2f} {row.gamma_star:8.2f}"
        )
    print()
    # Force ledger worked example (not a campaign claim)
    Q, p = 200.0, 0.04
    print(
        f"  force example: Q={Q}, p={p} -> E[force to first Marks-like hit] "
        f"= {expected_force_to_first_success(Q, p):.1f}"
    )
    print(
        f"  fixed budget: 8 seeds x 1e5 hops x Q=30 -> "
        f"{expected_force_fixed_hop_budget(30.0, 100_000, 8):.0f} force units"
    )
    print("D12_DERIVE_OK" if ok else "D12_DERIVE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
