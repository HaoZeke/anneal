"""D14: Screening force gain and costly bias reopening (new).

Complements D12 (emptiness / free bias factor γ) and D13 (two-funnel
multi-start). Here bias and screening *cost force*, so the design problem
is not "take γ→∞" or "screen everything".

---------------------------------------------------------------------
Part I — Screening
---------------------------------------------------------------------

Each trial:
  1. Propose a move (cheap): cost q0 ≥ 0.
  2. With probability ρ, a *screen* rejects without quench (cost 0 extra).
  3. Otherwise quench at cost Q ≫ q0.

Unscreened trial force: q0+Q.
Screened pipeline expected force per trial:
  q0 + (1-ρ) Q.

Define the force ratio
  η(ρ) = (q0 + (1-ρ)Q) / (q0+Q) ∈ (0,1].

Same number of *accepted-to-quench* trials T requires more proposals if
screening rejects: proposals = T / (1-ρ) if ρ is reject fraction among
all proposals that would have been quenched under no screen... 

Careful model (two variants):

Variant S1 (filter on doomed proposals). Among proposals, fraction ρ are
provably useless (e.g. unphysical, return-to-same minimum fingerprint)
and can be dropped with *zero* false-negative rate on the GM event.
Then to obtain M quenched hops you still need M quenches; you only save
the wasted quenches you would have spent on the ρ-fraction. If without
screen a fraction ρ of quenches were wasted,
  force_no_screen = M_total (q0+Q) with M_total quenches
  force_screen = M_useful (q0+Q) + M_waste q0
               = M_total[(1-ρ)(q0+Q) + ρ q0]
  ratio η = 1 - ρ Q/(q0+Q).

Variant S2 (aggressive screen with false-negative rate δ). Screen drops
fraction ρ of proposals but also kills a fraction δ of the proposals that
would have been GM successes. Effective success probability p' = p(1-δ)
per quenched hop, and quench rate (1-ρ). Force per wall-clock trial as above.

Prop S-A: η(ρ) = 1 - ρ Q/(q0+Q) is decreasing in ρ and in Q.
Prop S-B: under S2, effective force to first success
  E[force] = (q0+(1-ρ)Q) / (p_trial) with p_trial = (1-ρ) p (1-δ)
  if each trial is: propose, maybe screen, maybe quench, success only on
  quench with prob p(1-δ).
  Compared to no screen (ρ=0,δ=0): E0 = (q0+Q)/(p).
  Screen helps iff η_eff < 1 where
  E/E0 = [(q0+(1-ρ)Q)/(q0+Q)] * [1/((1-ρ)(1-δ))]
       = (1 - ρ Q/(q0+Q)) / ((1-ρ)(1-δ)).

Prop S-C: for δ=0, E/E0 = (1 - ρ Q/(q0+Q))/(1-ρ) = 1 + ρ q0/((1-ρ)(q0+Q)) > 1
  wait — if success only on quench and screen only drops doomed quenches,
  we should use S1 not S2. S2 assumes screen randomly thins *all* proposals
  including good ones.

S1 (safe screen): force ratio η_S1 = 1 - ρ Q/(q0+Q) < 1 always for ρ,Q>0.
S2 (lossy screen): help iff (1 - ρ Q/(q0+Q)) < (1-ρ)(1-δ).

---------------------------------------------------------------------
Part II — Costly bias reopening
---------------------------------------------------------------------

D12: free factor γ reopens when γ > γ_* = b/b_max.
Now each unit of log-bias (or each factor of γ) costs force: to maintain
well-tempered bias that achieves effective depth b/γ, spend C(γ) force
from the budget, leaving B' = B - C(γ) for the search.

Prop B-A: nonempty under residual budget requires
  b/γ < b_max(g, d, B - C(γ)) = 2 g ln(B-C(γ)+e) / d
  ⇔ γ > b d / (2 g ln(B-C(γ)+e)).

Prop B-B: linear cost model C(γ) = c0 (γ-1)_+  (pay proportional to
  excess over 1). Then feasible γ solves
  γ (ln(B - c0(γ-1) + e)) > b d / (2g).
  Define φ(γ) = γ ln(B - c0(γ-1) + e). Need φ(γ) > κ with κ=bd/(2g).

Prop B-C: φ is eventually decreasing once C eats the budget; there is a
  maximal γ_max = 1 + (B-B_min)/c0. If max_{γ∈[1,γ_max]} φ(γ) ≤ κ,
  *no* costly bias reopens the window — multi-start / MH / CSA required.

Prop B-D: free bias (c0=0) recovers D12: need γ > κ / ln(B+e) = γ_*.

Executable SymPy + numeric grids.
Run: PYTHONPATH=. python -m proofs.d14_screen_bias_force
"""
from __future__ import annotations

import math

import sympy as sp

from proofs.d11_budget_feasible_temp import EULER_E
from proofs.d12_quenched_ledger_and_window import gamma_min


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------


def eta_s1(rho: float, Q: float, q0: float) -> float:
    """Safe-screen force ratio: 1 - ρ Q/(q0+Q)."""
    if not (0.0 <= rho <= 1.0) or Q < 0.0 or q0 < 0.0:
        raise ValueError("invalid")
    denom = q0 + Q
    if denom <= 0.0:
        return 1.0
    return 1.0 - rho * Q / denom


def force_ratio_s2(rho: float, delta: float, Q: float, q0: float) -> float:
    """E/E0 for lossy screen S2: (1 - ρ Q/(q0+Q)) / ((1-ρ)(1-δ))."""
    if not (0.0 <= rho < 1.0 and 0.0 <= delta < 1.0):
        raise ValueError("invalid rho/delta")
    return eta_s1(rho, Q, q0) / ((1.0 - rho) * (1.0 - delta))


def screen_s2_helps(rho: float, delta: float, Q: float, q0: float) -> bool:
    return force_ratio_s2(rho, Q=Q, q0=q0, delta=delta) < 1.0 - 1e-15


def symbolic_eta_s1() -> bool:
    rho, Q, q0 = sp.symbols("rho Q q0", positive=True)
    eta = 1 - rho * Q / (q0 + Q)
    # d/dρ η = -Q/(q0+Q) < 0
    d_rho = sp.simplify(sp.diff(eta, rho))
    ok = d_rho == -Q / (q0 + Q)
    # d/dQ η = -ρ q0 / (q0+Q)^2 ≤ 0
    d_Q = sp.simplify(sp.diff(eta, Q))
    ok &= sp.simplify(d_Q - (-rho * q0 / (q0 + Q) ** 2)) == 0
    # ρ=0 ⇒ η=1; ρ=1 ⇒ η = q0/(q0+Q)
    ok &= sp.simplify(eta.subs(rho, 0) - 1) == 0
    ok &= sp.simplify(eta.subs(rho, 1) - q0 / (q0 + Q)) == 0
    return bool(ok)


def symbolic_s2_help_criterion() -> bool:
    """S2 helps iff 1 - ρ Q/(q0+Q) < (1-ρ)(1-δ)."""
    rho, delta, Q, q0 = sp.symbols("rho delta Q q0", positive=True)
    lhs = 1 - rho * Q / (q0 + Q)
    rhs = (1 - rho) * (1 - delta)
    # Rearrange: help when lhs - rhs < 0
    diff = sp.simplify(lhs - rhs)
    # At δ=0: lhs - rhs = 1 - ρ Q/(q0+Q) - (1-ρ) = ρ - ρ Q/(q0+Q) = ρ q0/(q0+Q) ≥ 0
    # so S2 with δ=0 never helps (random thinning) — matches Prop S-C narrative
    d0 = sp.simplify(diff.subs(delta, 0) - rho * q0 / (q0 + Q))
    ok = d0 == 0
    return bool(ok)


def numeric_s1_saves() -> bool:
    # ρ=0.5, Q=100, q0=1 → η = 1 - 0.5*100/101 ≈ 0.505
    eta = eta_s1(0.5, 100.0, 1.0)
    return abs(eta - (1 - 50 / 101)) < 1e-12 and eta < 1.0


def numeric_s2_delta0_never_helps() -> bool:
    for rho in (0.1, 0.3, 0.5, 0.8):
        if screen_s2_helps(rho, 0.0, Q=100.0, q0=1.0):
            return False
    return True


def numeric_s2_helps_when_delta_small_and_rho_high_Q() -> bool:
    # Actually with δ=0 never helps. With small δ but wait - random thin always
    # multiplies denominator. For help need δ negative impossible.
    # Lossy screen helps only if δ is negative? Let's re-read.
    # E/E0 = η/((1-ρ)(1-δ)). For this <1 need η < (1-ρ)(1-δ).
    # With δ=0, η = 1 - ρ Q/(q0+Q) vs 1-ρ: η - (1-ρ) = ρ q0/(q0+Q) ≥ 0 so η ≥ 1-ρ.
    # With δ>0, RHS smaller, harder to help. So pure random S2 NEVER helps!
    # That IS the theorem — important design result: screening only helps under S1
    # (safe identification of doomed proposals), not random thinning.
    return numeric_s2_delta0_never_helps()


# ---------------------------------------------------------------------------
# Costly bias
# ---------------------------------------------------------------------------


def residual_budget(B: float, C: float) -> float:
    return max(B - C, 0.0)


def gamma_star_costly(b: float, g: float, d: int, B: float, C: float) -> float:
    """γ needed if only B-C force remains for search."""
    Bp = residual_budget(B, C)
    if Bp <= 0:
        return float("inf")
    return gamma_min(b, g, d, Bp)


def linear_bias_cost(gamma: float, c0: float) -> float:
    return c0 * max(gamma - 1.0, 0.0)


def phi_linear(gamma: float, B: float, c0: float) -> float:
    """φ(γ)=γ ln(B - c0(γ-1) + e); domain γ such that residual ≥ 0."""
    C = linear_bias_cost(gamma, c0)
    Bp = B - C
    if Bp < 0:
        return float("-inf")
    return gamma * math.log(Bp + EULER_E)


def kappa(b: float, g: float, d: int) -> float:
    return b * d / (2.0 * g)


def feasible_costly_bias(
    b: float, g: float, d: int, B: float, c0: float, n_grid: int = 400
) -> tuple[bool, float, float]:
    """Return (exists feasible γ, γ_best, φ_max).

    Feasible when φ(γ) > κ.
    """
    kap = kappa(b, g, d)
    # γ_max when C=B: c0(γ-1)=B ⇒ γ=1+B/c0 if c0>0
    if c0 <= 0:
        # free bias: φ=γ ln(B+e); any γ>κ/ln(B+e)
        gstar = gamma_min(b, g, d, B)
        return True, gstar, float("inf")
    gmax = 1.0 + B / c0
    best_g, best_phi = 1.0, phi_linear(1.0, B, c0)
    for i in range(n_grid + 1):
        gam = 1.0 + (gmax - 1.0) * i / n_grid
        ph = phi_linear(gam, B, c0)
        if ph > best_phi:
            best_phi, best_g = ph, gam
    return best_phi > kap, best_g, best_phi


def symbolic_free_bias_recovers_d12() -> bool:
    """c0=0 ⇒ need γ > b d / (2g ln(B+e)) = γ_*."""
    b, g, d, B = sp.symbols("b g d B", positive=True)
    e = sp.E
    gstar = b * d / (2 * g * sp.log(B + e))
    # match d12 definition
    bmax = 2 * g * sp.log(B + e) / d
    ok = sp.simplify(gstar - b / bmax) == 0
    return bool(ok)


def numeric_costly_can_block() -> bool:
    """High c0: even best φ ≤ κ on LJ75-scale — bias alone cannot reopen."""
    b, g, d, B = 8.69, 1.21, 225, 3e6
    # enormous cost per γ unit: burning budget before reaching γ~54
    ok_block, _, phi_max = feasible_costly_bias(b, g, d, B, c0=1e5)
    # κ = b d/(2g) ≈ 8.69*225/(2.42) ≈ 807
    kap = kappa(b, g, d)
    # With c0 large, γ_max=1+B/c0 small, φ max small
    return (not ok_block) and phi_max < kap


def numeric_cheap_bias_can_open() -> bool:
    """Tiny c0: behaves like free bias, feasible."""
    b, g, d, B = 8.69, 1.21, 225, 3e6
    ok, gbest, phi_max = feasible_costly_bias(b, g, d, B, c0=1e-6)
    return ok and phi_max > kappa(b, g, d)


def numeric_phi_has_interior_max() -> bool:
    """φ(γ)=γ ln(B-c0(γ-1)+e) has an interior maximum for moderate c0."""
    B, c0 = 1000.0, 10.0
    gmax = 1.0 + B / c0
    # sample
    phis = [(g, phi_linear(g, B, c0)) for g in [1.0 + i * (gmax - 1) / 50 for i in range(51)]]
    g_at_max = max(phis, key=lambda t: t[1])[0]
    return 1.0 < g_at_max < gmax - 1e-9


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("S-A symbolic η(ρ,Q)", symbolic_eta_s1()),
        ("S-B symbolic S2 criterion / δ=0 identity", symbolic_s2_help_criterion()),
        ("S1 numeric saves force", numeric_s1_saves()),
        ("S2 random thin never helps (δ≥0)", numeric_s2_helps_when_delta_small_and_rho_high_Q()),
        ("B free bias recovers D12 γ_*", symbolic_free_bias_recovers_d12()),
        ("B costly bias can block LJ75-scale", numeric_costly_can_block()),
        ("B cheap bias can open", numeric_cheap_bias_can_open()),
        ("B φ interior max under linear cost", numeric_phi_has_interior_max()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("D14: Screening force gain and costly bias reopening")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print()
    print("  S1 η(ρ=0.5,Q=100,q0=1) =", round(eta_s1(0.5, 100, 1), 6))
    print("  S2 E/E0 (ρ=0.5,δ=0) =", round(force_ratio_s2(0.5, 0.0, 100, 1), 6), "(>1 never helps)")
    b, g, d, B = 8.69, 1.21, 225, 3e6
    print(f"  κ=bd/(2g) = {kappa(b,g,d):.4f}, free γ_* = {gamma_min(b,g,d,B):.4f}")
    for c0 in (1e-6, 1.0, 100.0, 1e5):
        feas, gb, ph = feasible_costly_bias(b, g, d, B, c0)
        print(f"  c0={c0:.0e}: feasible={feas} γ_best~{gb:.2f} φ_max={ph:.4f}")
    print("D14_DERIVE_OK" if ok else "D14_DERIVE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
