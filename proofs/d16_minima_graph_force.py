"""D16: Mean force-to-GM on a minima network (master equation) — new.

Wales-style kinetics uses rate matrices on local minima. Here each transition
i→j is a *hop* that costs force Q_ij > 0 (typically quench force), and we want
expected total force to absorption at the global minimum — the natural
objective under a charged ledger.

Setup.
  Finite directed graph on minima {0,...,n-1}, GM index g absorbing.
  Hop probability P_ij (row-stochastic on transient states; P_gg=1).
  Force cost C_ij ≥ 0 on edge i→j (C_ii=0 if self-loop from rejection).

Let F_i = expected remaining force to absorption from i.
Then F_g = 0 and for i ≠ g:
  F_i = sum_j P_ij (C_ij + F_j).

In matrix form on transient set T:
  F_T = c_T + P_TT F_T
  ⇒ (I - P_TT) F_T = c_T
where c_i = sum_j P_ij C_ij (expected one-step force from i).

Theorems.
  T1. If every transient state can reach g with positive probability,
      I-P_TT is nonsingular and F_T is unique and finite.
  T2. If C_ij = Q constant on all real hops (j≠i) and self-loops free,
      then c_i = Q (1-P_ii) and F_i = Q E_i[N_hops] where N_hops is the
      number of non-self transitions until absorption — more precisely
      F_i = Q * (expected number of charged steps).
  T3. Two-funnel coarsening: states I (trap class) and G (good class) + GM.
      Match D13 when escape I→G has prob ε per hop and G→GM has prob p.

Also: force-optimal edge reweighting is *not* the same as rate-optimal for
mean time when C_ij vary (prefer cheap edges even if slower in hops).

Run: PYTHONPATH=. python -m proofs.d16_minima_graph_force
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp


def expected_force(
    P: np.ndarray, C: np.ndarray, gm: int
) -> np.ndarray:
    """Solve F_i = sum_j P_ij (C_ij + F_j), F_gm=0."""
    n = P.shape[0]
    assert P.shape == (n, n) and C.shape == (n, n)
    trans = [i for i in range(n) if i != gm]
    m = len(trans)
    idx = {i: k for k, i in enumerate(trans)}
    A = np.eye(m)
    b = np.zeros(m)
    for i in trans:
        ii = idx[i]
        b[ii] = float(np.sum(P[i] * C[i]))  # c_i
        for j in trans:
            jj = idx[j]
            A[ii, jj] -= P[i, j]
        # terms P_ig (C_ig + 0) already in b via c_i; no F_g
    F_t = np.linalg.solve(A, b)
    F = np.zeros(n)
    for i in trans:
        F[i] = F_t[idx[i]]
    return F


def t1_nonsingular_line() -> bool:
    """Path 0→1→2=GM, always progress: unique F."""
    P = np.array(
        [
            [0.2, 0.8, 0.0],
            [0.1, 0.1, 0.8],
            [0.0, 0.0, 1.0],
        ]
    )
    C = np.ones((3, 3)) * 10.0
    np.fill_diagonal(C, 0.0)
    F = expected_force(P, C, gm=2)
    # From 1: mostly one hop cost 10 then done: F1 ≈ 10/0.8 * something
    return F[2] == 0.0 and F[0] > F[1] > 0 and np.isfinite(F).all()


def t2_constant_Q_means_hop_count() -> bool:
    """C_ij=Q for j≠i, C_ii=0 ⇒ F = Q * expected charged steps.

    Charged step = transition that is not a free self-loop; we define
    every attempt costs Q including rejected self-loops for this test:
    use C_ij=Q for all i,j including i=i (every step costs Q), then
    F_i = Q * E[steps to absorption].
    """
    Q = 5.0
    # pure progress: from 0 always to GM 1
    P = np.array([[0.0, 1.0], [0.0, 1.0]])
    C = Q * np.ones((2, 2))
    F = expected_force(P, C, gm=1)
    # one step from 0 costs Q
    return abs(F[0] - Q) < 1e-12 and abs(F[1]) < 1e-15


def symbolic_two_state_force() -> bool:
    """States T (transient) and G (GM). P_TG=p, P_TT=1-p, cost Q per step.

    F_T = (1-p)(Q + F_T) + p(Q + 0) ⇒ F_T = Q/p.
    Matches D12 geometric force.
    """
    p, Q = sp.symbols("p Q", positive=True)
    # F = (1-p)(Q+F) + p Q
    F = sp.symbols("F", positive=True)
    eq = sp.Eq(F, (1 - p) * (Q + F) + p * Q)
    sol = sp.solve(eq, F)[0]
    return sp.simplify(sol - Q / p) == 0


def two_funnel_coarsened_force(
    alpha: float,
    p: float,
    eps: float,
    Q: float,
    Q0: float,
) -> float:
    """Expected force from 'Start' under coarsened chain.

    States: S (start), I, G, M (GM absorbing).
    S --Q0--> I with 1-α, G with α (setup).
    I: hop cost Q, to G with ε, stay I with 1-ε (ε=0 ⇒ absorbed in I: infinite
       force unless we add death — for ε=0 use D13 finite hop budget instead).
    G: hop cost Q, to M with p, stay G with 1-p.

    For ε>0, F finite from I.
    """
    # indices S=0, I=1, G=2, M=3
    # We'll fold S as paying Q0 then branching into I or G by building
    # F from I and G only, then E from S = Q0 + α F_G + (1-α) F_I
    # P on {I,G,M}:
    # I→G: ε, I→I: 1-ε, G→M: p, G→G: 1-p, M→M: 1
    if eps <= 0.0:
        # infinite if start in I with positive prob and no escape
        if alpha >= 1.0 - 1e-15:
            # only G: F_G = Q/p
            return Q0 + Q / p
        return float("inf")
    P = np.array(
        [
            [1.0 - eps, eps, 0.0],  # I → I,G,M
            [0.0, 1.0 - p, p],  # G → I,G,M
            [0.0, 0.0, 1.0],  # M
        ]
    )
    C = Q * np.ones((3, 3))
    np.fill_diagonal(C, 0.0)
    # self-loops at I and G still "hops" that cost Q in SSBH (rejected or same)
    # charge every attempt: C_ii = Q as well
    C = Q * np.ones((3, 3))
    F = expected_force(P, C, gm=2)
    # F[0]=F_I, F[1]=F_G
    return Q0 + alpha * F[1] + (1.0 - alpha) * F[0]


def t3_matches_d12_when_alpha_1() -> bool:
    Q, p, Q0 = 10.0, 0.05, 3.0
    # α=1 ⇒ F = Q0 + Q/p
    F = two_funnel_coarsened_force(1.0, p, eps=0.0, Q=Q, Q0=Q0)
    return abs(F - (Q0 + Q / p)) < 1e-12


def t3_eps_finite() -> bool:
    F = two_funnel_coarsened_force(0.2, 0.1, eps=0.05, Q=5.0, Q0=2.0)
    return math.isfinite(F) and F > 0


def t3_eps_zero_alpha_partial_infinite() -> bool:
    F = two_funnel_coarsened_force(0.5, 0.1, eps=0.0, Q=5.0, Q0=2.0)
    return F == float("inf")


def force_vs_time_when_costs_differ() -> bool:
    """Cheaper edge preferred for mean force even if hop count higher.

    Graph: 0 → 1 (GM) direct: P=0.1, C=100
           0 → 2 → 1: P=0.9 to 2 cost 1, then 2→1 sure cost 1
    Mean hops may differ; mean force should use the cheap path more if we
    rewire — here fixed P, just compute F_0.
    """
    # states 0,2,1 with gm=1
    P = np.array(
        [
            [0.0, 0.1, 0.9],  # 0 → 0,1,2
            [0.0, 1.0, 0.0],  # 1 GM
            [0.0, 1.0, 0.0],  # 2 → 1
        ]
    )
    C = np.array(
        [
            [0.0, 100.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    F = expected_force(P, C, gm=1)
    # From 0: 0.1*(100+0) + 0.9*(1 + F_2), F_2=1
    # F0 = 10 + 0.9*(1+1) = 10+1.8=11.8
    return abs(F[0] - 11.8) < 1e-10


def symbolic_linear_system_two_transient() -> bool:
    """I,G transient, M absorb — recover F_G=Q/p, F_I = Q/ε + F_G for eps escape."""
    p, eps, Q = sp.symbols("p eps Q", positive=True)
    FG, FI = sp.symbols("F_G F_I", positive=True)
    # every attempt costs Q
    # G: F_G = (1-p)(Q+F_G) + p Q ⇒ F_G = Q/p
    eqG = sp.Eq(FG, (1 - p) * (Q + FG) + p * Q)
    solG = sp.simplify(sp.solve(eqG, FG)[0])
    ok = sp.simplify(solG - Q / p) == 0
    # I: F_I = (1-eps)(Q+F_I) + eps(Q+F_G)
    eqI = sp.Eq(FI, (1 - eps) * (Q + FI) + eps * (Q + solG))
    solI = sp.simplify(sp.solve(eqI, FI)[0])
    # F_I = Q/eps + Q/p
    ok &= sp.simplify(solI - (Q / eps + Q / p)) == 0
    return bool(ok)


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("T1 line graph finite F", t1_nonsingular_line()),
        ("T2 single hop F=Q", t2_constant_Q_means_hop_count()),
        ("symbolic two-state F=Q/p", symbolic_two_state_force()),
        ("T3 α=1 matches D12", t3_matches_d12_when_alpha_1()),
        ("T3 ε>0 finite force", t3_eps_finite()),
        ("T3 ε=0 partial α ⇒ inf", t3_eps_zero_alpha_partial_infinite()),
        ("heterogeneous costs F exact", force_vs_time_when_costs_differ()),
        ("symbolic F_I=Q/ε+Q/p", symbolic_linear_system_two_transient()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("D16: Mean force-to-GM on minima networks")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print()
    print("  F_I = Q/ε + Q/p  (escape then local) — force adds, hops add")
    print("  ε=0 and α<1 ⇒ infinite expected force without multi-start (D13)")
    print("D16_DERIVE_OK" if ok else "D16_DERIVE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
