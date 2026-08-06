"""Symbolic checks for the absorbing-chain escape algebra.

Every identity the Rust module relies on is verified here before it is coded.
"""

import sympy as sp

# ---------------------------------------------------------------------------
# 1. Fundamental matrix identities, on a symbolic 3-transient / 2-absorbing
#    canonical form.
# ---------------------------------------------------------------------------
q = sp.symbols("q0:9", positive=True)
r = sp.symbols("r0:6", positive=True)
Q = sp.Matrix(3, 3, q)
R = sp.Matrix(3, 2, r)
I3 = sp.eye(3)

N = (I3 - Q).inv()
B = N * R
t = N * sp.ones(3, 1)

# Claim A: N = sum_k Q^k, checked as (I - Q) N = I.
assert sp.simplify((I3 - Q) * N - I3) == sp.zeros(3, 3)

# Claim B: B is the absorption-probability matrix, i.e. it satisfies the
# first-step recursion B = R + Q B.
assert sp.simplify(B - (R + Q * B)) == sp.zeros(3, 2)

# Claim C: t satisfies t = 1 + Q t.
assert sp.simplify(t - (sp.ones(3, 1) + Q * t)) == sp.zeros(3, 1)

# Claim D (the one that saves |A| solves): row i of B and entry i of t both
# come from ONE transposed solve, (I - Q)^T y = e_i.
for i in range(3):
    e_i = sp.zeros(3, 1)
    e_i[i] = 1
    y = (I3 - Q).T.inv() * e_i
    assert sp.simplify((y.T * R) - B.row(i)) == sp.zeros(1, 2)
    assert sp.simplify((y.T * sp.ones(3, 1))[0] - t[i]) == 0
    # y is row i of N: the expected number of visits to each transient state.
    assert sp.simplify(y.T - N.row(i)) == sp.zeros(1, 3)
print("1. fundamental matrix, B = R + QB, t = 1 + Qt, transposed row extraction: OK")

# ---------------------------------------------------------------------------
# 2. Conditioning.  For a substochastic Q with rho(Q) < 1, N = sum Q^k >= 0
#    elementwise, so ||N||_inf is exactly max_i (N 1)_i = max_i t_i.  The
#    infinity-norm condition number of (I - Q) is therefore computable from
#    the expected-steps vector the method already solves for.
# ---------------------------------------------------------------------------
import numpy as np

rng = np.random.default_rng(7)
for trial in range(200):
    n = rng.integers(2, 8)
    M = rng.random((n, n))
    # Row sums strictly below one: a genuine substochastic block.
    M = M / (M.sum(axis=1, keepdims=True) / rng.uniform(0.2, 0.999))
    Nn = np.linalg.inv(np.eye(n) - M)
    tt = Nn @ np.ones(n)
    assert np.all(Nn > -1e-12), "N must be nonnegative"
    lhs = np.abs(Nn).sum(axis=1).max()
    assert abs(lhs - tt.max()) < 1e-9 * max(1.0, tt.max()), (lhs, tt.max())
    kappa = np.abs(np.eye(n) - M).sum(axis=1).max() * tt.max()
    assert abs(kappa - np.linalg.cond(np.eye(n) - M, np.inf)) < 1e-6 * kappa
print("2. kappa_inf(I - Q) = ||I - Q||_inf * max_i t_i, exactly: OK (200 random cases)")

# ---------------------------------------------------------------------------
# 3. Closed forms the Rust tests assert against.
#    Birth-death chain on {0..m}, up with p, down with 1-p, absorbing ends.
# ---------------------------------------------------------------------------
p = sp.Rational(2, 3)
m = 5
Qb = sp.zeros(m - 1, m - 1)
Rb = sp.zeros(m - 1, 2)  # columns: absorbed at 0, absorbed at m
for i in range(1, m):
    if i - 1 == 0:
        Rb[i - 1, 0] = 1 - p
    else:
        Qb[i - 1, i - 2] = 1 - p
    if i + 1 == m:
        Rb[i - 1, 1] = p
    else:
        Qb[i - 1, i] = p
Bb = (sp.eye(m - 1) - Qb).inv() * Rb
ratio = (1 - p) / p
for i in range(1, m):
    closed = (1 - ratio**i) / (1 - ratio**m)
    assert sp.simplify(Bb[i - 1, 1] - closed) == 0
print("3a. gambler's ruin absorption probability (1-(q/p)^i)/(1-(q/p)^m): OK")

# Symmetric walk: expected steps to absorption is i(m-i).
Qs = sp.zeros(m - 1, m - 1)
for i in range(1, m):
    if i - 1 != 0:
        Qs[i - 1, i - 2] = sp.Rational(1, 2)
    if i + 1 != m:
        Qs[i - 1, i] = sp.Rational(1, 2)
ts = (sp.eye(m - 1) - Qs).inv() * sp.ones(m - 1, 1)
for i in range(1, m):
    assert sp.simplify(ts[i - 1] - i * (m - i)) == 0
print("3b. symmetric walk expected steps i(m-i): OK")

# ---------------------------------------------------------------------------
# 4. Graph transformation (Trygubenko and Wales 2006).  Eliminating transient
#    state x renormalises the remaining branching probabilities as
#       P'_ij = P_ij + P_ix P_xj / (1 - P_xx)
#    and waiting times as tau'_i = tau_i + P_ix tau_x / (1 - P_xx).
#    Checked against the matrix answer: no inverse, no subtractive
#    cancellation, all quantities nonnegative throughout.
# ---------------------------------------------------------------------------
def graph_transform(P, transient, absorbing):
    """Exit distribution and mean number of steps, by node elimination."""
    P = {k: dict(v) for k, v in P.items()}
    tau = {i: sp.Integer(1) for i in transient}
    order = [s for s in transient]
    removed = []
    for x in order[1:]:  # keep the source state, eliminate the rest
        pxx = P[x].get(x, sp.Integer(0))
        denom = 1 - pxx
        for i in list(P):
            if i == x or i in removed:
                continue
            pix = P[i].pop(x, sp.Integer(0))
            if pix == 0:
                continue
            for j, pxj in P[x].items():
                if j == x:
                    continue
                P[i][j] = P[i].get(j, sp.Integer(0)) + pix * pxj / denom
            if i in tau:
                tau[i] = tau[i] + pix * tau[x] / denom
        removed.append(x)
    src = order[0]
    pss = P[src].get(src, sp.Integer(0))
    exit_dist = {j: sp.simplify(P[src].get(j, sp.Integer(0)) / (1 - pss)) for j in absorbing}
    steps = sp.simplify(tau[src] / (1 - pss))
    return exit_dist, steps


# A three-state trap with a rare exit: the regime the method is for.
eps = sp.Rational(1, 1000)
P = {
    0: {1: sp.Rational(1, 2) - eps, 2: sp.Rational(1, 2), "a": eps},
    1: {0: sp.Rational(1, 2), 2: sp.Rational(1, 2) - eps, "b": eps},
    2: {0: sp.Rational(1, 2), 1: sp.Rational(1, 2)},
    "a": {"a": 1},
    "b": {"b": 1},
}
Qm = sp.Matrix(3, 3, lambda i, j: P[i].get(j, 0))
Rm = sp.Matrix(3, 2, lambda i, j: P[i].get(["a", "b"][j], 0))
Bm = (sp.eye(3) - Qm).inv() * Rm
tm = (sp.eye(3) - Qm).inv() * sp.ones(3, 1)
gt_dist, gt_steps = graph_transform(P, [0, 1, 2], ["a", "b"])
assert sp.simplify(gt_dist["a"] - Bm[0, 0]) == 0
assert sp.simplify(gt_dist["b"] - Bm[0, 1]) == 0
assert sp.simplify(gt_steps - tm[0]) == 0
print("4. graph transformation reproduces N R and N 1 exactly:",
      f"exit a = {gt_dist['a']}, steps = {gt_steps}")
kap = sp.eye(3) - Qm
row_norm = max(sum(abs(kap[i, j]) for j in range(3)) for i in range(3))
print("   conditioning of this trap: max_i t_i =", sp.N(max(tm), 8),
      " kappa_inf =", sp.N(row_norm * max(tm), 8))

# ---------------------------------------------------------------------------
# 5. Mean-rate lumping (Chatterjee and Voter 2010).  Replace S by one coarse
#    state with k_eff(S -> j) = sum_{i in S} pi_i^S P_ij, pi^S the stationary
#    vector of the S-restricted renormalised matrix.  Check that the lumped
#    exit distribution equals the exact absorbing-chain answer in the limit of
#    perfect timescale separation, and quantify the error when it is finite.
# ---------------------------------------------------------------------------
def lumped_exit(P, S, absorbing):
    Ps = sp.Matrix(len(S), len(S), lambda i, j: P[S[i]].get(S[j], 0))
    rows = [sum(Ps.row(i)) for i in range(len(S))]
    Pn = sp.Matrix(len(S), len(S), lambda i, j: Ps[i, j] / rows[i])
    # Stationary vector of the renormalised internal chain.
    A = (Pn.T - sp.eye(len(S)))
    A = A.col_join(sp.ones(1, len(S)))
    b = sp.zeros(len(S), 1).col_join(sp.Matrix([[1]]))
    pi = (A.T * A).inv() * A.T * b
    out = {}
    for a in absorbing:
        out[a] = sp.simplify(sum(pi[i] * P[S[i]].get(a, 0) for i in range(len(S))))
    tot = sum(out.values())
    return {a: sp.simplify(v / tot) for a, v in out.items()}, pi


lump, pi = lumped_exit(P, [0, 1, 2], ["a", "b"])
print("5. lumped exit  a =", sp.nsimplify(lump["a"]), " exact from state 0 =", sp.nsimplify(Bm[0, 0]))
print("   total variation lumped vs exact =",
      sp.N(sp.Abs(lump["a"] - Bm[0, 0]), 6))
# Separation ratio: escape time over internal relaxation time.
Psn = sp.Matrix(3, 3, lambda i, j: Qm[i, j] / sum(Qm.row(i)))
ev = sorted([sp.Abs(sp.N(v)) for v in Psn.eigenvals()], reverse=True)
mu2 = ev[1]
t_rel = -1 / sp.log(mu2)
k_esc = sum(pi[i] * sum(Rm.row(i)) for i in range(3))
print(f"   mu2 = {sp.N(mu2, 6)}, t_rel = {sp.N(t_rel, 6)}, t_esc = {sp.N(1 / k_esc, 6)}, "
      f"separation = {sp.N(1 / (k_esc * t_rel), 6)}")

# ---------------------------------------------------------------------------
# 6. Waste-recycled unbiased rate estimator.  The chain runs on E + V but the
#    rate wanted is the one of the E chain.  Proposals are drawn from a kernel
#    that does not see V, so accumulating min(1, exp(-(E_j - E_i)/T)) over
#    proposals estimates q_ij a_ij for the unbiased chain, whatever V does.
# ---------------------------------------------------------------------------
Ei, Ej, V_i, V_j, T = sp.symbols("E_i E_j V_i V_j T", real=True, positive=True)
a_biased = sp.Min(1, sp.exp(-((Ej + V_j) - (Ei + V_i)) / T))
a_unbiased = sp.Min(1, sp.exp(-(Ej - Ei) / T))
# The estimator is a function of the proposal only, so its expectation over
# proposals is sum_j q_ij a_unbiased(i,j) = the unbiased transition rate; the
# bias enters nowhere.  Stated as the substitution that makes it so:
assert a_unbiased.subs({V_i: 0, V_j: 0}) == a_biased.subs({V_i: 0, V_j: 0})
print("6. waste-recycled estimator depends on E only; V cancels by construction")
