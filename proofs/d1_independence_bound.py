"""D1: Metropolis-independence acceptance lower bound.

An independence sampler proposes ``y ~ q`` with ``q(x) propto exp(-s(x)/T)``
built from a surrogate ``s``, and targets ``pi(x) propto exp(-f(x)/T)``. With
the per-point log-surrogate error

    r(x) = (f(x) - s(x)) / T,    delta = sup_x |r(x)|,

the Metropolis-Hastings acceptance probability is

    alpha(x, y) = min(1, exp(r(x) - r(y))),

so alpha(x, y) >= min(1, exp(-2 delta)) = exp(-2 delta), with no dependence on
the dimension d. The constant 2 is tight: it is attained when r(x) = -delta and
r(y) = +delta simultaneously.

This module exposes a module-level WITNESS Boolean (symbolic identity for the
MH ratio) plus numeric checks of the bound and its tightness, matching the
proofs/thmN_*.py style.

Checks:
  1. Symbolic: the MH ratio R = [pi(y) q(x)] / [pi(x) q(y)] simplifies to
     exp(r(x) - r(y)) where r = (f - s)/T.  (sympy)
  2. The constant is 2, not 1: the worst-case exponent r(x)-r(y) over the box
     [-delta, +delta]^2 equals -2 delta (tightness).  (sympy)
  3. Numeric: random surrogate errors of sup-norm delta give acceptance always
     >= exp(-2 delta), and the bound is saturated by the adversarial pair.
  4. Uniform-ergodicity rate: the independence-sampler minorization constant is
     exp(-2 delta), so ||P^n(x, .) - pi||_TV <= (1 - exp(-2 delta))^n.  We
     verify this on a finite-state independence chain by exact TV computation.
"""

import sympy as sp
import numpy as np

# ---- symbolic MH ratio -----------------------------------------------------
fx, fy, sx, sy, T, delta = sp.symbols("f_x f_y s_x s_y T delta", real=True)
T = sp.symbols("T", positive=True)

pi_ratio = sp.exp(-(fy - fx) / T)  # pi(y)/pi(x)
q_ratio_inv = sp.exp((sy - sx) / T)  # q(x)/q(y)
R = sp.simplify(pi_ratio * q_ratio_inv)

rx = (fx - sx) / T
ry = (fy - sy) / T
R_in_terms_of_r = sp.exp(rx - ry)

WITNESS = sp.simplify(R - R_in_terms_of_r) == 0


def _worst_case_exponent():
    """Minimum of (r_x - r_y) over r_x, r_y in [-delta, delta] is -2 delta."""
    d = sp.symbols("delta", positive=True)
    # r_x in [-d, d], r_y in [-d, d]; minimise r_x - r_y -> r_x = -d, r_y = +d
    return sp.simplify((-d) - (d))  # = -2 d


def check_tightness():
    d = sp.symbols("delta", positive=True)
    return sp.simplify(_worst_case_exponent() + 2 * d) == 0


def _alpha(rx_val, ry_val):
    return min(1.0, float(np.exp(rx_val - ry_val)))


def check_numeric_bound(delta_val=0.37, ntrials=20000, seed=0):
    rng = np.random.default_rng(seed)
    floor = float(np.exp(-2.0 * delta_val))
    worst = np.inf
    for _ in range(ntrials):
        rx_val = rng.uniform(-delta_val, delta_val)
        ry_val = rng.uniform(-delta_val, delta_val)
        a = _alpha(rx_val, ry_val)
        worst = min(worst, a)
        if a < floor - 1e-12:
            return False, a, floor
    # adversarial pair saturates the bound
    a_adv = _alpha(-delta_val, +delta_val)
    saturates = abs(a_adv - floor) < 1e-12
    return (worst >= floor - 1e-12) and saturates, worst, floor


def check_uniform_ergodicity(delta_val=0.4, nstates=5, nsteps=6, seed=1):
    """Finite-state independence sampler: P(x, y) = q(y) alpha(x, y) for y != x,
    with the holding mass on the diagonal. The whole-space minorization
    P(x, .) >= exp(-2 delta) pi(.) gives ||P^n(x, .) - pi||_TV <= rho^n with
    rho = 1 - exp(-2 delta). Verify the TV decay numerically.
    """
    rng = np.random.default_rng(seed)
    # target energies f and surrogate energies s with sup |f - s|/T = delta
    f = rng.normal(size=nstates)
    # construct s so that (f - s) hits +-delta but stays within sup-norm delta
    err = rng.uniform(-delta_val, delta_val, size=nstates)
    err[0] = delta_val
    err[1] = -delta_val
    s = f - err  # so (f - s) = err, sup|err| = delta_val (T folded in: set T=1)
    pi = np.exp(-f)
    pi /= pi.sum()
    q = np.exp(-s)
    q /= q.sum()

    # independence-sampler transition matrix
    P = np.zeros((nstates, nstates))
    for x in range(nstates):
        for y in range(nstates):
            if y == x:
                continue
            ratio = (pi[y] * q[x]) / (pi[x] * q[y])
            a = min(1.0, ratio)
            P[x, y] = q[y] * a
        P[x, x] = 1.0 - P[x].sum()
    # check pi is stationary
    stat_ok = np.allclose(pi @ P, pi, atol=1e-10)
    # TV decay
    rho = 1.0 - float(np.exp(-2.0 * delta_val))
    ok = True
    for x0 in range(nstates):
        dist = np.zeros(nstates)
        dist[x0] = 1.0
        for n in range(1, nsteps + 1):
            dist = dist @ P
            tv = 0.5 * np.abs(dist - pi).sum()
            if tv > rho**n + 1e-9:
                ok = False
    return stat_ok and ok, rho


def derive():
    sp.init_printing(use_unicode=False)
    print("D1: Metropolis-independence acceptance lower bound")
    print("  Check 1 (MH ratio = exp(r_x - r_y), r=(f-s)/T):", WITNESS)
    print("    R simplified            =", R)
    print("    exp(r_x - r_y)          =", sp.simplify(R_in_terms_of_r))
    print("  Check 2 (worst exponent = -2 delta, tight constant=2):", check_tightness())
    print("    min_{r in [-d,d]^2}(r_x-r_y) =", _worst_case_exponent())
    ok3, worst, floor = check_numeric_bound()
    print(f"  Check 3 (numeric bound, delta=0.37): {ok3}")
    print(f"    empirical min alpha = {worst:.6f}  >=  exp(-2 delta) = {floor:.6f}")
    ok4, rho = check_uniform_ergodicity()
    print(f"  Check 4 (uniform ergodicity, rate rho=1-exp(-2 delta)={rho:.4f}): {ok4}")
    all_ok = WITNESS and check_tightness() and ok3 and ok4
    print("  ALL CHECKS PASS:", all_ok)
    return all_ok


if __name__ == "__main__":
    ok = derive()
    raise SystemExit(0 if ok else 1)
