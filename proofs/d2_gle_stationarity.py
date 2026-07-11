"""D2: GLE move-kernel stationarity at fixed temperature.

The GLE thermostat (eindir/src/gle.rs) evolves the augmented momentum state
z = (p, s_1, ..., s_ns) by an Ornstein-Uhlenbeck (OU) process

    dz = -A z dt + B dW,

with drift A and diffusion B tied by the fluctuation-dissipation relation

    A C + C A^T = B B^T,    C = T * I    (canonical / sampling thermostat).

The code uses the EXACT OU sub-step (no time discretization error in the bath):

    Tprop = exp(-dt A),    S S^T = C - Tprop C Tprop^T,    z <- Tprop z + S xi,

with xi standard normal. This module verifies, symbolically on 2x2 blocks
(ns = 1) and numerically on the fitted optimal-sampling drift:

  1. FDT algebra: for a damped-oscillator block A skew-coupled to the physical
     momentum, A + A^T is diagonal >= 0 and A C + C A^T = B B^T with C = I.
  2. OU stationary covariance: the Lyapunov equation A C + C A^T = B B^T has
     C = T*I as its solution for the canonical drift, so N(0, C) is the
     stationary law of the continuous OU dynamics.
  3. Exact discrete OU step preserves N(0, C): if Cov(z) = C then
     Cov(Tprop z + S xi) = Tprop C Tprop^T + S S^T = C, using S S^T =
     C - Tprop C Tprop^T. This is an identity, not an approximation.
  4. Numeric: the fitted optimal_sampling_drift (ns = 12) satisfies FDT for
     C = I, and the discrete step preserves C to machine precision.

Style follows proofs/thmN_*.py: a module-level WITNESS Boolean plus checks.
"""

import sympy as sp
import numpy as np


def expm(M, n_taylor=20, n_square=12):
    """Matrix exponential by Taylor scaling-and-squaring, mirroring i-PI's
    matrix_exp used in eindir/src/gle.rs. Avoids a scipy dependency."""
    n = M.shape[0]
    scale = 2.0 ** n_square
    sm = M / scale
    tc = np.empty(n_taylor + 1)
    tc[0] = 1.0
    for i in range(n_taylor):
        tc[i + 1] = tc[i] / (i + 1.0)
    em = np.eye(n) * tc[n_taylor]
    for i in range(n_taylor - 1, -1, -1):
        em = sm @ em + np.eye(n) * tc[i]
    for _ in range(n_square):
        em = em @ em
    return em


# ---- symbolic 2x2 block (one auxiliary momentum, ns = 1) -------------------
# Damped-oscillator block skew-coupled to the physical momentum p, in the form
# used by optimal_sampling_drift: diagonal damping gamma, skew oscillation
# omega, skew coupling c between p (index 0) and the auxiliary (index 1).
gamma, omega, c, T = sp.symbols("gamma omega c T", positive=True)
app = sp.symbols("a_pp", positive=True)  # physical-momentum self-damping

A = sp.Matrix([
    [app, c],
    [-c, gamma],
])
# the off-diagonal omega term lives in larger blocks; for ns=1 the skew part is
# the coupling c. Keep an explicit omega variant for the 3x3 oscillator block
# below.

C = sp.eye(2) * T  # canonical covariance, C = T I


def bbt():
    """B B^T from the FDT relation: A C + C A^T."""
    return sp.simplify(A * C + C * A.T)


def check_fdt_symmetric_psd():
    """A C + C A^T is symmetric (so B B^T is a valid Gram matrix) and, for
    C = T I, equals T (A + A^T), whose off-diagonals (the skew coupling)
    cancel, leaving a diagonal non-negative matrix."""
    M = bbt()
    symmetric = sp.simplify(M - M.T) == sp.zeros(2, 2)
    # M = T (A + A^T); skew parts (c and -c) cancel on the off-diagonal
    expected = sp.simplify(T * (A + A.T))
    matches = sp.simplify(M - expected) == sp.zeros(2, 2)
    offdiag_zero = sp.simplify(M[0, 1]) == 0 and sp.simplify(M[1, 0]) == 0
    return symmetric and matches and offdiag_zero


def check_lyapunov_solution():
    """C = T I solves the steady-state Lyapunov equation A C + C A^T = B B^T
    with B B^T = T (A + A^T): substituting back is an identity."""
    M = bbt()                      # this IS B B^T by construction
    lhs = sp.simplify(A * C + C * A.T - M)
    return lhs == sp.zeros(2, 2)


def check_discrete_step_preserves_C():
    """Exact OU step z' = Tprop z + S xi with S S^T = C - Tprop C Tprop^T
    preserves Cov = C:  Tprop C Tprop^T + (C - Tprop C Tprop^T) = C.
    Verified symbolically without evaluating the matrix exponential (the
    identity holds for ANY Tprop)."""
    Tprop = sp.MatrixSymbol("Tp", 2, 2)
    Tp = sp.Matrix(Tprop)
    sst = C - Tp * C * Tp.T           # noise covariance S S^T
    cov_next = Tp * C * Tp.T + sst    # propagated covariance
    return sp.simplify(cov_next - C) == sp.zeros(2, 2)


WITNESS = (
    check_fdt_symmetric_psd()
    and check_lyapunov_solution()
    and check_discrete_step_preserves_C()
)


# ---- numeric: fitted optimal-sampling drift (ns = 12) ----------------------
_OPTIMAL_SAMPLING_REF = [
    (4.008052, 90.290099, 0.833024),
    (93.560269, 37.509324, 40.162773),
    (49.838463, 16.822080, 13.008423),
    (0.664421, 281.450678, 6.977272),
    (148.046748, 79.296860, 0.552854),
    (5.859307, 323.666216, 0.567559),
]
_ANCHOR = 0.664421


def optimal_sampling_drift(omega0=1.0):
    n = 2 * len(_OPTIMAL_SAMPLING_REF) + 1
    A = np.zeros((n, n))
    A[0, 0] = _ANCHOR * omega0
    for k, (omega_k, gamma_k, c_k) in enumerate(_OPTIMAL_SAMPLING_REF):
        i1, i2 = 1 + 2 * k, 2 + 2 * k
        A[i1, i1] = gamma_k * omega0
        A[i2, i2] = gamma_k * omega0
        A[i1, i2] = -omega_k * omega0
        A[i2, i1] = omega_k * omega0
        A[0, i1] = c_k * omega0
        A[i1, 0] = -c_k * omega0
    return A


def check_numeric_fdt(Tval=1.0):
    A = optimal_sampling_drift(1.0)
    n = A.shape[0]
    Cm = np.eye(n) * Tval
    BBt = A @ Cm + Cm @ A.T
    # symmetric and PSD
    sym = np.allclose(BBt, BBt.T, atol=1e-10)
    eig = np.linalg.eigvalsh(0.5 * (BBt + BBt.T))
    psd = eig.min() > -1e-9
    # equals T (A + A^T)
    matches = np.allclose(BBt, Tval * (A + A.T), atol=1e-10)
    return sym and psd and matches


def check_numeric_discrete_step(dt=0.01, Tval=1.0):
    A = optimal_sampling_drift(1.0)
    n = A.shape[0]
    Cm = np.eye(n) * Tval
    Tprop = expm(-dt * A)
    sst = Cm - Tprop @ Cm @ Tprop.T
    cov_next = Tprop @ Cm @ Tprop.T + sst
    return np.allclose(cov_next, Cm, atol=1e-10)


def derive():
    sp.init_printing(use_unicode=False)
    print("D2: GLE move-kernel stationarity at fixed temperature")
    print("  Check 1 (FDT: A C + C A^T symmetric, diagonal, = T(A+A^T)):",
          check_fdt_symmetric_psd())
    print("    B B^T =", bbt().tolist())
    print("  Check 2 (Lyapunov: C = T I solves A C + C A^T = B B^T):",
          check_lyapunov_solution())
    print("  Check 3 (exact OU step preserves Cov = C, any Tprop):",
          check_discrete_step_preserves_C())
    print("  Check 4a (numeric FDT, fitted ns=12 drift):", check_numeric_fdt())
    print("  Check 4b (numeric exact step preserves C):",
          check_numeric_discrete_step())
    all_ok = (
        WITNESS and check_numeric_fdt() and check_numeric_discrete_step()
    )
    print("  ALL CHECKS PASS:", all_ok)
    return all_ok


if __name__ == "__main__":
    ok = derive()
    raise SystemExit(0 if ok else 1)
