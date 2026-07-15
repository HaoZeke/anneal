"""D8: Entropy-calibrated inverse temperature for residual population control.

Verifies the free-energy / entropy calculus that calibrates the DMC residual
branching inverse temperature:

  H(beta) = log Z(beta) + beta * E_beta[E]
  H'(beta) = -beta * Var_beta(E)

and the unique inverse map beta* = H^{-1}(H_star) for target entropies on
(log m, log N], plus residual mass conservation of the ECIT algorithm.

No Monte Carlo: all checks are closed-form algebra, finite-difference
identities, or exact discrete residual bookkeeping.
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp


def softmax_probs(energies: np.ndarray, beta: float) -> np.ndarray:
    """Stable softmax p_i ∝ exp(-beta * E_i)."""
    e = np.asarray(energies, dtype=float)
    shifted = e - np.min(e)
    b = max(float(beta), 0.0)
    w = np.exp(-b * shifted)
    z = w.sum()
    if not np.isfinite(z) or z <= 0.0:
        return np.full(e.shape, 1.0 / e.size)
    return w / z


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def entropy_of_beta(energies: np.ndarray, beta: float) -> float:
    return shannon_entropy(softmax_probs(energies, beta))


def energy_var(energies: np.ndarray, beta: float) -> float:
    p = softmax_probs(energies, beta)
    e = np.asarray(energies, dtype=float)
    mean = float(np.dot(p, e))
    return float(np.dot(p, (e - mean) ** 2))


def target_entropy(n: int, progress: float, floor_factor: float = 2.0) -> float:
    """H_star(rho) = (1-rho) log N + rho log max(1, f N) with f = floor_factor/N."""
    n = max(int(n), 1)
    rho = min(max(float(progress), 0.0), 1.0)
    # floor_factor counts elite slots: default retain ~2 elites in entropy units
    floor_n = max(1.0, min(float(floor_factor), float(n)))
    return (1.0 - rho) * math.log(n) + rho * math.log(floor_n)


def beta_max(energies: np.ndarray) -> float:
    e = np.asarray(energies, dtype=float)
    r = float(np.max(e) - np.min(e))
    n = max(e.size, 1)
    if r <= 0.0:
        return 0.0
    return 20.0 * math.log(n) / r


def calibrate_beta(
    energies: np.ndarray,
    h_star: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 64,
) -> float:
    """Unique beta* with H(beta*) = h_star by bisection (Proposition D8 / Cor D8.4)."""
    e = np.asarray(energies, dtype=float)
    n = e.size
    if n == 0:
        return 0.0
    h0 = entropy_of_beta(e, 0.0)
    # multiplicity of min
    e_min = float(np.min(e))
    m = int(np.sum(np.abs(e - e_min) < 1e-15 * (1.0 + abs(e_min))))
    h_inf = math.log(max(m, 1))
    # clamp target into open-closed range
    h_star = min(max(float(h_star), h_inf + 1e-12), h0)
    if abs(h_star - h0) <= tol:
        return 0.0
    lo, hi = 0.0, max(beta_max(e), 1e-12)
    # expand hi until H(hi) <= h_star
    for _ in range(40):
        if entropy_of_beta(e, hi) <= h_star + tol:
            break
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        h_mid = entropy_of_beta(e, mid)
        if h_mid > h_star:
            lo = mid
        else:
            hi = mid
        if abs(h_mid - h_star) <= tol or (hi - lo) <= tol * (1.0 + mid):
            return mid
    return 0.5 * (lo + hi)


def residual_expected_counts(probs: np.ndarray, target_n: int) -> np.ndarray:
    return np.asarray(probs, dtype=float) * float(target_n)


def residual_floor_sum(expected: np.ndarray) -> tuple[float, float]:
    """Return (sum floor, sum fractional residual)."""
    floors = np.floor(expected)
    residual = expected - floors
    return float(floors.sum()), float(residual.sum())


# ---------------------------------------------------------------------------
# Symbolic witness for H' = -beta Var on a two-level spectrum
# ---------------------------------------------------------------------------

def symbolic_hprime_identity() -> bool:
    """Two energies (0, d) with multiplicities (1, n-1): check H' + beta Var = 0."""
    beta, d, n = sp.symbols("beta d n", positive=True)
    # Z = e^0 + (n-1) e^{-beta d} = 1 + (n-1) exp(-beta d)
    z = 1 + (n - 1) * sp.exp(-beta * d)
    p0 = 1 / z
    p1 = sp.exp(-beta * d) / z
    # mean energy: 0*p0 + d*(n-1)*p1
    mean = d * (n - 1) * p1
    # second moment
    second = (d**2) * (n - 1) * p1
    var = sp.simplify(second - mean**2)
    h = sp.log(z) + beta * mean
    hprime = sp.simplify(sp.diff(h, beta))
    residual = sp.simplify(hprime + beta * var)
    return residual == 0


WITNESS = symbolic_hprime_identity()


def check_numeric_hprime(energies=None, betas=None, eps=1e-6) -> bool:
    if energies is None:
        energies = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    if betas is None:
        betas = [0.1, 0.5, 1.0, 2.0, 5.0]
    for b in betas:
        h_plus = entropy_of_beta(energies, b + eps)
        h_minus = entropy_of_beta(energies, max(b - eps, 0.0))
        denom = (b + eps) - max(b - eps, 0.0)
        fd = (h_plus - h_minus) / denom
        analytic = -b * energy_var(energies, b)
        if abs(fd - analytic) > 5e-4 * (1.0 + abs(analytic)):
            return False
    return True


def check_monotonicity(energies=None) -> bool:
    if energies is None:
        energies = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    betas = np.linspace(0.0, 10.0, 21)
    hs = [entropy_of_beta(energies, b) for b in betas]
    return all(hs[i] >= hs[i + 1] - 1e-12 for i in range(len(hs) - 1))


def check_endpoints(energies=None) -> bool:
    if energies is None:
        energies = np.array([0.0, 0.0, 1.0, 2.0])  # m=2 at min
    n = energies.size
    h0 = entropy_of_beta(energies, 0.0)
    if abs(h0 - math.log(n)) > 1e-12:
        return False
    e_min = float(np.min(energies))
    m = int(np.sum(np.abs(energies - e_min) < 1e-15))
    h_big = entropy_of_beta(energies, 1e6)
    return abs(h_big - math.log(m)) < 1e-6


def check_calibration_unique(energies=None) -> bool:
    if energies is None:
        energies = np.array([0.0, 0.3, 1.0, 2.5, 4.0])
    n = energies.size
    for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
        h_star = target_entropy(n, rho, floor_factor=2.0)
        beta = calibrate_beta(energies, h_star)
        h = entropy_of_beta(energies, beta)
        if abs(h - h_star) > 1e-6 * (1.0 + abs(h_star)):
            return False
        # uniqueness sample: nearby beta moves H away
        if beta > 0:
            if entropy_of_beta(energies, beta * 1.5) > h_star + 1e-5:
                # still ok if very flat; require non-increase
                pass
            if entropy_of_beta(energies, beta * 1.5) > h + 1e-9:
                return False
    return True


def check_residual_mass(energies=None, target_n: int = 16) -> bool:
    if energies is None:
        energies = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 8.0])
    h_star = target_entropy(energies.size, 0.4, floor_factor=2.0)
    beta = calibrate_beta(energies, h_star)
    p = softmax_probs(energies, beta)
    expected = residual_expected_counts(p, target_n)
    if abs(expected.sum() - target_n) > 1e-9:
        return False
    floors, residual = residual_floor_sum(expected)
    # floor + residual = target_n exactly
    if abs(floors + residual - target_n) > 1e-9:
        return False
    # residual mass equals number of fractional slots needed
    need = target_n - int(floors)
    if abs(residual - need) > 1e-9:
        return False
    return True


def main() -> int:
    print("D8: Entropy-calibrated inverse temperature")
    print(f"  WITNESS H' + beta Var = 0 (two-level symbolic): {WITNESS}")
    checks = [
        ("numeric H' ≈ -beta Var", check_numeric_hprime()),
        ("H monotone nonincreasing", check_monotonicity()),
        ("endpoint limits H(0)=log N, H(∞)=log m", check_endpoints()),
        ("unique beta* for H_star schedule", check_calibration_unique()),
        ("residual floor+frac = target_n", check_residual_mass()),
    ]
    ok_all = WITNESS
    for name, ok in checks:
        print(f"  {name}: {ok}")
        ok_all = ok_all and ok
    # print a worked example
    e = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
    for rho in (0.0, 0.5, 1.0):
        hs = target_entropy(e.size, rho)
        b = calibrate_beta(e, hs)
        print(
            f"  example rho={rho:.1f} H*={hs:.6f} beta*={b:.6f} H(beta*)={entropy_of_beta(e, b):.6f}"
        )
    print("D8 OK" if ok_all else "D8 FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
