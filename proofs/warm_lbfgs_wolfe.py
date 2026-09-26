"""Verification for warm_lbfgs_wolfe.org — strong Wolfe ⇒ curvature.

Mirrors the mathematical claims used by src/methods/warm_lbfgs.rs.
"""
from __future__ import annotations

import math
import random


def curvature_from_wolfe(
    g_dot_d: float,
    g_new_dot_d: float,
    alpha: float,
    c2: float,
) -> float:
    """y^T s = alpha * (g_new^T d - g^T d) under s = alpha d."""
    return alpha * (g_new_dot_d - g_dot_d)


def theorem1_holds(
    g_dot_d: float,
    g_new_dot_d: float,
    alpha: float,
    c2: float,
) -> bool:
    """Check (C): y^Ts >= (1-c2)*(-g^Td)*alpha > 0 under W2 lower half."""
    if not (g_dot_d < 0 and alpha > 0 and 0 < c2 < 1):
        return False
    # W2 lower: g_new^T d >= c2 * g^T d  (since g^Td < 0)
    if g_new_dot_d + 1e-15 < c2 * g_dot_d:
        return False
    yTs = curvature_from_wolfe(g_dot_d, g_new_dot_d, alpha, c2)
    bound = (1.0 - c2) * (-g_dot_d) * alpha
    return yTs + 1e-12 >= bound and yTs > 0.0


def test_theorem1_random() -> bool:
    rng = random.Random(0)
    for _ in range(200):
        g_dot_d = -rng.uniform(1e-3, 10.0)
        c2 = rng.uniform(0.1, 0.95)
        alpha = rng.uniform(1e-4, 2.0)
        # sample g_new^T d in [c2 g^Td, -c2 g^Td] (strong Wolfe band)
        lo = c2 * g_dot_d
        hi = -c2 * g_dot_d
        g_new = rng.uniform(lo, hi)
        if not theorem1_holds(g_dot_d, g_new, alpha, c2):
            return False
    return True


def test_theorem1_boundary() -> bool:
    # equality case g_new^T d = c2 g^T d
    g_dot_d = -4.0
    c2 = 0.9
    alpha = 0.5
    g_new = c2 * g_dot_d
    yTs = curvature_from_wolfe(g_dot_d, g_new, alpha, c2)
    bound = (1.0 - c2) * (-g_dot_d) * alpha
    return abs(yTs - bound) < 1e-12 and yTs > 0


def test_armijo_descent() -> bool:
    f0 = 10.0
    slope = -2.0  # g^T d
    c1 = 1e-4
    alpha = 0.25
    # worst Armijo point: equality in W1
    f_new = f0 + c1 * alpha * slope
    return f_new < f0


def test_eval_bound() -> bool:
    L, K = 20, 15
    return 1 + K * L == 301


def all_checks() -> list[tuple[str, bool]]:
    return [
        ("Thm1 random strong-Wolfe samples", test_theorem1_random()),
        ("Thm1 boundary equality", test_theorem1_boundary()),
        ("Thm2 Armijo descent", test_armijo_descent()),
        ("Thm3 eval bound identity", test_eval_bound()),
    ]


WITNESS = all(v for _, v in all_checks())


def main() -> int:
    print("WarmLbfgs / strong Wolfe curvature — verification")
    print()
    ok = True
    for name, v in all_checks():
        print(f"  {name}: {v}")
        ok = ok and bool(v)
    print("WARM_LBFGS_WOLFE_OK" if ok else "WARM_LBFGS_WOLFE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
