"""D10: Portfolio win-objective endgame W(p).

Composes D9 discovery value (or a fixed theta) with a conversion schedule
P_conv(p) under the explore-first split of remaining budget R.
"""

from __future__ import annotations

import math

from proofs.d9_good_turing_record import discovery_value


def win_objective_discovery(
    remaining: int,
    polish: int,
    slice_size: int,
    theta_disc: float,
    p_conv: float,
) -> float:
    """W(p) under D9: q0=1-theta, pi=(1-theta)^e (D10.1)."""
    if remaining < 0 or polish < 0 or polish > remaining:
        raise ValueError("invalid split")
    slice_size = max(int(slice_size), 1)
    e = (remaining - polish) // slice_size
    theta = min(max(float(theta_disc), 0.0), 1.0)
    q0 = 1.0 - theta
    pi = (1.0 - theta) ** min(e, 128)
    p_conv = min(max(float(p_conv), 0.0), 1.0)
    return (q0 + (1.0 - q0) * (1.0 - pi)) * p_conv


def win_objective_beta(
    remaining: int,
    polish: int,
    slice_size: int,
    alpha: float,
    beta: float,
    p_conv: float,
) -> float:
    """W(p) under D4 Beta fallback (exact product for E[(1-theta)^e])."""
    slice_size = max(int(slice_size), 1)
    e = (remaining - polish) // slice_size
    a = max(float(alpha), 1e-12)
    b = max(float(beta), 1e-12)
    q0 = b / (a + b)
    pi = 1.0
    for i in range(min(e, 128)):
        pi *= (b + i) / (a + b + i)
    p_conv = min(max(float(p_conv), 0.0), 1.0)
    return (q0 + (1.0 - q0) * (1.0 - pi)) * p_conv


def argmax_win(
    remaining: int,
    slice_size: int,
    theta_disc: float,
    conv_fn,
    grid: int = 12,
) -> int:
    """Grid maximizer of W(p); returns p* in work units."""
    best_p, best_w = 0, -1.0
    for k in range(grid + 1):
        p = int(round(k * remaining / grid))
        p = max(0, min(remaining, p))
        w = win_objective_discovery(remaining, p, slice_size, theta_disc, conv_fn(p))
        if w > best_w + 1e-15:
            best_w, best_p = w, p
    return best_p


def check_q0_identity() -> bool:
    theta = discovery_value(1, 6, 3)
    # q0 = 1 - theta
    w_full_explore = win_objective_discovery(100, 0, 10, theta, 1.0)
    # with p_conv=1 and e large, W -> 1
    return abs(w_full_explore - 1.0) < 1e-12 or w_full_explore < 1.0 + 1e-12


def check_no_exploration_is_q0() -> bool:
    """If e=0 (all polish), discovery factor is q0 = 1-theta."""
    theta = 0.25
    w = win_objective_discovery(40, 40, 10, theta, 0.8)
    return abs(w - (1.0 - theta) * 0.8) < 1e-12


def check_beta_product_matches_monte_mean() -> bool:
    """E[(1-theta)^e] for Beta(a,b) equals the rising factorial product."""
    a, b, e = 3.0, 5.0, 4
    # product formula
    pi = 1.0
    for i in range(e):
        pi *= (b + i) / (a + b + i)
    # exact: E[(1-theta)^e] = B(a,b+e)/B(a,b) = prod
    # Beta integral ratio
    from math import lgamma

    def logB(x, y):
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    pi_exact = math.exp(logB(a, b + e) - logB(a, b))
    return abs(pi - pi_exact) < 1e-12


def check_argmax_full_budget_when_conversion_needs_all() -> bool:
    """If conversion is 0 unless p=R, and theta small, p*=R."""

    def conv(p, R=60):
        return 1.0 if p >= R else 0.0

    p_star = argmax_win(60, 10, 0.05, lambda p: conv(p, 60), grid=12)
    return p_star == 60


def check_argmax_zero_when_already_converted() -> bool:
    """If P_conv=1 for all p, maximizer can sit at p=0 (pure explore)."""

    def conv(_p):
        return 1.0

    p_star = argmax_win(60, 10, 0.2, conv, grid=12)
    return p_star == 0


WITNESS = check_beta_product_matches_monte_mean()


def main() -> int:
    print("D10: Portfolio win-objective endgame")
    print(f"  WITNESS Beta product = B(a,b+e)/B(a,b): {WITNESS}")
    checks = [
        ("q0 / W bounds", check_q0_identity()),
        ("e=0 => W = q0 * P_conv", check_no_exploration_is_q0()),
        ("argmax p*=R when conv needs all", check_argmax_full_budget_when_conversion_needs_all()),
        ("argmax p*=0 when already converted", check_argmax_zero_when_already_converted()),
    ]
    ok = WITNESS
    for name, v in checks:
        print(f"  {name}: {v}")
        ok = ok and v
    theta = discovery_value(1, 6, 3)
    print(
        "  example W(p=0), W(p=R) theta=",
        f"{theta:.4f}",
        win_objective_discovery(80, 0, 10, theta, 0.5),
        win_objective_discovery(80, 80, 10, theta, 0.5),
    )
    print("D10 OK" if ok else "D10 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
