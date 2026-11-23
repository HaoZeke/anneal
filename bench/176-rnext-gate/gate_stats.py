"""Type-7 medians, Fisher exact tests, and Mann-Whitney ranks."""

from __future__ import annotations

import math


def median_type7(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    if any(math.isinf(v) for v in ordered):
        mid_lo = ordered[(n - 1) // 2]
        mid_hi = ordered[n // 2]
        if math.isinf(mid_lo) or math.isinf(mid_hi):
            return math.inf
    h = 1.0 + (n - 1) * 0.5
    lo = int(h)
    frac = h - lo
    if frac == 0.0:
        return ordered[lo - 1]
    return ordered[lo - 1] * (1.0 - frac) + ordered[lo] * frac


def fisher_two_sided_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    total_n = n_a + n_b
    denom = math.comb(total_n, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / denom

    p_obs = prob(hits_a)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    p_val = sum(
        prob(k) for k in range(lo, hi + 1) if prob(k) <= p_obs * (1.0 + 1e-12)
    )
    return min(1.0, p_val)


def fisher_loss_one_sided_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    total_n = n_a + n_b
    denom = math.comb(total_n, total_hits)
    lo = max(0, total_hits - n_b)
    hi = min(n_a, total_hits)
    p_val = sum(
        math.comb(n_a, k) * math.comb(n_b, total_hits - k) / denom
        for k in range(hits_a, hi + 1)
        if lo <= k <= hi
    )
    return min(1.0, p_val)


def mann_whitney_p(b_vals: list[float], a_vals: list[float]) -> tuple[float, float]:
    """Return (two_sided_p, one_sided_worse_p). Misses are ranked at +inf."""
    n1 = len(b_vals)
    n2 = len(a_vals)
    combined = [(val, 0) for val in b_vals] + [(val, 1) for val in a_vals]
    combined.sort(key=lambda pair: pair[0])
    n = n1 + n2
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = avg_rank
        t = j - i
        if t > 1:
            tie_term += t**3 - t
        i = j

    r1 = sum(ranks[k] for k in range(n) if combined[k][1] == 0)
    u_b = r1 - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    var = (
        (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
        if n > 1
        else 0.0
    )
    if var <= 0.0:
        return 1.0, 1.0
    sigma = math.sqrt(var)
    diff = abs(u_b - mean)
    z_two = max(0.0, diff - 0.5) / sigma
    p_two = min(1.0, math.erfc(z_two / math.sqrt(2.0)))
    z_worse = (u_b - mean - 0.5) / sigma
    p_worse = 0.5 * math.erfc(z_worse / math.sqrt(2.0))
    return p_two, min(1.0, max(0.0, p_worse))


def format_num(val: float, digits: int = 6) -> str:
    if math.isinf(val):
        return "inf"
    if math.isnan(val):
        return "nan"
    if val == int(val):
        return str(int(val))
    if abs(val - round(val)) < 1e-9 and abs(val) >= 1.0:
        return f"{val:.1f}"
    return f"{val:.{digits}g}"


def format_p(val: float) -> str:
    return f"{val:.10g}"
