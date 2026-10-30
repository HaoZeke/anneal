"""Hit-rate statistics shared by the GFN2 boards: Wilson, Fisher, McNemar."""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p of [[a, b], [c, d]], summing tables no likelier."""
    n1, n2, k = a + b, c + d, a + c
    total = math.comb(n1 + n2, k)

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / total

    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-7)))


def mcnemar(only_left: int, only_right: int) -> float:
    """Exact two-sided McNemar p from the two discordant counts."""
    m = only_left + only_right
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(only_left, only_right) + 1)) / 2**m
    return min(1.0, 2 * tail)
