# Copyright (c) anneal developers
"""Paired tests for two arms run on the same seeds from the same starts.

A paired outcome table counts the seeds both arms hit, the seeds only the
first arm hits, the seeds only the second hits, and the seeds neither hits.

McNemar's exact test conditions on the discordant seeds: under the null each
is equally likely to favor either arm, so the count favoring the first arm is
binomial with p = 1/2. The two-sided p is twice the smaller tail, capped at 1.

The interval of the paired difference in hit rate is method 10 of Newcombe,
Stat. Med. 17, 2635 (1998): the two Wilson score intervals combined with the
phi correlation of the paired table, phi's numerator reduced by n/2 when
positive and set to 0 when a margin is empty.

The Wilcoxon signed-rank test drops zero differences, gives the absolute
differences mid-ranks, and takes T+, the sum of the ranks of the positive
differences. Its null distribution is the exact one over all 2^m sign
assignments of the m nonzero ranks, which holds with ties. The two-sided p is
the probability of a T+ at least as far from its null mean as the one seen.

Fisher's exact test is for two arms that do not share starts. Conditional on
the margins of the 2x2 table, the hits of the first arm are hypergeometric,
and the two-sided p sums every table no more probable than the one seen, with
a relative tolerance of 1e-12 so that ties to rounding count.

Z95 is the 0.975 quantile of the standard normal, so every interval is 95%.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist

Z95 = NormalDist().inv_cdf(0.975)


def outcome_table(first: list[bool], second: list[bool]) -> tuple[int, int, int, int]:
    """Both, first only, second only, and neither, over seeds in the same order."""
    if len(first) != len(second):
        raise ValueError("paired arms need the same seeds")
    both = sum(a and b for a, b in zip(first, second))
    only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second))
    return both, only_first, only_second, len(first) - both - only_first - only_second


def mcnemar_exact(only_first: int, only_second: int) -> float:
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**discordant)))


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = hits / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def newcombe_paired(both: int, only_first: int, only_second: int, neither: int,
                    z: float = Z95) -> tuple[float, float]:
    """Method 10 interval for the first arm's hit rate less the second's."""
    n = both + only_first + only_second + neither
    first, second = both + only_first, both + only_second
    p1, p2 = first / n, second / n
    l1, u1 = wilson(first, n, z)
    l2, u2 = wilson(second, n, z)
    margins = first * (n - first) * second * (n - second)
    if margins == 0:
        phi = 0.0
    else:
        numerator = both * neither - only_first * only_second
        if numerator > 0:
            numerator = max(numerator - n / 2, 0.0)
        phi = numerator / math.sqrt(margins)

    def root(a: float, b: float) -> float:
        return math.sqrt(max(a * a - 2 * phi * a * b + b * b, 0.0))

    d = p1 - p2
    return d - root(p1 - l1, u2 - p2), d + root(u1 - p1, p2 - l2)


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j + 2) / 2
        i = j + 1
    return ranks


def wilcoxon_exact(differences: list[float]) -> tuple[int, float, float]:
    """Nonzero pairs m, T+, and the two-sided exact p."""
    nonzero = [d for d in differences if d != 0]
    if not nonzero:
        return 0, 0.0, 1.0
    ranks = midranks([abs(d) for d in nonzero])
    t_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    doubled = [round(2 * r) for r in ranks]
    # ways[t] counts the sign assignments whose doubled T+ is t.
    ways = [1] + [0] * sum(doubled)
    for r in doubled:
        for t in range(len(ways) - 1, r - 1, -1):
            ways[t] += ways[t - r]
    center = sum(doubled)
    seen = abs(round(2 * 2 * t_plus) - center)
    as_far = sum(w for t, w in enumerate(ways) if abs(2 * t - center) >= seen)
    return len(nonzero), t_plus, float(min(Fraction(1), Fraction(as_far, 2 ** len(doubled))))


def fisher_exact(hits_first: int, n_first: int, hits_second: int, n_second: int) -> float:
    """Two-sided exact p for two unpaired arms."""
    hits = hits_first + hits_second
    total = n_first + n_second

    def probability(value: int) -> float:
        return math.comb(n_first, value) * math.comb(n_second, hits - value) / math.comb(total, hits)

    observed = probability(hits_first)
    return min(1.0, sum(
        probability(value)
        for value in range(max(0, hits - n_second), min(n_first, hits) + 1)
        if probability(value) <= observed * (1.0 + 1.0e-12)
    ))


def self_test() -> None:
    if mcnemar_exact(5, 0) != 0.0625 or mcnemar_exact(0, 0) != 1.0 or mcnemar_exact(3, 3) != 1.0:
        raise SystemExit("mcnemar")
    if outcome_table([True, True, False, False], [True, False, True, False]) != (1, 1, 1, 1):
        raise SystemExit("outcome table")
    # Newcombe (1998), paired example: 36, 12, 2, 0 gives 0.0569 to 0.3404.
    low, high = newcombe_paired(36, 12, 2, 0)
    if (round(low, 4), round(high, 4)) != (0.0569, 0.3404):
        raise SystemExit(f"paired newcombe {low} {high}")
    m, t, p = wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0])
    if (m, t, p) != (5, 15.0, 0.0625):
        raise SystemExit(f"wilcoxon {m} {t} {p}")
    m, t, p = wilcoxon_exact([0.0, 0.0])
    if (m, p) != (0, 1.0):
        raise SystemExit("wilcoxon with no nonzero pair")
    m, t, p = wilcoxon_exact([1.0, -1.0, 2.0, 2.0, -3.0])
    if (m, t, p) != (5, 8.5, 0.9375):
        raise SystemExit(f"wilcoxon with ties {m} {t} {p}")
    if abs(NormalDist().cdf(Z95) - 0.975) > 1e-12:
        raise SystemExit(f"z {Z95}")
    if fisher_exact(12, 12, 0, 12) != math.comb(24, 12) ** -1 * 2:
        raise SystemExit("fisher")


if __name__ == "__main__":
    self_test()
    print("ok")
