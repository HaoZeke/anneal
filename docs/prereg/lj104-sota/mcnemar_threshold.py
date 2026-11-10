#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Exact one-sided McNemar thresholds for a paired 30-seed comparison.

Among n discordant seeds, b is the number the candidate hits and
GMIN_BEST misses. Under the null, b is binomial with n trials and
probability 1/2. The one-sided p-value is the upper tail, the
probability of b or more. Alpha is 1/20. The comparison is exact:
20 times the tail is at most 2**n. For each discordant count from
0 through 30 this prints the smallest b that meets that inequality,
and the tail at that b. A blank minimum means no outcome at that
discordant count is a win.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

ALPHA = Fraction(1, 20)
DISCORDANT_MAX = 30


def tail(n: int, b: int) -> Fraction:
    """Probability of at least b successes in n fair Bernoulli trials."""
    if n == 0:
        return Fraction(1, 1)
    wins = sum(comb(n, k) for k in range(b, n + 1))
    return Fraction(wins, 1 << n)


def minimum_candidate_only(n: int) -> tuple[int, Fraction] | None:
    """Smallest candidate-only count whose exact one-sided tail is <= alpha."""
    if n <= 0:
        return None
    for b in range(0, n + 1):
        p = tail(n, b)
        if p <= ALPHA:
            return b, p
    return None


def render() -> str:
    lines = [
        "# Copyright (c) anneal developers",
        "discordant\tcandidate_only_min\tp_exact\tp_decimal",
    ]
    for n in range(0, DISCORDANT_MAX + 1):
        found = minimum_candidate_only(n)
        if found is None:
            lines.append(f"{n}\tnone\t\t")
            continue
        b, p = found
        lines.append(f"{n}\t{b}\t{p.numerator}/{p.denominator}\t{format(p, '.8f')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    print(render(), end="")


if __name__ == "__main__":
    main()
