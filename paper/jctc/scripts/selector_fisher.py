#!/usr/bin/env python3
"""Two-sided Fisher exact tests for posterior against uniform arm draws.

Seeds 2000-2005 are bench/depth-posterior-ablation/results.csv.
Seeds 7000-7005 are results/size-tiered-selection.csv.
LJ38 and LJ55 are the first list only.
LJ75, LJ98, and LJ104 pool both lists.
A hit is the committed hit flag. The flag is checked against a quenched
energy within 0.001 of the Cambridge minimum.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIRST = ROOT / "bench" / "depth-posterior-ablation" / "results.csv"
SECOND = ROOT / "results" / "size-tiered-selection.csv"

REFERENCE = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    98: -543.665361,
    104: -582.086642,
}
POOLED = (75, 98, 104)
FIRST_ONLY = (38, 55)
TOL = 0.001


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact probability for two hit counts."""
    misses_a = n_a - hits_a
    misses_b = n_b - hits_b
    if min(hits_a, misses_a, hits_b, misses_b) < 0:
        raise SystemExit("hit count exceeds the seed count")
    row_a = n_a
    row_b = n_b
    col_hit = hits_a + hits_b
    total = row_a + row_b
    lo = max(0, col_hit - row_b)
    hi = min(row_a, col_hit)
    ways = math.comb(total, col_hit)

    def probability(hits: int) -> float:
        return math.comb(row_a, hits) * math.comb(row_b, col_hit - hits) / ways

    observed = probability(hits_a)
    pooled = 0.0
    for hits in range(lo, hi + 1):
        probability_k = probability(hits)
        if probability_k <= observed + 1e-15:
            pooled += probability_k
    return pooled


def load_first() -> dict[tuple[int, str], list[int]]:
    grouped: dict[tuple[int, str], list[int]] = {}
    with FIRST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            size = int(row["system"].removeprefix("lj"))
            arm = row["arm"]
            if arm not in ("posterior", "uniform"):
                raise SystemExit(f"unexpected arm {arm}")
            seed = int(row["seed"])
            if seed < 2000 or seed > 2005:
                raise SystemExit(f"seed {seed} is outside 2000-2005")
            hit = check_hit(size, row["hit"], row["best"])
            grouped.setdefault((size, arm), []).append(hit)
    return grouped


def load_second() -> dict[tuple[int, str], list[int]]:
    grouped: dict[tuple[int, str], list[int]] = {}
    with SECOND.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] == "tiered":
                continue
            size = int(row["n"])
            arm = row["arm"]
            if arm not in ("posterior", "uniform"):
                raise SystemExit(f"unexpected arm {arm}")
            seed = int(row["seed"])
            if seed < 7000 or seed > 7005:
                raise SystemExit(f"seed {seed} is outside 7000-7005")
            hit = check_hit(size, row["hit"], row["best"])
            grouped.setdefault((size, arm), []).append(hit)
    return grouped


def check_hit(size: int, flag: str, best: str) -> int:
    if flag not in ("0", "1"):
        raise SystemExit(f"hit flag {flag!r} is not 0 or 1")
    recorded = int(flag)
    gap = abs(float(best) - REFERENCE[size])
    within = gap < TOL
    if recorded != int(within):
        raise SystemExit(
            f"LJ{size} hit flag {recorded} disagrees with gap {gap:.8f}"
        )
    return recorded


def counts(rows: list[int]) -> tuple[int, int]:
    if len(rows) != 6:
        raise SystemExit(f"expected 6 seeds, found {len(rows)}")
    return sum(rows), len(rows)


def format_probability(probability: float) -> str:
    if abs(probability - 1.0) < 1e-9:
        return "1"
    return f"{probability:.6f}"


def main() -> None:
    first = load_first()
    second = load_second()
    for size in FIRST_ONLY + POOLED:
        for arm in ("posterior", "uniform"):
            if (size, arm) not in first:
                raise SystemExit(f"missing lj{size} {arm} on seeds 2000-2005")
    for size in POOLED:
        for arm in ("posterior", "uniform"):
            if (size, arm) not in second:
                raise SystemExit(f"missing LJ{size} {arm} on seeds 7000-7005")

    print("size posterior uniform fisher_two_sided seeds")
    for size in FIRST_ONLY:
        post, n_post = counts(first[(size, "posterior")])
        uniform, n_uniform = counts(first[(size, "uniform")])
        probability = fisher_two_sided(post, n_post, uniform, n_uniform)
        print(
            f"{size} {post}/{n_post} {uniform}/{n_uniform} "
            f"{format_probability(probability)} 6"
        )
    for size in POOLED:
        post_a, n_a = counts(first[(size, "posterior")])
        post_b, n_b = counts(second[(size, "posterior")])
        uni_a, m_a = counts(first[(size, "uniform")])
        uni_b, m_b = counts(second[(size, "uniform")])
        post = post_a + post_b
        uniform = uni_a + uni_b
        n_post = n_a + n_b
        n_uniform = m_a + m_b
        probability = fisher_two_sided(post, n_post, uniform, n_uniform)
        print(
            f"{size} {post}/{n_post} {uniform}/{n_uniform} "
            f"{format_probability(probability)} 12"
        )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
