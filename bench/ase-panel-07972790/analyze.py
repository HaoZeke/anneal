#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise one ASE panel from its raw rows.

Reads rows.csv in this directory and writes summary.csv. Hits, stops,
medians, and Fisher's exact test are computed from those rows.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
SUMMARY = HERE / "summary.csv"
ARMS = ("anneal", "ase_bh", "ase_mh")
HEADER = (
    "cell,arm,n,hits,stops,failures,median_wall_s,"
    "median_inside_per_eval_s,median_outside_per_eval_s,fisher_p_vs_anneal"
)


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    return float(left * (1.0 - weight) + ordered[int(lower)] * weight)


def median(values: list[float]) -> float:
    if not values:
        raise SystemExit("median of an empty sample")
    return type7(sorted(values), 0.5)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact test on hit counts.

    The table is hits versus non-hits for the two arms. Non-hits are the
    completed runs that did not hit. Probability mass is the hypergeometric
    probability of every table with the same margins; the p-value sums the
    tables whose probability is at most the observed one.
    """
    if not (0 <= hits_a <= n_a and 0 <= hits_b <= n_b):
        raise SystemExit("hit counts outside the completed runs")
    a_miss = n_a - hits_a
    b_miss = n_b - hits_b
    row_a = n_a
    col_hit = hits_a + hits_b
    total = n_a + n_b
    if total == 0 or col_hit == 0 or col_hit == total:
        return 1.0

    def probability(value: int) -> float:
        other = col_hit - value
        if other < 0 or other > n_b or value > row_a:
            return 0.0
        return (
            math.comb(row_a, value)
            * math.comb(n_b, other)
            / math.comb(total, col_hit)
        )

    observed = probability(hits_a)
    low = max(0, col_hit - n_b)
    high = min(row_a, col_hit)
    total_p = sum(
        probability(value)
        for value in range(low, high + 1)
        if probability(value) <= observed * (1.0 + 1.0e-12)
    )
    return min(1.0, total_p)


def load() -> list[dict[str, str]]:
    with ROWS.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SystemExit("no rows")
    return rows


def completed(row: dict[str, str]) -> bool:
    return row["status"] != "fail" and row["wall_s"] != "" and row["force_calls"] != ""


def main() -> int:
    rows = load()
    cells = list(dict.fromkeys(row["cell"] for row in rows))
    lines = [HEADER]
    for cell in cells:
        grouped = {arm: [row for row in rows if row["cell"] == cell and row["arm"] == arm] for arm in ARMS}
        stats = {}
        for arm in ARMS:
            group = grouped[arm]
            done = [row for row in group if completed(row)]
            failures = sum(row["status"] == "fail" for row in group)
            hits = sum(row["hit"] == "true" for row in done)
            stops = sum(row["capped"] == "true" for row in done)
            walls = [float(row["wall_s"]) for row in done]
            inside = [float(row["inside_s"]) / int(row["force_calls"]) for row in done if int(row["force_calls"]) > 0]
            outside = [float(row["outside_s"]) / int(row["force_calls"]) for row in done if int(row["force_calls"]) > 0]
            stats[arm] = {
                "n": len(done),
                "hits": hits,
                "stops": stops,
                "failures": failures,
                "wall": median(walls) if walls else None,
                "inside": median(inside) if inside else None,
                "outside": median(outside) if outside else None,
            }
        anneal = stats["anneal"]
        for arm in ARMS:
            item = stats[arm]
            if arm == "anneal" or anneal["n"] == 0 or item["n"] == 0:
                fisher = ""
            else:
                fisher = f"{fisher_two_sided(anneal['hits'], anneal['n'], item['hits'], item['n']):.6g}"
            def fmt(value: float | None) -> str:
                return "" if value is None else f"{value:.8g}"

            lines.append(
                ",".join(
                    [
                        cell,
                        arm,
                        str(item["n"]),
                        str(item["hits"]),
                        str(item["stops"]),
                        str(item["failures"]),
                        fmt(item["wall"]),
                        fmt(item["inside"]),
                        fmt(item["outside"]),
                        fisher,
                    ]
                )
            )
    text = "\n".join(lines) + "\n"
    if "--check" in sys.argv:
        if not SUMMARY.exists() or SUMMARY.read_text() != text:
            raise SystemExit("summary.csv does not match the raw rows")
        return 0
    SUMMARY.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
