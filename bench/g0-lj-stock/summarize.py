# Copyright (c) anneal developers
"""Compare stock rows with the committed paper rows.

Misses are ranked at positive infinity. Medians are Hyndman-Fan
type 7. Fisher is the two-sided exact test on hits. Mann-Whitney
is the two-sided normal approximation with the tie and continuity
corrections.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
MANIFEST = HERE / "manifest.csv"
SUMMARY = HERE / "summary.csv"

FIELDS = [
    "cell",
    "n",
    "stock_hits",
    "paper_hits",
    "fisher_p",
    "stock_calls_median",
    "paper_calls_median",
    "mannwhitney_u",
    "mannwhitney_p",
    "changed",
]


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    left = ordered[lower - 1]
    right = ordered[lower]
    if math.isinf(left) or math.isinf(right):
        return math.inf
    return left * (1.0 - frac) + right * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def ranked(hits: list[bool], calls: list[float]) -> list[float]:
    return [call if hit else math.inf for hit, call in zip(hits, calls, strict=True)]


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if value >= 1000:
        return f"{value:.1f}"
    return f"{value:.6g}"


def main() -> None:
    rows = list(csv.DictReader(ROWS.open()))
    manifest = list(csv.DictReader(MANIFEST.open()))
    by_cell: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    paper: dict[str, list[dict[str, str]]] = {}
    for row in manifest:
        paper.setdefault(row["cell"], []).append(row)
    out = []
    for cell in paper:
        stock = sorted(by_cell.get(cell, []), key=lambda row: int(row["seed"]))
        base = sorted(paper[cell], key=lambda row: int(row["seed"]))
        if len(stock) != 12 or len(base) != 12:
            raise SystemExit(f"{cell}: expected 12 rows, have stock {len(stock)} paper {len(base)}")
        if [row["seed"] for row in stock] != [row["seed"] for row in base]:
            raise SystemExit(f"{cell}: seed lists differ")
        stock_hit = [row["hit"] == "1" for row in stock]
        paper_hit = [row["paper_hit"] == "1" for row in base]
        stock_calls = ranked(
            stock_hit,
            [float(row["calls_to_hit"]) if row["hit"] == "1" else math.inf for row in stock],
        )
        paper_calls = ranked(
            paper_hit,
            [float(row["paper_calls"]) if row["paper_hit"] == "1" else math.inf for row in base],
        )
        hits_s = sum(stock_hit)
        hits_p = sum(paper_hit)
        p_f = fisher(hits_s, 12, hits_p, 12)
        u, p_w = mann_whitney(stock_calls, paper_calls)
        changed = hits_s != hits_p or p_f < 0.05 or p_w < 0.05
        out.append(
            {
                "cell": cell,
                "n": "12",
                "stock_hits": str(hits_s),
                "paper_hits": str(hits_p),
                "fisher_p": f"{p_f:.6g}",
                "stock_calls_median": fmt(type7(stock_calls, 0.5)),
                "paper_calls_median": fmt(type7(paper_calls, 0.5)),
                "mannwhitney_u": f"{u:.4g}",
                "mannwhitney_p": f"{p_w:.6g}",
                "changed": "1" if changed else "0",
            }
        )
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(
            f"{row['cell']} stock {row['stock_hits']}/12 paper {row['paper_hits']}/12 "
            f"Fisher {row['fisher_p']} median {row['stock_calls_median']} "
            f"paper-median {row['paper_calls_median']} MW {row['mannwhitney_p']} "
            f"changed {row['changed']}"
        )


if __name__ == "__main__":
    sys.exit(main())
