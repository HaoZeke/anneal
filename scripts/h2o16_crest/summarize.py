#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paired gaps for the (H2O)16 rows in bench/h2o16-crest/seeds.csv.

The hygienic panel replaces the original seed 15000 row with the rerun.
The original panel keeps that first row. Neither panel overwrites a row.
"""
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "bench" / "h2o16-crest" / "seeds.csv"
SUMMARY = ROOT / "bench" / "h2o16-crest" / "summary.csv"
REFERENCE = -81.396292669509
TOLERANCE = 1.0e-6
V8_BEST = -81.389905659858
BOOT_SEED = 20261110
BOOT_DRAWS = 10000
Z = 1.959963984540054


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: abs(values[i]))
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(values[order[end]]) == abs(values[order[start]]):
            end += 1
        rank = (start + 1 + end) / 2
        for pos in range(start, end):
            out[order[pos]] = rank
        start = end
    return out


def wilcoxon(gaps):
    kept = [gap for gap in gaps if gap != 0.0]
    n = len(kept)
    if n == 0:
        return n, 1.0, 0.0, 0.0, 0.0
    weight = ranks(kept)
    observed = sum(rank for gap, rank in zip(kept, weight) if gap > 0.0)
    total = n * (n + 1) / 2
    patterns = 1 << n
    low = high = 0
    for mask in range(patterns):
        score = sum(rank for bit, rank in enumerate(weight) if mask & (1 << bit))
        if score <= observed:
            low += 1
        if score >= observed:
            high += 1
    p_value = min(1.0, 2.0 * min(low, high) / patterns)
    negative = total - observed
    effect = 0.0 if observed + negative == 0 else (observed - negative) / (observed + negative)
    return n, p_value, observed, negative, effect


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def bootstrap(gaps):
    rng = random.Random(BOOT_SEED)
    n = len(gaps)
    draws = []
    for _ in range(BOOT_DRAWS):
        sample = [gaps[rng.randrange(n)] for _ in range(n)]
        draws.append(median(sample))
    draws.sort()
    return draws[int(0.025 * BOOT_DRAWS)], draws[min(BOOT_DRAWS - 1, int(0.975 * BOOT_DRAWS))]


def load():
    rows = []
    for line in SEEDS.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        seed, row, anneal, crest, awall, cwall, host = line.split(",")
        rows.append(
            {
                "seed": int(seed),
                "row": row,
                "anneal": float(anneal),
                "crest": float(crest),
                "awall": float(awall),
                "cwall": float(cwall),
                "host": host,
            }
        )
    return rows


def panel(rows, which):
    chosen = []
    for seed in range(15000, 15006):
        if which == "hygienic" and seed == 15000:
            row = next(item for item in rows if item["seed"] == seed and item["row"] == "rerun")
        else:
            row = next(item for item in rows if item["seed"] == seed and item["row"] == "original")
        chosen.append(row)
    return chosen


def line(name, rows):
    gaps = [row["anneal"] - row["crest"] for row in rows]
    n, p_value, positive, negative, effect = wilcoxon(gaps)
    low, high = bootstrap(gaps)
    anneal_hits = sum(row["anneal"] <= REFERENCE + TOLERANCE for row in rows)
    crest_hits = sum(row["crest"] <= REFERENCE + TOLERANCE for row in rows)
    return (
        f"{name},{n},{positive:.1f},{negative:.1f},{p_value:.10g},{effect:.10g},"
        f"{median(gaps):.10g},{low:.10g},{high:.10g},{anneal_hits},{crest_hits}\n"
    )


def main():
    rows = load()
    text = "panel,n,W_plus,W_minus,p,rank_biserial,median_gap_Eh,bootstrap_low_Eh,bootstrap_high_Eh,anneal_hits,crest_hits\n"
    text += line("hygienic", panel(rows, "hygienic"))
    text += line("original", panel(rows, "original"))
    SUMMARY.write_text(text)


if __name__ == "__main__":
    main()
