#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fisher exact test, Mann-Whitney U, and a bootstrap median-ratio interval.

`calls` in rows.csv already stores a miss as budget plus 1. The median
ratio is the crest median divided by the anneal median. The bootstrap
draws 10000 resamples from seed 20261123, anneal indices first.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
OUT = HERE / "stats.json"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_123
ARMS = ("anneal", "crest")


def fisher_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, summing tables no more likely than the observed."""
    hits = hits_a + hits_b
    total = n_a + n_b
    ways = math.comb(total, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """U for the first sample, and the two-sided tie-corrected normal p with continuity."""
    n1, n2 = len(a), len(b)
    pooled = sorted(list(a) + list(b))
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
    u = sum(rank[v] for v in a) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = abs(u - n1 * n2 / 2.0) - 0.5
    p = min(1.0, math.erfc(max(dev, 0.0) / math.sqrt(var) / math.sqrt(2.0)))
    return u, p


def load_calls() -> dict[str, list[int]]:
    grouped = {arm: [] for arm in ARMS}
    hits = {arm: [] for arm in ARMS}
    with ROWS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            if arm not in grouped:
                raise SystemExit(f"unexpected arm {arm!r}")
            if row["machine_id"] != "unrecorded":
                raise SystemExit("machine_id is not the literal unrecorded")
            grouped[arm].append(int(row["calls"]))
            hits[arm].append(int(row["hit"]))
    return {"calls": grouped, "hits": hits}


def bootstrap_median_ratio(anneal: list[int], crest: list[int]) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    left = np.asarray(anneal, dtype=float)
    right = np.asarray(crest, dtype=float)
    left_index = rng.integers(0, left.size, size=(BOOTSTRAP_SAMPLES, left.size))
    right_index = rng.integers(0, right.size, size=(BOOTSTRAP_SAMPLES, right.size))
    ratios = np.median(right[right_index], axis=1) / np.median(left[left_index], axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975], method="linear")
    point = float(np.median(right) / np.median(left))
    return point, float(low), float(high)


def summary() -> dict:
    loaded = load_calls()
    calls = loaded["calls"]
    hit_flags = loaded["hits"]
    counts = {arm: len(calls[arm]) for arm in ARMS}
    hit_counts = {arm: int(sum(hit_flags[arm])) for arm in ARMS}
    miss_counts = {arm: counts[arm] - hit_counts[arm] for arm in ARMS}
    anneal_hits, crest_hits = hit_counts["anneal"], hit_counts["crest"]
    anneal_n, crest_n = counts["anneal"], counts["crest"]
    odds = (anneal_hits * miss_counts["crest"]) / (miss_counts["anneal"] * crest_hits)
    u, mann_p = mann_whitney([float(v) for v in calls["anneal"]], [float(v) for v in calls["crest"]])
    point, low, high = bootstrap_median_ratio(calls["anneal"], calls["crest"])
    return {
        "arms": list(ARMS),
        "n": {**counts, "rows": sum(counts.values())},
        "hits": hit_counts,
        "misses": miss_counts,
        "calls": calls,
        "fisher": {
            "alternative": "two-sided",
            "table_rows": list(ARMS),
            "table_columns": ["hit", "miss"],
            "table": [
                [anneal_hits, miss_counts["anneal"]],
                [crest_hits, miss_counts["crest"]],
            ],
            "odds_ratio": odds,
            "p": fisher_p(anneal_hits, anneal_n, crest_hits, crest_n),
        },
        "mann_whitney": {
            "alternative": "two-sided",
            "continuity": True,
            "tie_correction": True,
            "sample_order": list(ARMS),
            "u": u,
            "p": mann_p,
        },
        "median_calls": {
            "anneal": float(np.median(np.asarray(calls["anneal"], dtype=float))),
            "crest": float(np.median(np.asarray(calls["crest"], dtype=float))),
        },
        "median_ratio": {
            "definition": "median crest calls divided by median anneal calls",
            "point": point,
            "ci95": [low, high],
            "resamples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "quantile": "linear",
            "draw": "anneal indices, then crest indices",
        },
    }


def main(argv: list[str]) -> None:
    path = OUT
    if argv:
        if len(argv) == 2 and argv[0] == "--out":
            path = Path(argv[1])
        else:
            raise SystemExit("usage: stats.py [--out PATH]")
    payload = summary()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    fisher = payload["fisher"]
    mann = payload["mann_whitney"]
    ratio = payload["median_ratio"]
    print(f"ROWS {payload['n']['rows']}")
    print(f"HITS anneal {payload['hits']['anneal']} crest {payload['hits']['crest']}")
    print(f"FISHER_P {fisher['p']!r}")
    print(f"MWU_U {mann['u']!r}")
    print(f"MWU_P {mann['p']!r}")
    print(f"RATIO {ratio['point']!r}")
    print(f"BOOT {ratio['ci95'][0]!r} {ratio['ci95'][1]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
