# Copyright (c) anneal developers
"""Paired ratios against develop (arm A) for each arm in rows.csv.

Geometric means of per-seed calls to hit (evals_to_hit, a miss counts
as the budget) and of charged calls per hop, with 95 per cent
percentile bootstrap intervals over seeds (4000 resamples, seed 1).
"""
import csv
import math
import random
import sys
from collections import defaultdict

rows = defaultdict(dict)
order = []
with open(sys.argv[1] if len(sys.argv) > 1 else "rows.csv") as f:
    for r in csv.DictReader(f):
        rows[(r["system"], r["arm"])][int(r["seed"])] = r
        if (r["system"], r["arm"]) not in order:
            order.append((r["system"], r["arm"]))


def interval(values):
    logs = [math.log(v) for v in values]
    rng = random.Random(1)
    means = sorted(
        math.exp(sum(rng.choice(logs) for _ in logs) / len(logs)) for _ in range(4000)
    )
    return math.exp(sum(logs) / len(logs)), means[100], means[3899]


print("system arm n hits hits_A calls_to_hit lo hi calls_per_hop lo hi")
for system, arm in order:
    if arm == "A":
        continue
    base, mine = rows[(system, "A")], rows[(system, arm)]
    seeds = sorted(set(base) & set(mine))
    hit = interval([int(mine[s]["evals_to_hit"]) / int(base[s]["evals_to_hit"]) for s in seeds])
    per = lambda r: int(r["charged"]) / int(r["hops"])
    walk = interval([per(mine[s]) / per(base[s]) for s in seeds])
    hits = sum(int(mine[s]["hit"]) for s in seeds)
    hits_a = sum(int(base[s]["hit"]) for s in seeds)
    print(system, arm, len(seeds), hits, hits_a, *(f"{v:.3f}" for v in hit + walk))
