# Copyright (c) anneal developers
"""Exact paired tests of each arm in rows.csv against arm B, seed by seed.

Every test enumerates all sign flips of the pairs that differ, so each p
value is exact and two-sided. Run as ``python3 bench/168-r8-m38r6/stats.py``.
"""

import csv
import itertools
import math
import statistics
from pathlib import Path


def tests(base: list[int], arm: list[int]) -> dict[str, float]:
    idx = [i for i in range(len(base)) if base[i] != arm[i]]
    d = [arm[i] - base[i] for i in idx]
    n = len(d)
    order = sorted(range(n), key=lambda k: abs(d[k]))
    rank = [0] * n
    for pos, k in enumerate(order):
        rank[k] = pos + 1
    centre = n * (n + 1) / 4
    w_obs = sum(rank[k] for k in range(n) if d[k] > 0)
    med_obs = statistics.median(arm) - statistics.median(base)
    log_obs = sum(math.log(arm[i] / base[i]) for i in idx)
    hit_w = hit_med = hit_log = 0
    for flips in itertools.product((False, True), repeat=n):
        w = sum(rank[k] for k in range(n) if (d[k] > 0) != flips[k])
        hit_w += abs(w - centre) >= abs(w_obs - centre) - 1e-9
        a, b = list(base), list(arm)
        log = 0.0
        for k, i in enumerate(idx):
            if flips[k]:
                a[i], b[i] = b[i], a[i]
            log += math.log(b[i] / a[i])
        hit_med += abs(statistics.median(b) - statistics.median(a)) >= abs(med_obs) - 1e-9
        hit_log += abs(log) >= abs(log_obs) - 1e-12
    total = 2**n
    slower = sum(x > 0 for x in d)
    tail = max(slower, n - slower)
    return {
        "differ": n,
        "slower": slower,
        "median_base": statistics.median(base),
        "median_arm": statistics.median(arm),
        "median_p": hit_med / total,
        "wilcoxon_w": w_obs,
        "wilcoxon_p": hit_w / total,
        "sign_p": min(1.0, 2 * sum(math.comb(n, j) for j in range(tail, n + 1)) / total),
        "geomean_ratio": math.exp(log_obs / len(base)),
        "geomean_p": hit_log / total,
    }


def main() -> None:
    calls: dict[str, dict[int, int]] = {}
    with open(Path(__file__).with_name("rows.csv")) as f:
        for row in csv.DictReader(f):
            assert row["hit"] == "1", row
            calls.setdefault(row["arm"], {})[int(row["seed"])] = int(row["evals_to_hit"])
    seeds = sorted(calls["B"])
    base = [calls["B"][s] for s in seeds]
    print("arm | seeds | differ | slower | median B | median arm | p median | W+ | p Wilcoxon | p sign | geo ratio | p geo")
    for arm in (a for a in calls if a != "B"):
        t = tests(base, [calls[arm][s] for s in seeds])
        print(
            f"{arm} | {len(seeds)} | {t['differ']} | {t['slower']} | {t['median_base']:.1f} | {t['median_arm']:.1f} | "
            f"{t['median_p']:.4f} | {t['wilcoxon_w']} | {t['wilcoxon_p']:.4f} | {t['sign_p']:.4f} | "
            f"{t['geomean_ratio']:.4f} | {t['geomean_p']:.4f}"
        )


if __name__ == "__main__":
    main()
