# Copyright (c) anneal developers
"""Intervals for one session of thirty starts.

Success uses a Wilson interval.
A rate difference uses the Newcombe interval.
A median difference uses a bootstrap interval.
Fisher's exact test and a paired permutation test are reported with them.
The older six-row table is one draw per start, not a rate.
"""

import math
import random
import sys
from pathlib import Path

Z = 1.959963984540054
ARMS = ("crest", "driver-count budget", "paper4000", "engrad_matched")


def quantile(values, probability):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return float(ordered[0])
    position = (count - 1) * probability + 1.0
    lower = int(math.floor(position))
    fraction = position - lower
    left = ordered[lower - 1]
    if fraction == 0.0 or lower >= count:
        return float(left)
    right = ordered[lower]
    return left + fraction * (right - left)


def iqr(values):
    return quantile(values, 0.75) - quantile(values, 0.25)


def wilson(successes, total):
    if total == 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + Z * Z / total
    centre = (phat + Z * Z / (2.0 * total)) / denom
    half = Z * math.sqrt(phat * (1.0 - phat) / total + Z * Z / (4.0 * total * total)) / denom
    return (centre - half, centre + half)


def newcombe(success_a, total_a, success_b, total_b):
    low_a, high_a = wilson(success_a, total_a)
    low_b, high_b = wilson(success_b, total_b)
    point = success_a / total_a - success_b / total_b
    return (point, low_a - high_b, high_a - low_b)


def fisher_two_sided(success_a, fail_a, success_b, fail_b):
    table = [success_a, fail_a, success_b, fail_b]
    row1 = success_a + fail_a
    row2 = success_b + fail_b
    col1 = success_a + success_b
    total = sum(table)

    def hyper(value):
        return (
            math.comb(col1, value)
            * math.comb(total - col1, row1 - value)
            / math.comb(total, row1)
        )

    lo = max(0, row1 - (total - col1))
    hi = min(row1, col1)
    observed = hyper(success_a)
    probability = 0.0
    for value in range(lo, hi + 1):
        term = hyper(value)
        if term <= observed + 1e-15:
            probability += term
    return min(1.0, probability)


def read_numbers(path):
    rows = {arm: [] for arm in ARMS}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("summary") or line.startswith("engrad_budget") or line.startswith("seed"):
            continue
        parts = line.split("\t")
        if len(parts) < 7 or parts[1] not in rows:
            continue
        def finite(text):
            if text in ("", "nan"):
                return None
            return float(text)

        rows[parts[1]].append(
            {
                "seed": int(parts[0]),
                "calls": finite(parts[4]),
                "wall": finite(parts[5]),
                "success": int(parts[6]),
            }
        )
    return rows


def paired(rows, left, right):
    by_seed_l = {item["seed"]: item for item in rows[left]}
    by_seed_r = {item["seed"]: item for item in rows[right]}
    shared = sorted(set(by_seed_l) & set(by_seed_r))
    diffs = []
    for seed in shared:
        left_calls = by_seed_l[seed]["calls"]
        right_calls = by_seed_r[seed]["calls"]
        if left_calls is None or right_calls is None:
            continue
        diffs.append(left_calls - right_calls)
    return diffs


def bootstrap_median(diffs, draws, rng):
    if not diffs:
        return (float("nan"), float("nan"), float("nan"))
    point = quantile(diffs, 0.5)
    samples = []
    count = len(diffs)
    for _ in range(draws):
        sample = [diffs[rng.randrange(count)] for _ in range(count)]
        samples.append(quantile(sample, 0.5))
    samples.sort()
    low = samples[int(0.025 * (draws - 1))]
    high = samples[int(0.975 * (draws - 1))]
    return (point, low, high)


def permutation_p(diffs, draws, rng):
    if not diffs:
        return float("nan")
    observed = abs(sum(diffs))
    hits = 0
    for _ in range(draws):
        total = 0.0
        for value in diffs:
            total += value if rng.randrange(2) else -value
        if abs(total) >= observed - 1e-12:
            hits += 1
    return hits / draws


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: water_significance.py NUMBERS.tsv")
    rows = read_numbers(sys.argv[1])
    lines = [
        "# single draws from v2.6.3 are not a rate",
        "# walls are the one session that wrote this table",
        "arm\tn\tsuccess\tsuccess_median\tsuccess_iqr\tn_calls\tmedian_calls\tcalls_iqr\tn_wall\tmedian_wall\twall_iqr\twilson_low\twilson_high",
    ]
    summary = {}
    for arm in ARMS:
        group = rows[arm]
        count = len(group)
        hits = sum(item["success"] for item in group)
        flags = [float(item["success"]) for item in group]
        calls = [item["calls"] for item in group if item["calls"] is not None]
        walls = [item["wall"] for item in group if item["wall"] is not None]
        low, high = wilson(hits, count)
        summary[arm] = (hits, count)
        lines.append(
            "\t".join(
                [
                    arm,
                    str(count),
                    str(hits),
                    f"{quantile(flags, 0.5):.10g}",
                    f"{iqr(flags):.10g}",
                    str(len(calls)),
                    f"{quantile(calls, 0.5):.10g}" if calls else "nan",
                    f"{iqr(calls):.10g}" if calls else "nan",
                    str(len(walls)),
                    f"{quantile(walls, 0.5):.10g}" if walls else "nan",
                    f"{iqr(walls):.10g}" if walls else "nan",
                    f"{low:.10g}",
                    f"{high:.10g}",
                ]
            )
        )
    rng = random.Random(263)
    lines.append("# contrast left minus right point newcombe_low newcombe_high boot_median boot_low boot_high fisher_p permutation_p")
    for left, right in (
        ("engrad_matched", "crest"),
        ("driver-count budget", "crest"),
        ("paper4000", "crest"),
    ):
        hits_l, n_l = summary[left]
        hits_r, n_r = summary[right]
        point, nlow, nhigh = newcombe(hits_l, n_l, hits_r, n_r)
        diffs = paired(rows, left, right)
        median, blow, bhigh = bootstrap_median(diffs, 10000, rng)
        fisher = fisher_two_sided(hits_l, n_l - hits_l, hits_r, n_r - hits_r)
        perm = permutation_p(diffs, 10000, rng)
        lines.append(
            f"{left}\t{right}\t{point:.10g}\t{nlow:.10g}\t{nhigh:.10g}\t{median:.10g}\t{blow:.10g}\t{bhigh:.10g}\t{fisher:.10g}\t{perm:.10g}"
        )
    out = Path(sys.argv[1]).with_name("SIGNIFICANCE.tsv")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
