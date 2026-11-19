# Copyright (c) anneal developers
"""Declared timing rule for the nm3-land part B outside-potential recheck.

The rule is fixed here, before any measurement row is recorded.

Cells are blj115n38, blj105n38, and sc126-55. Seeds are 2100-2123
(24 seeds), disjoint from the part B gate seeds 2000-2011. Arms are A,
B, and an independent rebuild A2 of arm A. For each seed the three arms
run back to back. Seed i uses permutation i mod 6 of the Williams
rotation below, so each arm occupies each slot equally often and each
ordered pair of distinct arms is an immediate successor equally often.

For one cell, and for a numerator arm X in {B, A2},

    R(X/A) = (sum outside_s(X) / sum charged(X))
             / (sum outside_s(A) / sum charged(A))

summed over the 24 seeds. The 95% interval is a percentile bootstrap:
10000 resamples, seeds drawn with replacement, numerator and denominator
taken from the same seed (paired), generator random.Random(12345) and
randrange. Endpoints are the Hyndman-Fan type 7 quantiles at 0.025 and
0.975. The per-seed median ratio is the type 7 median of the 24
per-seed outside-per-call ratios.

A cell passes if the lower endpoint of the B/A interval is <= 1.10, or
if the B/A point ratio lies inside the A2/A interval, endpoints
included. Otherwise the cell fails.
"""

from __future__ import annotations

import math
import random
from collections import Counter

CELLS: tuple[str, ...] = ("blj115n38", "blj105n38", "sc126-55")
SEEDS: tuple[int, ...] = tuple(range(2100, 2124))
ARMS: tuple[str, ...] = ("A", "B", "A2")

# Six permutations. Across one cycle each arm is twice in each slot, and
# each ordered pair of distinct arms occurs twice as a successor.
PERMUTATIONS: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "A2"),
    ("B", "A2", "A"),
    ("A2", "A", "B"),
    ("A", "A2", "B"),
    ("A2", "B", "A"),
    ("B", "A", "A2"),
)

THRESHOLD = 1.10
N_RESAMPLES = 10000
RNG_SEED = 12345
BOOTSTRAP_ALPHA = 0.05


def type7(values: list[float], p: float = 0.5) -> float:
    """Hyndman-Fan type 7 quantile."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("empty sample")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return ordered[idx]
    return ordered[idx] * (1.0 - frac) + ordered[idx + 1] * frac


def order_for_index(index: int) -> tuple[str, str, str]:
    if index < 0:
        raise ValueError(f"negative schedule index {index}")
    return PERMUTATIONS[index % len(PERMUTATIONS)]


def order_for_seed(seed: int) -> tuple[str, str, str]:
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is outside {SEEDS[0]}-{SEEDS[-1]}")
    return order_for_index(seed - SEEDS[0])


def order_tag(order: tuple[str, ...]) -> str:
    return "+".join(order)


def sum_ratio(
    outside_num: list[float],
    charged_num: list[int],
    outside_den: list[float],
    charged_den: list[int],
) -> float:
    """(sum outside_num / sum charged_num) / (sum outside_den / sum charged_den)."""
    n = len(outside_num)
    if n == 0 or len(charged_num) != n or len(outside_den) != n or len(charged_den) != n:
        raise ValueError("ratio inputs must be non-empty and aligned")
    num_calls = sum(charged_num)
    den_calls = sum(charged_den)
    if num_calls <= 0 or den_calls <= 0:
        raise ValueError("charged sum is zero")
    return (sum(outside_num) / num_calls) / (sum(outside_den) / den_calls)


def per_seed_ratios(
    outside_num: list[float],
    charged_num: list[int],
    outside_den: list[float],
    charged_den: list[int],
) -> list[float]:
    ratios: list[float] = []
    for outside_n, charged_n, outside_d, charged_d in zip(
        outside_num, charged_num, outside_den, charged_den, strict=True
    ):
        if charged_n <= 0 or charged_d <= 0:
            raise ValueError("per-seed charged is zero")
        ratios.append((outside_n / charged_n) / (outside_d / charged_d))
    return ratios


def median_per_seed_ratio(
    outside_num: list[float],
    charged_num: list[int],
    outside_den: list[float],
    charged_den: list[int],
) -> float:
    return type7(per_seed_ratios(outside_num, charged_num, outside_den, charged_den), 0.5)


def bootstrap_ci(
    outside_num: list[float],
    charged_num: list[int],
    outside_den: list[float],
    charged_den: list[int],
    n_resamples: int = N_RESAMPLES,
    rng_seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Paired percentile bootstrap interval, type 7 endpoints.

    Each resample draws len(data) seeds with replacement via
    random.Random(rng_seed).randrange, and uses that same index for the
    numerator and the denominator.
    """
    n = len(outside_num)
    if n == 0 or len(charged_num) != n or len(outside_den) != n or len(charged_den) != n:
        raise ValueError("bootstrap inputs must be non-empty and aligned")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(rng_seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(
            sum_ratio(
                [outside_num[i] for i in idx],
                [charged_num[i] for i in idx],
                [outside_den[i] for i in idx],
                [charged_den[i] for i in idx],
            )
        )
    tail = BOOTSTRAP_ALPHA / 2.0
    return type7(samples, tail), type7(samples, 1.0 - tail)


def verdict(ba_point: float, ba_lo: float, aa_lo: float, aa_hi: float) -> str:
    """PASS if ba_lo <= 1.10 or ba_point is inside the A2/A interval."""
    if ba_lo <= THRESHOLD or (aa_lo <= ba_point <= aa_hi):
        return "PASS"
    return "FAIL"


def _aligned(rows: list[dict[str, str]], cell: str, arm: str) -> list[dict[str, str]]:
    found: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["system"] != cell or row["arm"] != arm:
            continue
        seed = int(row["seed"])
        if seed in found:
            raise ValueError(f"duplicate row {cell} {arm} seed {seed}")
        found[seed] = row
    missing = [seed for seed in SEEDS if seed not in found]
    extra = [seed for seed in found if seed not in SEEDS]
    if missing or extra:
        raise ValueError(f"{cell} {arm} seed set does not match the declaration")
    return [found[seed] for seed in SEEDS]


def _series(arm_rows: list[dict[str, str]]) -> tuple[list[float], list[int]]:
    outside = [float(row["outside_s"]) for row in arm_rows]
    charged = [int(float(row["charged"])) for row in arm_rows]
    return outside, charged


def outcomes_match(left: dict[str, str], right: dict[str, str]) -> bool:
    """Recorded outcome identity: status, hit, calls, charged, and energy bits."""
    if left["status"] != "ok" or right["status"] != "ok":
        return False
    return (
        left["hit"] == right["hit"]
        and left["evals_to_hit"] == right["evals_to_hit"]
        and left["charged"] == right["charged"]
        and left["best_bits_hex"] == right["best_bits_hex"]
    )


def assert_schedule(rows: list[dict[str, str]]) -> dict[tuple[int, str], int]:
    """Require the declared rotation on every recorded row."""
    per_cell: dict[str, Counter[tuple[int, str]]] = {}
    overall: Counter[tuple[int, str]] = Counter()
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        cell = row["system"]
        arm = row["arm"]
        seed = int(row["seed"])
        slot = int(row["slot"])
        if cell not in CELLS or arm not in ARMS or seed not in SEEDS:
            raise ValueError(f"row outside the declared design: {cell} {arm} {seed}")
        key = (cell, arm, seed)
        if key in seen:
            raise ValueError(f"duplicate row {key}")
        seen.add(key)
        order = order_for_seed(seed)
        if row["arm_order"] != order_tag(order):
            raise ValueError(f"{cell} seed {seed} order {row['arm_order']} != {order_tag(order)}")
        if slot < 1 or slot > len(order) or order[slot - 1] != arm:
            raise ValueError(f"{cell} seed {seed} slot {slot} arm {arm} disagrees with {order}")
        mark = (slot, arm)
        overall[mark] += 1
        per_cell.setdefault(cell, Counter())[mark] += 1
    expect = len(SEEDS) // len(ARMS)
    for cell, counts in per_cell.items():
        for slot in (1, 2, 3):
            for arm in ARMS:
                if counts[(slot, arm)] != expect:
                    raise ValueError(f"unbalanced {cell} slot {slot} arm {arm}: {counts[(slot, arm)]}")
    return dict(overall)


def evaluate_cell(rows: list[dict[str, str]], cell: str) -> dict[str, object]:
    """Point ratios, intervals, medians, hits, calls, and the declared verdict."""
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell}")
    by_arm = {arm: _aligned(rows, cell, arm) for arm in ARMS}
    series = {arm: _series(by_arm[arm]) for arm in ARMS}
    r_ba = sum_ratio(*series["B"], *series["A"])
    ba_lo, ba_hi = bootstrap_ci(*series["B"], *series["A"])
    r_aa = sum_ratio(*series["A2"], *series["A"])
    aa_lo, aa_hi = bootstrap_ci(*series["A2"], *series["A"])
    median_ba = median_per_seed_ratio(*series["B"], *series["A"])
    median_aa = median_per_seed_ratio(*series["A2"], *series["A"])
    hits = {arm: sum(int(row["hit"]) for row in by_arm[arm]) for arm in ARMS}
    failed = {arm: sum(row["status"] != "ok" for row in by_arm[arm]) for arm in ARMS}
    charged = {arm: sum(series[arm][1]) for arm in ARMS}
    outside = {arm: sum(series[arm][0]) for arm in ARMS}
    evals_median = {
        arm: type7([float(row["evals_to_hit"]) for row in by_arm[arm]])
        for arm in ARMS
    }
    mismatches = [
        seed
        for seed, left, right in zip(SEEDS, by_arm["A"], by_arm["A2"], strict=True)
        if not outcomes_match(left, right)
    ]
    return {
        "cell": cell,
        "r_ba": r_ba,
        "ba_lo": ba_lo,
        "ba_hi": ba_hi,
        "r_aa": r_aa,
        "aa_lo": aa_lo,
        "aa_hi": aa_hi,
        "median_ba": median_ba,
        "median_aa": median_aa,
        "hits": hits,
        "failed": failed,
        "charged": charged,
        "outside": outside,
        "evals_median": evals_median,
        "identical": len(SEEDS) - len(mismatches),
        "n_seeds": len(SEEDS),
        "mismatch_seeds": mismatches,
        "verdict": verdict(r_ba, ba_lo, aa_lo, aa_hi),
    }


def overall_verdict(cell_reports: list[dict[str, object]]) -> str:
    failed = [str(report["cell"]) for report in cell_reports if report["verdict"] != "PASS"]
    if len(cell_reports) != len(CELLS):
        raise ValueError("every declared cell must be reported")
    if failed:
        return "FAIL: " + ", ".join(failed)
    return "PASS"
