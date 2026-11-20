#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the pre-registered gate from raw rows."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import statistics
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
SUMMARY_FIELDS = (
    "cell", "n", "budget", "reference", "hits_a", "hits_a2", "hits_b",
    "fisher_one_sided_p", "outside_per_call_median_a",
    "outside_per_call_median_a2", "outside_per_call_median_b",
    "outside_ratio_b_a", "outside_ratio_b_a_ci_low",
    "outside_ratio_b_a_ci_high", "outside_ratio_a2_a",
    "outside_ratio_a2_a_ci_low", "outside_ratio_a2_a_ci_high",
    "mannwhitney_b_slower_p", "rss_anon_median_a",
    "rss_anon_median_a2", "rss_anon_median_b", "rss_b_in_a_a2_band",
    "ru_maxrss_median_a", "ru_maxrss_median_a2", "ru_maxrss_median_b",
    "median_time_to_hit_a", "median_time_to_hit_a2",
    "median_time_to_hit_b", "failed_runs", "hit_clause", "cost_clause",
    "verdict",
)


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = 1 + (len(ordered) - 1) * p
    lower = int(position)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower - 1]
    return (
        ordered[lower - 1] * (1 - fraction)
        + ordered[lower] * fraction
    )


def ratio(left: list[float], right: list[float]) -> float:
    denominator = type7(left)
    numerator = type7(right)
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def fisher_a_greater(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def probability(a_hits: int) -> float:
        b_hits = total_hits - a_hits
        if not 0 <= b_hits <= n_b:
            return 0.0
        return (
            math.comb(n_a, a_hits)
            * math.comb(n_b, b_hits)
            / ways
        )

    return min(
        1.0,
        sum(probability(value) for value in range(hits_a, n_a + 1)),
    )


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def mannwhitney_b_slower(a: list[float], b: list[float]) -> float:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + 1 + end + 1) / 2
        width = end - index + 1
        tie_term += width**3 - width
        index = end + 1
    n_a, n_b = len(a), len(b)
    rank_b = sum(ranks[value] for value in b)
    u_b = rank_b - n_b * (n_b + 1) / 2
    mean = n_a * n_b / 2
    n = n_a + n_b
    variance = n_a * n_b / 12 * (
        (n + 1) - tie_term / (n * (n - 1))
    )
    if variance <= 0:
        return 1.0
    return normal_sf((u_b - mean - 0.5) / math.sqrt(variance))


def bootstrap_ratio(
    cell: str,
    label: str,
    left: list[float],
    right: list[float],
) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap needs equal samples")
    seed = int.from_bytes(
        hashlib.sha256(f"{cell}:{label}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        indices = [rng.randrange(len(left)) for _ in left]
        sample_left = [left[index] for index in indices]
        sample_right = [right[index] for index in indices]
        values.append(ratio(sample_left, sample_right))
    return type7(values, 0.025), type7(values, 0.975)


def energy_hit(row: dict[str, str], reference: float) -> bool:
    return bool(row["energy"]) and float(row["energy"]) <= reference + gate.HIT_TOL


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.8g}"


def summarize_cell(
    cell: str,
    budget: int,
    reference: float,
    seeds: tuple[int, ...],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    by = {
        (row["arm"], int(row["seed"])): row
        for row in rows
        if row["cell"] == cell
    }
    groups = {
        arm: [by[(arm, seed)] for seed in seeds if (arm, seed) in by]
        for arm in gate.ARMS
    }
    complete = all(len(groups[arm]) == len(seeds) for arm in gate.ARMS)
    failed = sum(
        int(row["exit_code"]) != 0 or not row["energy"]
        for arm in gate.ARMS
        for row in groups[arm]
    )
    hits = {
        arm: sum(energy_hit(row, reference) for row in groups[arm])
        for arm in gate.ARMS
    }
    fisher = fisher_a_greater(
        hits["A"], len(groups["A"]), hits["B"], len(groups["B"])
    ) if complete else math.nan

    valid_cost = complete and failed == 0
    outside = {
        arm: [float(row["per_call_s"]) for row in groups[arm]]
        for arm in gate.ARMS
    } if valid_cost else {arm: [] for arm in gate.ARMS}
    if valid_cost:
        ratio_b = ratio(outside["A"], outside["B"])
        ratio_a2 = ratio(outside["A"], outside["A2"])
        b_low, b_high = bootstrap_ratio(
            cell, "b-a", outside["A"], outside["B"]
        )
        a2_low, a2_high = bootstrap_ratio(
            cell, "a2-a", outside["A"], outside["A2"]
        )
        mw = mannwhitney_b_slower(outside["A"], outside["B"])
    else:
        ratio_b = ratio_a2 = b_low = b_high = a2_low = a2_high = math.nan
        mw = math.nan

    rss = {
        arm: type7([float(row["rss_anon_peak_kb"]) for row in groups[arm]])
        if groups[arm] else math.nan
        for arm in gate.ARMS
    }
    rss_band = (
        complete
        and min(rss["A"], rss["A2"]) <= rss["B"] <= max(rss["A"], rss["A2"])
    )
    ru = {
        arm: type7([float(row["ru_maxrss_kb"]) for row in groups[arm]])
        if groups[arm] else math.nan
        for arm in gate.ARMS
    }
    time_to_hit = {
        arm: [
            float(row["wall_s"]) if energy_hit(row, reference) else math.inf
            for row in groups[arm]
        ]
        for arm in gate.ARMS
    }
    hit_clause = complete and hits["B"] >= hits["A"]
    cost_clause = valid_cost and ratio_b <= gate.MAX_OUTSIDE_RATIO
    verdict = "PASS" if hit_clause and cost_clause else "FAIL"
    return {
        "cell": cell,
        "n": str(len(seeds)),
        "budget": str(budget),
        "reference": repr(reference),
        "hits_a": str(hits["A"]),
        "hits_a2": str(hits["A2"]),
        "hits_b": str(hits["B"]),
        "fisher_one_sided_p": fmt(fisher),
        "outside_per_call_median_a": (
            fmt(type7(outside["A"])) if valid_cost else ""
        ),
        "outside_per_call_median_a2": (
            fmt(type7(outside["A2"])) if valid_cost else ""
        ),
        "outside_per_call_median_b": (
            fmt(type7(outside["B"])) if valid_cost else ""
        ),
        "outside_ratio_b_a": fmt(ratio_b),
        "outside_ratio_b_a_ci_low": fmt(b_low),
        "outside_ratio_b_a_ci_high": fmt(b_high),
        "outside_ratio_a2_a": fmt(ratio_a2),
        "outside_ratio_a2_a_ci_low": fmt(a2_low),
        "outside_ratio_a2_a_ci_high": fmt(a2_high),
        "mannwhitney_b_slower_p": fmt(mw),
        "rss_anon_median_a": fmt(rss["A"]),
        "rss_anon_median_a2": fmt(rss["A2"]),
        "rss_anon_median_b": fmt(rss["B"]),
        "rss_b_in_a_a2_band": str(int(rss_band)),
        "ru_maxrss_median_a": fmt(ru["A"]),
        "ru_maxrss_median_a2": fmt(ru["A2"]),
        "ru_maxrss_median_b": fmt(ru["B"]),
        "median_time_to_hit_a": fmt(type7(time_to_hit["A"])),
        "median_time_to_hit_a2": fmt(type7(time_to_hit["A2"])),
        "median_time_to_hit_b": fmt(type7(time_to_hit["B"])),
        "failed_runs": str(failed),
        "hit_clause": "PASS" if hit_clause else "FAIL",
        "cost_clause": "PASS" if cost_clause else "FAIL",
        "verdict": verdict,
    }


def tip4p17_identity(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by = {
        (row["arm"], int(row["seed"])): row
        for row in rows
        if row["cell"] == "tip4p17"
    }
    output = []
    for seed in range(2000, 2004):
        a = by[("A", seed)]
        b = by[("B", seed)]
        energy = a["energy_bits"] == b["energy_bits"] != ""
        calls = a["charged"] == b["charged"] != ""
        final = a["final_sha256"] == b["final_sha256"] != ""
        output.append({
            "seed": str(seed),
            "energy_bits_equal": str(int(energy)),
            "calls_equal": str(int(calls)),
            "final_sha256_equal": str(int(final)),
            "all_equal": str(int(energy and calls and final)),
        })
    return output


def main() -> None:
    with (HERE / "rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summaries = [
        summarize_cell(cell, budget, reference, seeds, rows)
        for cell, budget, reference, seeds in gate.cells()
    ]
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    identity = tip4p17_identity(rows)
    with (HERE / "tip4p17-identity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed", "energy_bits_equal", "calls_equal",
                "final_sha256_equal", "all_equal",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(identity)
    write_report(summaries, identity)


def write_report(
    summaries: list[dict[str, str]],
    identity: list[dict[str, str]],
) -> None:
    hit_pass = all(row["hit_clause"] == "PASS" for row in summaries)
    cost_pass = all(row["cost_clause"] == "PASS" for row in summaries)
    verdict = "PASS" if hit_pass and cost_pass else "FAIL"
    lines = [
        "# PR #160 revert gate summary",
        "",
        f"Overall verdict: **{verdict}**.",
        "",
        f"Hit clause: **{'PASS' if hit_pass else 'FAIL'}**.",
        f"Outside-cost clause: **{'PASS' if cost_pass else 'FAIL'}**.",
        "",
        "| Cell | A hits | A2 hits | B hits | Fisher p | B/A outside | "
        "95% CI | A2/A outside | MWU p | RSS band | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['n']} | "
            f"{row['hits_a2']}/{row['n']} | {row['hits_b']}/{row['n']} | "
            f"{row['fisher_one_sided_p']} | {row['outside_ratio_b_a']} | "
            f"[{row['outside_ratio_b_a_ci_low']}, "
            f"{row['outside_ratio_b_a_ci_high']}] | "
            f"{row['outside_ratio_a2_a']} | "
            f"{row['mannwhitney_b_slower_p']} | "
            f"{row['rss_b_in_a_a2_band']} | {row['verdict']} |"
        )
    lines += [
        "",
        "`RSS band` is one when median B peak anonymous RSS lies between the",
        "A and A2 medians. `ru_maxrss` is present in `summary.csv` and does",
        "not gate.",
        "",
        "## TIP4P17 seeds 2000--2003",
        "",
        "| Seed | Energy bits | Calls | Final coordinates | All |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in identity:
        lines.append(
            f"| {row['seed']} | {row['energy_bits_equal']} | "
            f"{row['calls_equal']} | {row['final_sha256_equal']} | "
            f"{row['all_equal']} |"
        )
    lines += [
        "",
        "Failures and nonzero exits are retained in `rows.csv`; their count is",
        "reported per cell in `summary.csv`.",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))
    print(verdict)


if __name__ == "__main__":
    main()
