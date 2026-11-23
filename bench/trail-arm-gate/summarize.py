#!/usr/bin/env python3
"""Analyze the preregistered trail-arm gate."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
FIELDS = (
    "cell",
    "n",
    "hits_a",
    "hits_a2",
    "hits_b",
    "fisher_a_greater_p",
    "mannwhitney_b_more_calls_p",
    "outside_b_a",
    "outside_b_a_ci_low",
    "outside_b_a_ci_high",
    "outside_a2_a",
    "outside_a2_a_ci_low",
    "outside_a2_a_ci_high",
    "time_limit",
    "time_adjusted_limit",
    "time_adjusted_pass",
    "rss_b_a",
    "rss_b_a_ci_low",
    "rss_b_a_ci_high",
    "rss_a2_a",
    "rss_a2_a_ci_low",
    "rss_a2_a_ci_high",
    "memory_limit",
    "b_stall_triggers",
    "b_trail_redraws",
    "short_seeds",
    "short_identical",
    "hit_clause",
    "time_clause",
    "memory_clause",
    "verdict",
)


def type7(values: list[float], probability: float = 0.5) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction


def ratio(left: list[float], right: list[float]) -> float:
    denominator = type7(left)
    numerator = type7(right)
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def bootstrap(
    cell: str,
    label: str,
    left: list[float],
    right: list[float],
) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("paired samples differ")
    seed = int.from_bytes(
        hashlib.sha256(f"{cell}:{label}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAPS):
        indices = [rng.randrange(len(left)) for _ in left]
        estimates.append(
            ratio(
                [left[index] for index in indices],
                [right[index] for index in indices],
            )
        )
    return type7(estimates, 0.025), type7(estimates, 0.975)


def fisher_a_greater(hits_a: int, hits_b: int, n: int) -> float:
    total = hits_a + hits_b
    denominator = math.comb(2 * n, total)

    def probability(value: int) -> float:
        other = total - value
        if not 0 <= other <= n:
            return 0.0
        return math.comb(n, value) * math.comb(n, other) / denominator

    return min(1.0, sum(probability(value) for value in range(hits_a, n + 1)))


def mannwhitney_b_greater(a: list[float], b: list[float]) -> float:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2
        width = end - index + 1
        tie_term += width**3 - width
        index = end + 1
    n_a = len(a)
    n_b = len(b)
    rank_b = sum(ranks[value] for value in b)
    u_b = rank_b - n_b * (n_b + 1) / 2
    mean = n_a * n_b / 2
    n = n_a + n_b
    variance = n_a * n_b / 12 * (
        n + 1 - tie_term / (n * (n - 1))
    )
    if variance <= 0:
        return 1.0
    z_value = (u_b - mean - 0.5) / math.sqrt(variance)
    return 0.5 * math.erfc(z_value / math.sqrt(2))


def fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.8g}"


def summarize_cell(
    cell: str,
    budget: int,
    rows: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    by = {
        (row["arm"], int(row["seed"])): row
        for row in rows
        if row["cell"] == cell
    }
    groups = {
        arm: [by[(arm, seed)] for seed in gate.SEEDS]
        for arm in gate.ARMS
    }
    if any(
        int(row["exit_code"]) != 0 or not row["energy_bits"]
        for arm in gate.ARMS
        for row in groups[arm]
    ):
        raise ValueError(f"{cell} has failed rows")
    hits = {
        arm: sum(int(row["hit"]) for row in groups[arm])
        for arm in gate.ARMS
    }
    scored_calls = {
        arm: [
            float(row["calls"]) if int(row["hit"]) else float(budget + 1)
            for row in groups[arm]
        ]
        for arm in gate.ARMS
    }
    outside = {
        arm: [float(row["outside_s"]) for row in groups[arm]]
        for arm in gate.ARMS
    }
    rss = {
        arm: [float(row["rss_anon_kb"]) for row in groups[arm]]
        for arm in gate.ARMS
    }
    time_b_low, time_b_high = bootstrap(
        cell,
        "outside-b-a",
        outside["A"],
        outside["B"],
    )
    time_a2_low, time_a2_high = bootstrap(
        cell,
        "outside-a2-a",
        outside["A"],
        outside["A2"],
    )
    rss_b_low, rss_b_high = bootstrap(
        cell,
        "rss-b-a",
        rss["A"],
        rss["B"],
    )
    rss_a2_low, rss_a2_high = bootstrap(
        cell,
        "rss-a2-a",
        rss["A"],
        rss["A2"],
    )
    short_rows = []
    for seed in gate.SEEDS:
        a = by[("A", seed)]
        b = by[("B", seed)]
        if int(b["stall_triggers"]) == 0:
            energy_equal = a["energy_bits"] == b["energy_bits"]
            calls_equal = a["calls"] == b["calls"]
            short_rows.append(
                {
                    "cell": cell,
                    "seed": str(seed),
                    "energy_bits_equal": str(int(energy_equal)),
                    "calls_equal": str(int(calls_equal)),
                    "identical": str(int(energy_equal and calls_equal)),
                }
            )
    short_identical = all(
        int(row["identical"]) for row in short_rows
    )
    hit_clause = hits["B"] >= hits["A"]
    time_clause = time_b_high <= 1.10
    adjusted_limit = max(1.10, time_a2_high)
    memory_limit = max(1.10, rss_a2_high)
    memory_clause = not (rss_b_low > memory_limit)
    verdict = hit_clause and time_clause and memory_clause
    summary = {
        "cell": cell,
        "n": str(len(gate.SEEDS)),
        "hits_a": str(hits["A"]),
        "hits_a2": str(hits["A2"]),
        "hits_b": str(hits["B"]),
        "fisher_a_greater_p": fmt(
            fisher_a_greater(hits["A"], hits["B"], len(gate.SEEDS))
        ),
        "mannwhitney_b_more_calls_p": fmt(
            mannwhitney_b_greater(scored_calls["A"], scored_calls["B"])
        ),
        "outside_b_a": fmt(ratio(outside["A"], outside["B"])),
        "outside_b_a_ci_low": fmt(time_b_low),
        "outside_b_a_ci_high": fmt(time_b_high),
        "outside_a2_a": fmt(ratio(outside["A"], outside["A2"])),
        "outside_a2_a_ci_low": fmt(time_a2_low),
        "outside_a2_a_ci_high": fmt(time_a2_high),
        "time_limit": "1.1",
        "time_adjusted_limit": fmt(adjusted_limit),
        "time_adjusted_pass": "PASS" if time_b_high <= adjusted_limit else "FAIL",
        "rss_b_a": fmt(ratio(rss["A"], rss["B"])),
        "rss_b_a_ci_low": fmt(rss_b_low),
        "rss_b_a_ci_high": fmt(rss_b_high),
        "rss_a2_a": fmt(ratio(rss["A"], rss["A2"])),
        "rss_a2_a_ci_low": fmt(rss_a2_low),
        "rss_a2_a_ci_high": fmt(rss_a2_high),
        "memory_limit": fmt(memory_limit),
        "b_stall_triggers": str(
            sum(int(row["stall_triggers"]) for row in groups["B"])
        ),
        "b_trail_redraws": str(
            sum(int(row["trail_redraws"]) for row in groups["B"])
        ),
        "short_seeds": str(len(short_rows)),
        "short_identical": "PASS" if short_identical else "FAIL",
        "hit_clause": "PASS" if hit_clause else "FAIL",
        "time_clause": "PASS" if time_clause else "FAIL",
        "memory_clause": "PASS" if memory_clause else "FAIL",
        "verdict": "PASS" if verdict else "FAIL",
    }
    return summary, short_rows


def write_report(summaries: list[dict[str, str]]) -> None:
    overall = all(row["verdict"] == "PASS" for row in summaries)
    lines = [
        "# Trail-arm gate summary",
        "",
        f"Overall verdict: **{'PASS' if overall else 'FAIL'}**.",
        "",
        "| Cell | Hits A/B | Fisher | MW calls | Time B/A CI | "
        "A2/A high | RSS B/A CI | RSS limit | Redraws | "
        "Short identical | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_b']} | "
            f"{row['fisher_a_greater_p']} | "
            f"{row['mannwhitney_b_more_calls_p']} | "
            f"{row['outside_b_a']} "
            f"[{row['outside_b_a_ci_low']}, "
            f"{row['outside_b_a_ci_high']}] | "
            f"{row['outside_a2_a_ci_high']} | "
            f"{row['rss_b_a']} [{row['rss_b_a_ci_low']}, "
            f"{row['rss_b_a_ci_high']}] | "
            f"{row['memory_limit']} | {row['b_trail_redraws']} | "
            f"{row['short_seeds']} {row['short_identical']} | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "Fisher and Mann-Whitney probabilities are one-sided for A having",
        "more hits and B using more calls, respectively. Mann-Whitney uses",
        "a tie-corrected normal approximation with continuity correction.",
        "Misses are scored as budget plus one call.",
        "",
        "Time and RSS intervals are paired deterministic seed bootstraps.",
        "The time verdict uses 1.10. The adjusted time limit is reported in",
        "`summary.csv` only as a diagnostic. Memory fails only when the B/A",
        "lower bound exceeds both 1.10 and the A2/A upper bound.",
        "",
        "Short seeds have zero B stall-bandit triggers. Their final-energy",
        "bits and calls are checked in `short-pins.csv`.",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))
    print("PASS" if overall else "FAIL")


def main() -> None:
    with (HERE / "rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summaries = []
    short_rows = []
    for cell, budget, _ in gate.cells():
        summary, short = summarize_cell(cell, budget, rows)
        summaries.append(summary)
        short_rows.extend(short)
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    with (HERE / "short-pins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cell",
                "seed",
                "energy_bits_equal",
                "calls_equal",
                "identical",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(short_rows)
    write_report(summaries)


if __name__ == "__main__":
    main()
