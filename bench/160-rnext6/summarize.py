#!/usr/bin/env python3
"""Analyze the focused TIP4P rerun using registered rules."""

from __future__ import annotations

import csv
import math
import random
import statistics
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 1606
SUMMARY_FIELDS = (
    "cell",
    "n",
    "hits_a",
    "hits_a2",
    "hits_b",
    "fisher_a_greater_p",
    "median_calls_a",
    "median_calls_a2",
    "median_calls_b",
    "mannwhitney_b_slower_p",
    "outside_ratio_b_a",
    "outside_ratio_b_a_ci_low",
    "outside_ratio_b_a_ci_high",
    "outside_ratio_a2_a",
    "outside_ratio_a2_a_ci_low",
    "outside_ratio_a2_a_ci_high",
    "time_limit",
    "time_clause",
    "adjusted_time_limit",
    "adjusted_time_clause",
    "rss_ratio_b_a",
    "rss_ratio_b_a_ci_low",
    "rss_ratio_b_a_ci_high",
    "rss_ratio_a2_a",
    "rss_ratio_a2_a_ci_low",
    "rss_ratio_a2_a_ci_high",
    "memory_limit",
    "memory_clause",
    "rss_anon_median_a",
    "rss_anon_median_a2",
    "rss_anon_median_b",
    "rigid_shell_sum",
    "rigid_recorded_sum",
    "verdict",
)
SEED_FIELDS = (
    "cell",
    "seed",
    "a_per_call_s",
    "a2_per_call_s",
    "b_first_per_call_s",
    "b_second_per_call_s",
    "b_mean_per_call_s",
    "b_a_ratio",
    "a2_a_ratio",
    "a_rss_kb",
    "a2_rss_kb",
    "b_mean_rss_kb",
    "b_a_rss_ratio",
    "a2_a_rss_ratio",
)


def type7(values: list[float], probability: float = 0.5) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def bootstrap_interval(
    values: list[float],
    generator: random.Random,
) -> tuple[float, float, float]:
    estimate = geometric_mean(values)
    size = len(values)
    samples = [
        geometric_mean([values[generator.randrange(size)] for _ in range(size)])
        for _ in range(BOOTSTRAPS)
    ]
    return estimate, type7(samples, 0.025), type7(samples, 0.975)


def fisher_a_greater(hits_a: int, hits_b: int, n: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(2 * n, total_hits)

    def probability(a_hits: int) -> float:
        b_hits = total_hits - a_hits
        if not 0 <= b_hits <= n:
            return 0.0
        return math.comb(n, a_hits) * math.comb(n, b_hits) / ways

    return min(1.0, sum(probability(value) for value in range(hits_a, n + 1)))


def mannwhitney_b_slower(a: list[float], b: list[float]) -> float:
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
    n_a, n_b = len(a), len(b)
    u_b = sum(ranks[value] for value in b) - n_b * (n_b + 1) / 2
    mean = n_a * n_b / 2
    count = n_a + n_b
    variance = n_a * n_b / 12 * (
        count + 1 - tie_term / (count * (count - 1))
    )
    if variance <= 0:
        return 1.0
    z_score = (u_b - mean - 0.5) / math.sqrt(variance)
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def fmt(value: float) -> str:
    return f"{value:.8g}"


def aggregate(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summaries: list[dict[str, str]] = []
    seed_rows: list[dict[str, str]] = []
    generator = random.Random(BOOTSTRAP_SEED)
    for cell in gate.CELLS:
        cell_rows = [row for row in rows if row["cell"] == cell]
        by: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in cell_rows:
            by.setdefault((row["arm"], int(row["seed"])), []).append(row)
        time_ba: list[float] = []
        time_a2a: list[float] = []
        rss_ba: list[float] = []
        rss_a2a: list[float] = []
        calls: dict[str, list[float]] = {"A": [], "A2": [], "B": []}
        call_scores: dict[str, list[float]] = {
            "A": [],
            "A2": [],
            "B": [],
        }
        hits: dict[str, int] = {"A": 0, "A2": 0, "B": 0}
        rss_values: dict[str, list[float]] = {"A": [], "A2": [], "B": []}
        for seed in gate.SEEDS:
            a_rows = by[("A", seed)]
            a2_rows = by[("A2", seed)]
            b_rows = by[("B", seed)]
            if len(a_rows) != 1 or len(a2_rows) != 1 or len(b_rows) != 2:
                raise RuntimeError(f"incomplete rows for {cell} seed {seed}")
            if len({row["hit"] for row in b_rows}) != 1:
                raise RuntimeError(f"B hit mismatch for {cell} seed {seed}")
            if len({row["calls"] for row in b_rows}) != 1:
                raise RuntimeError(f"B calls mismatch for {cell} seed {seed}")
            a, a2 = a_rows[0], a2_rows[0]
            a_time = float(a["per_call_s"])
            a2_time = float(a2["per_call_s"])
            b_times = [float(row["per_call_s"]) for row in b_rows]
            b_time = statistics.fmean(b_times)
            a_rss = float(a["rss_anon_end_kb"])
            a2_rss = float(a2["rss_anon_end_kb"])
            b_rsses = [float(row["rss_anon_end_kb"]) for row in b_rows]
            b_rss = statistics.fmean(b_rsses)
            time_ba.append(b_time / a_time)
            time_a2a.append(a2_time / a_time)
            rss_ba.append(b_rss / a_rss)
            rss_a2a.append(a2_rss / a_rss)
            for arm, selected in (("A", a), ("A2", a2), ("B", b_rows[0])):
                hit = int(selected["hit"])
                hits[arm] += hit
                calls[arm].append(float(selected["calls"]))
                call_scores[arm].append(
                    float(selected["calls"]) if hit else gate.BUDGET + 1
                )
            rss_values["A"].append(a_rss)
            rss_values["A2"].append(a2_rss)
            rss_values["B"].append(b_rss)
            seed_rows.append(
                {
                    "cell": cell,
                    "seed": str(seed),
                    "a_per_call_s": a["per_call_s"],
                    "a2_per_call_s": a2["per_call_s"],
                    "b_first_per_call_s": b_rows[0]["per_call_s"],
                    "b_second_per_call_s": b_rows[1]["per_call_s"],
                    "b_mean_per_call_s": fmt(b_time),
                    "b_a_ratio": fmt(time_ba[-1]),
                    "a2_a_ratio": fmt(time_a2a[-1]),
                    "a_rss_kb": fmt(a_rss),
                    "a2_rss_kb": fmt(a2_rss),
                    "b_mean_rss_kb": fmt(b_rss),
                    "b_a_rss_ratio": fmt(rss_ba[-1]),
                    "a2_a_rss_ratio": fmt(rss_a2a[-1]),
                }
            )
        b_time, b_time_low, b_time_high = bootstrap_interval(
            time_ba,
            generator,
        )
        a2_time, a2_time_low, a2_time_high = bootstrap_interval(
            time_a2a,
            generator,
        )
        b_rss, b_rss_low, b_rss_high = bootstrap_interval(
            rss_ba,
            generator,
        )
        a2_rss, a2_rss_low, a2_rss_high = bootstrap_interval(
            rss_a2a,
            generator,
        )
        adjusted_time_limit = max(1.10, a2_time_high)
        memory_limit = max(1.10, a2_rss_high)
        time_pass = b_time_high <= 1.10
        adjusted_time_pass = b_time_high <= adjusted_time_limit
        memory_pass = b_rss_low <= memory_limit
        shell_sum = sum(
            int(row["rigid_shell"]) for row in cell_rows if row["arm"] == "B"
        )
        recorded_sum = sum(
            int(row["rigid_recorded"])
            for row in cell_rows
            if row["arm"] == "B"
        )
        summaries.append(
            {
                "cell": cell,
                "n": str(len(gate.SEEDS)),
                "hits_a": str(hits["A"]),
                "hits_a2": str(hits["A2"]),
                "hits_b": str(hits["B"]),
                "fisher_a_greater_p": fmt(
                    fisher_a_greater(hits["A"], hits["B"], len(gate.SEEDS))
                ),
                "median_calls_a": fmt(type7(calls["A"])),
                "median_calls_a2": fmt(type7(calls["A2"])),
                "median_calls_b": fmt(type7(calls["B"])),
                "mannwhitney_b_slower_p": fmt(
                    mannwhitney_b_slower(
                        call_scores["A"],
                        call_scores["B"],
                    )
                ),
                "outside_ratio_b_a": fmt(b_time),
                "outside_ratio_b_a_ci_low": fmt(b_time_low),
                "outside_ratio_b_a_ci_high": fmt(b_time_high),
                "outside_ratio_a2_a": fmt(a2_time),
                "outside_ratio_a2_a_ci_low": fmt(a2_time_low),
                "outside_ratio_a2_a_ci_high": fmt(a2_time_high),
                "time_limit": "1.1",
                "time_clause": "PASS" if time_pass else "FAIL",
                "adjusted_time_limit": fmt(adjusted_time_limit),
                "adjusted_time_clause": "PASS" if adjusted_time_pass else "FAIL",
                "rss_ratio_b_a": fmt(b_rss),
                "rss_ratio_b_a_ci_low": fmt(b_rss_low),
                "rss_ratio_b_a_ci_high": fmt(b_rss_high),
                "rss_ratio_a2_a": fmt(a2_rss),
                "rss_ratio_a2_a_ci_low": fmt(a2_rss_low),
                "rss_ratio_a2_a_ci_high": fmt(a2_rss_high),
                "memory_limit": fmt(memory_limit),
                "memory_clause": "PASS" if memory_pass else "FAIL",
                "rss_anon_median_a": fmt(type7(rss_values["A"])),
                "rss_anon_median_a2": fmt(type7(rss_values["A2"])),
                "rss_anon_median_b": fmt(type7(rss_values["B"])),
                "rigid_shell_sum": str(shell_sum),
                "rigid_recorded_sum": str(recorded_sum),
                "verdict": "PASS" if time_pass and memory_pass else "FAIL",
            }
        )
    return summaries, seed_rows


def write_position(rows: list[dict[str, str]]) -> None:
    fields = ("arm", "position", "n", "median_per_call_s")
    output = []
    for arm in ("A", "A2", "B"):
        for position in range(1, 5):
            values = [
                float(row["per_call_s"])
                for row in rows
                if row["arm"] == arm and int(row["position"]) == position
            ]
            if values:
                output.append(
                    {
                        "arm": arm,
                        "position": str(position),
                        "n": str(len(values)),
                        "median_per_call_s": fmt(type7(values)),
                    }
                )
    with (HERE / "position-effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def write_report(summaries: list[dict[str, str]]) -> None:
    overall = "PASS" if all(row["verdict"] == "PASS" for row in summaries) else "FAIL"
    lines = [
        "# PR #160 focused TIP4P rerun",
        "",
        f"Overall verdict: **{overall}**.",
        "",
        "| Cell | Hits A/A2/B | Fisher p | MW p | B/A time CI | "
        "A2/A time CI | Time | Adjusted | B/A RSS CI | Memory | Verdict |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_a2']}/"
            f"{row['hits_b']} | {row['fisher_a_greater_p']} | "
            f"{row['mannwhitney_b_slower_p']} | "
            f"{row['outside_ratio_b_a']} "
            f"[{row['outside_ratio_b_a_ci_low']}, "
            f"{row['outside_ratio_b_a_ci_high']}] | "
            f"{row['outside_ratio_a2_a']} "
            f"[{row['outside_ratio_a2_a_ci_low']}, "
            f"{row['outside_ratio_a2_a_ci_high']}] | "
            f"{row['time_clause']} | {row['adjusted_time_clause']} | "
            f"{row['rss_ratio_b_a']} [{row['rss_ratio_b_a_ci_low']}, "
            f"{row['rss_ratio_b_a_ci_high']}] | "
            f"{row['memory_clause']} | {row['verdict']} |"
        )
    lines += [
        "",
        "Misses are scored as budget plus one in the Mann-Whitney test.",
        "The adjusted time comparison uses the larger of 1.10 and the",
        "A2/A confidence upper bound. Memory fails only when the B/A",
        "confidence lower bound exceeds the registered memory limit.",
        "Per-seed ratios and run-position medians are in separate CSV files.",
        "",
    ]
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    with (HERE / "rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = len(gate.CELLS) * len(gate.SEEDS) * 4
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    summaries, seed_rows = aggregate(rows)
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summaries)
    with (HERE / "per-seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SEED_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(seed_rows)
    write_position(rows)
    write_report(summaries)


if __name__ == "__main__":
    main()
