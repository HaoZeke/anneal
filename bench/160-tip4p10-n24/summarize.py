#!/usr/bin/env python3
"""Analyze the 24-seed focused TIP4P10 timing gate."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
T_CRITICAL_23 = 2.06865761041904
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
    "adjusted_time_limit",
    "hit_clause",
    "time_clause",
    "rss_anon_median_a",
    "rss_anon_median_a2",
    "rss_anon_median_b",
    "rss_median_ratio_b_a",
    "rss_median_ratio_a2_a",
    "rigid_shell_sum",
    "rigid_recorded_sum",
    "verdict",
)
SEED_FIELDS = (
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


def fmt(value: float) -> str:
    return f"{value:.8g}"


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


def log_t_interval(ratios: list[float]) -> tuple[float, float, float]:
    logs = [math.log(value) for value in ratios]
    mean = statistics.fmean(logs)
    half = T_CRITICAL_23 * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def main() -> None:
    rows_path = HERE / "rows-tip4p10-n24.csv"
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(gate.SEEDS) * len(gate.ORDER):
        raise RuntimeError("focused row count mismatch")
    by: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        by.setdefault((row["arm"], int(row["seed"])), []).append(row)
    hits = {"A": 0, "A2": 0, "B": 0}
    calls = {"A": [], "A2": [], "B": []}
    scores = {"A": [], "A2": [], "B": []}
    rss = {"A": [], "A2": [], "B": []}
    ratios_b_a: list[float] = []
    ratios_a2_a: list[float] = []
    seed_rows: list[dict[str, str]] = []
    for seed in gate.SEEDS:
        a_rows = by[("A", seed)]
        a2_rows = by[("A2", seed)]
        b_rows = by[("B", seed)]
        if len(a_rows) != 1 or len(a2_rows) != 1 or len(b_rows) != 2:
            raise RuntimeError(f"incomplete seed {seed}")
        if len({row["hit"] for row in b_rows}) != 1:
            raise RuntimeError(f"B hit mismatch at seed {seed}")
        if len({row["calls"] for row in b_rows}) != 1:
            raise RuntimeError(f"B calls mismatch at seed {seed}")
        a, a2 = a_rows[0], a2_rows[0]
        b_first, b_second = sorted(b_rows, key=lambda row: int(row["position"]))
        a_time = float(a["per_call_s"])
        a2_time = float(a2["per_call_s"])
        b_time = statistics.fmean(
            [float(b_first["per_call_s"]), float(b_second["per_call_s"])]
        )
        a_rss = float(a["rss_anon_end_kb"])
        a2_rss = float(a2["rss_anon_end_kb"])
        b_rss = statistics.fmean(
            [
                float(b_first["rss_anon_end_kb"]),
                float(b_second["rss_anon_end_kb"]),
            ]
        )
        ratios_b_a.append(b_time / a_time)
        ratios_a2_a.append(a2_time / a_time)
        for arm, selected in (("A", a), ("A2", a2), ("B", b_first)):
            hit = int(selected["hit"])
            value = float(selected["calls"])
            hits[arm] += hit
            calls[arm].append(value)
            scores[arm].append(value if hit else gate.base.BUDGET + 1)
        rss["A"].append(a_rss)
        rss["A2"].append(a2_rss)
        rss["B"].append(b_rss)
        seed_rows.append(
            {
                "seed": str(seed),
                "a_per_call_s": a["per_call_s"],
                "a2_per_call_s": a2["per_call_s"],
                "b_first_per_call_s": b_first["per_call_s"],
                "b_second_per_call_s": b_second["per_call_s"],
                "b_mean_per_call_s": fmt(b_time),
                "b_a_ratio": fmt(ratios_b_a[-1]),
                "a2_a_ratio": fmt(ratios_a2_a[-1]),
                "a_rss_kb": fmt(a_rss),
                "a2_rss_kb": fmt(a2_rss),
                "b_mean_rss_kb": fmt(b_rss),
            }
        )
    b_ratio, b_low, b_high = log_t_interval(ratios_b_a)
    a2_ratio, a2_low, a2_high = log_t_interval(ratios_a2_a)
    adjusted_limit = max(1.10, a2_high)
    hit_pass = hits["B"] >= hits["A"]
    time_pass = b_high <= adjusted_limit
    rss_medians = {arm: type7(values) for arm, values in rss.items()}
    summary = {
        "cell": gate.CELL,
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
            mannwhitney_b_slower(scores["A"], scores["B"])
        ),
        "outside_ratio_b_a": fmt(b_ratio),
        "outside_ratio_b_a_ci_low": fmt(b_low),
        "outside_ratio_b_a_ci_high": fmt(b_high),
        "outside_ratio_a2_a": fmt(a2_ratio),
        "outside_ratio_a2_a_ci_low": fmt(a2_low),
        "outside_ratio_a2_a_ci_high": fmt(a2_high),
        "adjusted_time_limit": fmt(adjusted_limit),
        "hit_clause": "PASS" if hit_pass else "FAIL",
        "time_clause": "PASS" if time_pass else "FAIL",
        "rss_anon_median_a": fmt(rss_medians["A"]),
        "rss_anon_median_a2": fmt(rss_medians["A2"]),
        "rss_anon_median_b": fmt(rss_medians["B"]),
        "rss_median_ratio_b_a": fmt(rss_medians["B"] / rss_medians["A"]),
        "rss_median_ratio_a2_a": fmt(rss_medians["A2"] / rss_medians["A"]),
        "rigid_shell_sum": str(
            sum(int(row["rigid_shell"]) for row in rows if row["arm"] == "B")
        ),
        "rigid_recorded_sum": str(
            sum(
                int(row["rigid_recorded"])
                for row in rows
                if row["arm"] == "B"
            )
        ),
        "verdict": "PASS" if hit_pass and time_pass else "FAIL",
    }
    write_csv(HERE / "summary-tip4p10-n24.csv", SUMMARY_FIELDS, [summary])
    write_csv(HERE / "per-seed-tip4p10-n24.csv", SEED_FIELDS, seed_rows)
    write_positions(rows)
    write_report(summary)


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_positions(rows: list[dict[str, str]]) -> None:
    output = []
    for position in range(1, 5):
        selected = [
            float(row["per_call_s"]) * 1.0e6
            for row in rows
            if int(row["position"]) == position
        ]
        output.append(
            {
                "arm": gate.ORDER[position - 1],
                "position": str(position),
                "n": str(len(selected)),
                "median_us_per_call": fmt(type7(selected)),
            }
        )
    write_csv(
        HERE / "position-tip4p10-n24.csv",
        ("arm", "position", "n", "median_us_per_call"),
        output,
    )


def write_report(summary: dict[str, str]) -> None:
    lines = [
        "# PR #160 TIP4P10 24-seed focused result",
        "",
        f"Verdict: **{summary['verdict']}**.",
        "",
        f"Hits A/A2/B: {summary['hits_a']}/{summary['hits_a2']}/"
        f"{summary['hits_b']}.",
        f"One-sided Fisher p: {summary['fisher_a_greater_p']}.",
        f"One-sided Mann-Whitney p: {summary['mannwhitney_b_slower_p']}.",
        "",
        f"B/A outside ratio: {summary['outside_ratio_b_a']} "
        f"[{summary['outside_ratio_b_a_ci_low']}, "
        f"{summary['outside_ratio_b_a_ci_high']}].",
        f"A2/A outside ratio: {summary['outside_ratio_a2_a']} "
        f"[{summary['outside_ratio_a2_a_ci_low']}, "
        f"{summary['outside_ratio_a2_a_ci_high']}].",
        f"Adjusted time limit: {summary['adjusted_time_limit']}.",
        "",
        f"Median RssAnon A/A2/B: {summary['rss_anon_median_a']}/"
        f"{summary['rss_anon_median_a2']}/"
        f"{summary['rss_anon_median_b']} KiB.",
        f"Median RssAnon B/A: {summary['rss_median_ratio_b_a']}.",
        f"Median RssAnon A2/A: {summary['rss_median_ratio_a2_a']}.",
        "",
        "Per-seed ratios and per-position medians are in separate CSV files.",
        "",
    ]
    (HERE / "SUMMARY-tip4p10-n24.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
