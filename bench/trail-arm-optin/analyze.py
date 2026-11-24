#!/usr/bin/env python3
"""Analyze the opt-in stall-arm gate."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261123
ARMS = ("A", "B", "A2")

SUMMARY_FIELDS = [
    "cell",
    "n",
    "hits_a",
    "hits_b",
    "hits_a2",
    "fisher_b_a",
    "fisher_a2_a",
    "median_calls_a",
    "median_calls_b",
    "median_calls_a2",
    "mwu_b_a",
    "mwu_a2_a",
    "redraws_a",
    "redraws_b",
    "redraws_a2",
    "stall_triggers_a",
    "stall_triggers_b",
    "stall_triggers_a2",
    "outside_b_a",
    "outside_b_a_lo",
    "outside_b_a_hi",
    "outside_a2_a",
    "outside_a2_a_lo",
    "outside_a2_a_hi",
    "rss_b_a",
    "rss_b_a_lo",
    "rss_b_a_hi",
    "rss_a2_a",
    "rss_a2_a_lo",
    "rss_a2_a_hi",
    "strict_time_bar",
    "noise_time_bar",
    "memory_gate",
    "trajectory_differences",
    "early_stop_pairs",
    "charged_a",
    "charged_b",
    "charged_a2",
    "wall_a",
    "wall_b",
    "wall_a2",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fisher(
    hits_left: int,
    n_left: int,
    hits_right: int,
    n_right: int,
) -> float:
    total = hits_left + hits_right
    ways = math.comb(n_left + n_right, total)

    def probability(k: int) -> float:
        return (
            math.comb(n_left, k)
            * math.comb(n_right, total - k)
            / ways
        )

    observed = probability(hits_left)
    low = max(0, total - n_right)
    high = min(n_left, total)
    return min(
        1.0,
        sum(
            probability(k)
            for k in range(low, high + 1)
            if probability(k) <= observed * (1.0 + 1e-12)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> float:
    n_left = len(left)
    n_right = len(right)
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    ties = 0.0
    index = 0
    while index < len(pooled):
        end = index
        while (
            end + 1 < len(pooled)
            and pooled[end + 1] == pooled[index]
        ):
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2.0
        count = end - index + 1
        ties += count**3 - count
        index = end + 1
    u_value = (
        sum(ranks[value] for value in left)
        - n_left * (n_left + 1) / 2.0
    )
    total = n_left + n_right
    variance = (
        n_left
        * n_right
        / 12.0
        * ((total + 1) - ties / (total * (total - 1)))
    )
    if variance <= 0.0:
        return 1.0
    deviation = max(
        abs(u_value - n_left * n_right / 2.0) - 0.5,
        0.0,
    )
    return min(
        1.0,
        math.erfc(deviation / math.sqrt(variance) / math.sqrt(2.0)),
    )


def quantile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def geometric_mean(values: list[float]) -> float:
    if any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def paired_ratio(
    left: dict[int, dict[str, str]],
    right: dict[int, dict[str, str]],
    field: str,
) -> tuple[float, float, float]:
    seeds = sorted(left)
    ratios = [
        float(right[seed][field]) / float(left[seed][field])
        for seed in seeds
    ]
    estimate = geometric_mean(ratios)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [
        geometric_mean(
            [ratios[rng.randrange(len(ratios))] for _ in ratios]
        )
        for _ in range(BOOTSTRAPS)
    ]
    return (
        estimate,
        quantile_type7(samples, 0.025),
        quantile_type7(samples, 0.975),
    )


def fmt(value: float) -> str:
    return f"{value:.6g}"


def arm_values(
    by_arm: dict[str, dict[int, dict[str, str]]],
    arm: str,
    field: str,
) -> list[float]:
    return [
        float(row[field])
        for _, row in sorted(by_arm[arm].items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.rows)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["cell"], []).append(row)

    summary: list[dict[str, str]] = []
    incomplete: list[str] = []
    hit_losses: list[str] = []
    time_failures: list[str] = []
    memory_failures: list[str] = []
    total_redraws = {arm: 0 for arm in ARMS}
    total_charged = {arm: 0 for arm in ARMS}
    total_wall = {arm: 0.0 for arm in ARMS}
    total_early = 0

    for cell, items in sorted(grouped.items()):
        by_arm: dict[str, dict[int, dict[str, str]]] = {
            arm: {} for arm in ARMS
        }
        for row in items:
            by_arm[row["arm"]][int(row["seed"])] = row
        seed_sets = [set(by_arm[arm]) for arm in ARMS]
        if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets):
            incomplete.append(cell)
            continue
        n = len(seed_sets[0])
        hits = {
            arm: sum(
                row["hit"] == "1"
                for row in by_arm[arm].values()
            )
            for arm in ARMS
        }
        calls = {
            arm: arm_values(by_arm, arm, "calls_to_hit")
            for arm in ARMS
        }
        redraws = {
            arm: sum(
                int(row["redraws"])
                for row in by_arm[arm].values()
            )
            for arm in ARMS
        }
        triggers = {
            arm: sum(
                int(row["stall_triggers"])
                for row in by_arm[arm].values()
            )
            for arm in ARMS
        }
        charged = {
            arm: sum(
                int(row["charged"])
                for row in by_arm[arm].values()
            )
            for arm in ARMS
        }
        wall = {
            arm: sum(
                float(row["outside_s"])
                for row in by_arm[arm].values()
            )
            for arm in ARMS
        }
        outside_b = paired_ratio(
            by_arm["A"],
            by_arm["B"],
            "outside_s",
        )
        outside_a2 = paired_ratio(
            by_arm["A"],
            by_arm["A2"],
            "outside_s",
        )
        rss_b = paired_ratio(
            by_arm["A"],
            by_arm["B"],
            "rss_anon_kib",
        )
        rss_a2 = paired_ratio(
            by_arm["A"],
            by_arm["A2"],
            "rss_anon_kib",
        )
        strict_bar = outside_b[2] <= 1.10
        noise_time_limit = max(1.10, outside_a2[2])
        noise_bar = outside_b[2] <= noise_time_limit
        memory_limit = max(1.10, rss_a2[2])
        memory_gate = rss_b[1] <= memory_limit
        real_time_regression = (
            outside_b[1] > 1.0
            and outside_b[1] > outside_a2[2]
        )
        if hits["B"] < hits["A"]:
            hit_losses.append(cell)
        if real_time_regression:
            time_failures.append(cell)
        if not memory_gate:
            memory_failures.append(cell)
        differences = 0
        early = 0
        for seed in sorted(by_arm["A"]):
            a_row = by_arm["A"][seed]
            b_row = by_arm["B"][seed]
            fields = (
                "hit",
                "calls_to_hit",
                "charged",
                "final_energy_bits",
                "redraws",
                "stall_triggers",
            )
            differences += int(
                any(a_row[field] != b_row[field] for field in fields)
            )
            early += int(a_row["charged"] != b_row["charged"])
        total_early += early
        for arm in ARMS:
            total_redraws[arm] += redraws[arm]
            total_charged[arm] += charged[arm]
            total_wall[arm] += wall[arm]
        summary.append(
            {
                "cell": cell,
                "n": str(n),
                "hits_a": str(hits["A"]),
                "hits_b": str(hits["B"]),
                "hits_a2": str(hits["A2"]),
                "fisher_b_a": fmt(
                    fisher(hits["B"], n, hits["A"], n)
                ),
                "fisher_a2_a": fmt(
                    fisher(hits["A2"], n, hits["A"], n)
                ),
                "median_calls_a": fmt(statistics.median(calls["A"])),
                "median_calls_b": fmt(statistics.median(calls["B"])),
                "median_calls_a2": fmt(
                    statistics.median(calls["A2"])
                ),
                "mwu_b_a": fmt(
                    mann_whitney(calls["B"], calls["A"])
                ),
                "mwu_a2_a": fmt(
                    mann_whitney(calls["A2"], calls["A"])
                ),
                "redraws_a": str(redraws["A"]),
                "redraws_b": str(redraws["B"]),
                "redraws_a2": str(redraws["A2"]),
                "stall_triggers_a": str(triggers["A"]),
                "stall_triggers_b": str(triggers["B"]),
                "stall_triggers_a2": str(triggers["A2"]),
                "outside_b_a": fmt(outside_b[0]),
                "outside_b_a_lo": fmt(outside_b[1]),
                "outside_b_a_hi": fmt(outside_b[2]),
                "outside_a2_a": fmt(outside_a2[0]),
                "outside_a2_a_lo": fmt(outside_a2[1]),
                "outside_a2_a_hi": fmt(outside_a2[2]),
                "rss_b_a": fmt(rss_b[0]),
                "rss_b_a_lo": fmt(rss_b[1]),
                "rss_b_a_hi": fmt(rss_b[2]),
                "rss_a2_a": fmt(rss_a2[0]),
                "rss_a2_a_lo": fmt(rss_a2[1]),
                "rss_a2_a_hi": fmt(rss_a2[2]),
                "strict_time_bar": "PASS" if strict_bar else "FAIL",
                "noise_time_bar": "PASS" if noise_bar else "FAIL",
                "memory_gate": "PASS" if memory_gate else "FAIL",
                "trajectory_differences": str(differences),
                "early_stop_pairs": str(early),
                "charged_a": str(charged["A"]),
                "charged_b": str(charged["B"]),
                "charged_a2": str(charged["A2"]),
                "wall_a": f"{wall['A']:.6f}",
                "wall_b": f"{wall['B']:.6f}",
                "wall_a2": f"{wall['A2']:.6f}",
            }
        )

    path_exercised = total_redraws["B"] > 0
    if incomplete or not summary:
        verdict = "PARTIAL"
        reason = "required rows are incomplete"
    elif not path_exercised:
        verdict = "PARTIAL"
        reason = "PATH NOT EXERCISED"
    elif hit_losses or time_failures or memory_failures:
        verdict = "FIX"
        failures = hit_losses + time_failures + memory_failures
        reason = "gate failure in " + ", ".join(sorted(set(failures)))
    else:
        verdict = "PASS"
        reason = "path exercised with no gate failure"

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Trail-arm opt-in gate",
        "",
        f"Verdict: **{verdict}** — {reason}.",
        "",
        (
            "Redraw totals: "
            + ", ".join(
                f"{arm}={total_redraws[arm]}" for arm in ARMS
            )
            + "."
        ),
        "",
        (
            "| cell | n | hits A/B/A2 | median calls A/B/A2 | "
            "redraws A/B/A2 | B/A time (95% CI) | "
            "A2/A time (95% CI) | B/A RSS (95% CI) | gates |"
        ),
        (
            "| --- | ---: | --- | --- | --- | --- | --- | --- | "
            "--- |"
        ),
    ]
    for row in summary:
        lines.append(
            f"| {row['cell']} | {row['n']} | "
            f"{row['hits_a']}/{row['hits_b']}/{row['hits_a2']} | "
            f"{row['median_calls_a']}/{row['median_calls_b']}/"
            f"{row['median_calls_a2']} | "
            f"{row['redraws_a']}/{row['redraws_b']}/"
            f"{row['redraws_a2']} | "
            f"{row['outside_b_a']} "
            f"[{row['outside_b_a_lo']}, {row['outside_b_a_hi']}] | "
            f"{row['outside_a2_a']} "
            f"[{row['outside_a2_a_lo']}, "
            f"{row['outside_a2_a_hi']}] | "
            f"{row['rss_b_a']} "
            f"[{row['rss_b_a_lo']}, {row['rss_b_a_hi']}] | "
            f"strict {row['strict_time_bar']}; "
            f"noise {row['noise_time_bar']}; "
            f"memory {row['memory_gate']} |"
        )
    lines.extend(
        [
            "",
            "Fisher and Mann-Whitney p-values are in `summary.csv`.",
            (
                "B trajectories differ from A when "
                "`trajectory_differences` is nonzero."
            ),
            (
                "A calls-to-hit benefit is present when B's median is "
                "lower; its two-sided p-value is `mwu_b_a`."
            ),
            "",
            (
                "Early-stop per-call artifact flag: "
                f"{total_early} A/B seed pairs had unequal charged calls."
            ),
            (
                "Aggregate charged calls: "
                + ", ".join(
                    f"{arm}={total_charged[arm]}" for arm in ARMS
                )
                + "."
            ),
            (
                "Aggregate outside wall seconds: "
                + ", ".join(
                    f"{arm}={total_wall[arm]:.6f}" for arm in ARMS
                )
                + "."
            ),
        ]
    )
    args.report.write_text("\n".join(lines) + "\n")
    args.verdict.write_text(f"{verdict}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
