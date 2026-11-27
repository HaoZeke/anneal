#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Analyze paired PR 176 follow-up rows without dropping outliers."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

from gate_io import durable_write_text, load_rows
from gate_stats import fisher_two_sided_p, mann_whitney_p, median_type7
from panel import CELL_BY_NAME, CELLS, SEEDS, SESSION

BOOTSTRAP_DRAWS = 100_000
T_CRIT_23 = 2.068657610
ARMS = ("A", "A2", "B")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(
    values: list[float], seed: int
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return (
        statistics.fmean(values),
        quantile(means, 0.025),
        quantile(means, 0.975),
    )


def paired_log_ci(values: list[float]) -> tuple[float, float, float]:
    logs = [math.log(value) for value in values]
    mean = statistics.fmean(logs)
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    half = T_CRIT_23 * standard_error
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def ratio_stats(
    numerator: list[float], denominator: list[float], seed: int
) -> dict[str, float]:
    ratios = [top / bottom for top, bottom in zip(numerator, denominator)]
    boot, boot_lo, boot_hi = bootstrap_mean_ci(ratios, seed)
    log, log_lo, log_hi = paired_log_ci(ratios)
    return {
        "mean": boot,
        "boot_lo": boot_lo,
        "boot_hi": boot_hi,
        "log": log,
        "log_lo": log_lo,
        "log_hi": log_hi,
        "upper": max(boot_hi, log_hi),
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def select(rows: list[dict[str, str]], cell: str, arm: str):
    return sorted(
        (
            row for row in rows
            if row.get("session_id") == SESSION
            and row.get("cell") == cell
            and row.get("arm") == arm
            and row.get("status") == "ok"
        ),
        key=lambda row: int(row["seed"]),
    )


def values(rows, field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def write_csv(path: Path, fields: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def outlier_rows(grouped) -> list[tuple[str, str, dict[str, str]]]:
    found = []
    for (cell, arm), rows in grouped.items():
        marked: set[int] = set()
        for field in ("wall_s", "outside_per_charged_s"):
            sample = values(rows, field)
            q1 = quantile(sample, 0.25)
            q3 = quantile(sample, 0.75)
            spread = q3 - q1
            low = q1 - 1.5 * spread
            high = q3 + 1.5 * spread
            marked.update(
                index for index, value in enumerate(sample)
                if value < low or value > high
            )
        found.extend((cell, arm, rows[index]) for index in sorted(marked))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = load_rows(Path(args.rows))
    grouped = {
        (spec.cell, arm): select(rows, spec.cell, arm)
        for spec in CELLS for arm in ARMS
    }
    arm_rows = []
    summary_rows = []
    report = ["# PR 176 Paired Follow-up Results", ""]
    complete_all = True
    pass_all = True

    for spec in CELLS:
        groups = {arm: grouped[(spec.cell, arm)] for arm in ARMS}
        complete = all(
            [int(row["seed"]) for row in groups[arm]] == list(SEEDS)
            for arm in ARMS
        )
        complete_all &= complete
        for arm in ARMS:
            sample = groups[arm]
            arm_rows.append({
                "cell": spec.cell,
                "arm": arm,
                "runs": len(sample),
                "hits": sum(int(row["hit"]) for row in sample),
                "wall_mean_s": fmt(statistics.fmean(values(sample, "wall_s"))),
                "wall_median_s": fmt(median_type7(values(sample, "wall_s"))),
                "outside_per_call_mean_s": fmt(statistics.fmean(
                    values(sample, "outside_per_charged_s")
                )),
                "outside_per_call_median_s": fmt(median_type7(
                    values(sample, "outside_per_charged_s")
                )),
                "charged_mean": fmt(statistics.fmean(
                    values(sample, "charged")
                )),
                "charged_median": fmt(median_type7(
                    values(sample, "charged")
                )),
                "rss_anon_mean_kb": fmt(statistics.fmean(
                    values(sample, "rss_anon_kb")
                )),
                "rss_anon_median_kb": fmt(median_type7(
                    values(sample, "rss_anon_kb")
                )),
            })
        if not complete:
            pass_all = False
            report.extend([
                f"## {spec.cell}", "",
                "Result: PARTIAL. Paired inference was not run.", "",
            ])
            continue

        by_seed = {
            arm: {int(row["seed"]): row for row in groups[arm]}
            for arm in ARMS
        }
        ordered = {
            arm: [by_seed[arm][seed] for seed in SEEDS] for arm in ARMS
        }
        wall = ratio_stats(
            values(ordered["B"], "wall_s"),
            values(ordered["A"], "wall_s"),
            176_100 + spec.budget,
        )
        outside = ratio_stats(
            values(ordered["B"], "outside_per_charged_s"),
            values(ordered["A"], "outside_per_charged_s"),
            176_200 + spec.budget,
        )
        rebuild = ratio_stats(
            values(ordered["A2"], "outside_per_charged_s"),
            values(ordered["A"], "outside_per_charged_s"),
            176_300 + spec.budget,
        )
        hits = {
            arm: sum(int(row["hit"]) for row in ordered[arm])
            for arm in ARMS
        }
        fisher = fisher_two_sided_p(hits["A"], 24, hits["B"], 24)
        calls = {
            arm: [
                float(row["charged"]) if int(row["hit"]) else spec.budget + 1.0
                for row in ordered[arm]
            ]
            for arm in ARMS
        }
        mw_p, mw_worse = mann_whitney_p(calls["B"], calls["A"])
        fixed_pass = hits["B"] >= hits["A"] and outside["upper"] <= 1.10
        adjusted_bar = max(1.10, rebuild["upper"])
        adjusted_pass = (
            hits["B"] >= hits["A"] and outside["upper"] <= adjusted_bar
        )
        pass_all &= fixed_pass
        summary_rows.append({
            "cell": spec.cell,
            "hits_a": hits["A"],
            "hits_a2": hits["A2"],
            "hits_b": hits["B"],
            "fisher_two_sided_p": fmt(fisher),
            "mw_calls_two_sided_p": fmt(mw_p),
            "mw_calls_b_worse_p": fmt(mw_worse),
            "calls_a_median": fmt(median_type7(calls["A"])),
            "calls_b_median": fmt(median_type7(calls["B"])),
            "wall_b_a_mean": fmt(wall["mean"]),
            "wall_boot_lo": fmt(wall["boot_lo"]),
            "wall_boot_hi": fmt(wall["boot_hi"]),
            "wall_log_ratio": fmt(wall["log"]),
            "wall_log_lo": fmt(wall["log_lo"]),
            "wall_log_hi": fmt(wall["log_hi"]),
            "outside_b_a_mean": fmt(outside["mean"]),
            "outside_boot_lo": fmt(outside["boot_lo"]),
            "outside_boot_hi": fmt(outside["boot_hi"]),
            "outside_log_ratio": fmt(outside["log"]),
            "outside_log_lo": fmt(outside["log_lo"]),
            "outside_log_hi": fmt(outside["log_hi"]),
            "outside_a2_a_upper": fmt(rebuild["upper"]),
            "adjusted_bar": fmt(adjusted_bar),
            "fixed_verdict": "PASS" if fixed_pass else "FAIL",
            "adjusted_verdict": "PASS" if adjusted_pass else "FAIL",
        })
        report.extend([
            f"## {spec.cell}", "",
            f"Hits A/A2/B: {hits['A']}/{hits['A2']}/{hits['B']}; "
            f"Fisher p={fmt(fisher)}.", "",
            f"Wall B/A mean ratio {fmt(wall['mean'])}, bootstrap 95% CI "
            f"[{fmt(wall['boot_lo'])}, {fmt(wall['boot_hi'])}]; paired "
            f"log ratio {fmt(wall['log'])}, 95% CI "
            f"[{fmt(wall['log_lo'])}, {fmt(wall['log_hi'])}].", "",
            f"Outside per call B/A mean ratio {fmt(outside['mean'])}, "
            f"bootstrap 95% CI [{fmt(outside['boot_lo'])}, "
            f"{fmt(outside['boot_hi'])}]; paired log ratio "
            f"{fmt(outside['log'])}, 95% CI "
            f"[{fmt(outside['log_lo'])}, {fmt(outside['log_hi'])}].", "",
            f"A2/A outside upper bound {fmt(rebuild['upper'])}; fixed bar "
            f"1.10; adjusted bar {fmt(adjusted_bar)}.", "",
            f"Charged calls A/B medians {fmt(median_type7(calls['A']))}/"
            f"{fmt(median_type7(calls['B']))}; Mann-Whitney two-sided "
            f"p={fmt(mw_p)}.", "",
            f"Fixed verdict: {'PASS' if fixed_pass else 'FAIL'}. "
            f"Adjusted comparison: {'PASS' if adjusted_pass else 'FAIL'}.", "",
        ])

    outliers = outlier_rows(grouped)
    report.extend(["## Outlier runs", ""])
    if not outliers:
        report.extend(["No Tukey outliers in wall or outside time per call.", ""])
    else:
        report.append(
            "Cell | arm | seed | wall s | outside/call s | calls | hit | "
            "RssAnon KiB"
        )
        report.append("--- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
        for cell, arm, row in outliers:
            report.append(
                f"{cell} | {arm} | {row['seed']} | {row['wall_s']} | "
                f"{row['outside_per_charged_s']} | {row['charged']} | "
                f"{row['hit']} | {row['rss_anon_kb']}"
            )
        report.append("")

    report.extend([
        "## Overall", "",
        "Result: " + (
            "PASS" if complete_all and pass_all
            else "PARTIAL" if not complete_all
            else "FAIL"
        ) + ".", "",
    ])
    arm_fields = list(arm_rows[0])
    summary_fields = list(summary_rows[0]) if summary_rows else ["cell"]
    write_csv(Path(args.arms), arm_fields, arm_rows)
    write_csv(Path(args.summary), summary_fields, summary_rows)
    durable_write_text(Path(args.report), "\n".join(report))
    return 0 if complete_all and pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
