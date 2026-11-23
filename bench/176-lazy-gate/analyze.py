#!/usr/bin/env python3
"""Score the lazy-basin gate. A FAIL is never relabeled."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

from gate_io import durable_write_text, load_rows
from gate_stats import fisher_two_sided_p, mann_whitney_p, median_type7, student_t_crit
from panel import (
    BOOTSTRAP_DRAWS,
    CELLS,
    IDENTITY_ARM,
    OUTSIDE_BAR,
    PANEL_ARMS,
    SESSION,
)

IDENTITY_FIELDS = (
    "hit",
    "charged",
    "best_bits",
    "struct_sha256",
    "final_energy_bits",
    "final_struct_sha256",
)


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
    values: list[float], seed: int, draws: int
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    ]
    return statistics.fmean(values), quantile(means, 0.025), quantile(means, 0.975)


def paired_log_ci(values: list[float]) -> tuple[float, float, float]:
    logs = [math.log(value) for value in values]
    mean = statistics.fmean(logs)
    point = math.exp(mean)
    if len(logs) < 2:
        return point, point, point
    spread = statistics.stdev(logs)
    if spread == 0.0:
        return point, point, point
    half = student_t_crit(len(logs) - 1) * spread / math.sqrt(len(logs))
    return point, math.exp(mean - half), math.exp(mean + half)


def ratio_stats(
    numerator: list[float], denominator: list[float], seed: int, draws: int
) -> dict[str, float]:
    ratios = [top / bottom for top, bottom in zip(numerator, denominator)]
    if any(value <= 0.0 for value in ratios):
        raise ValueError("non-positive ratio")
    mean, boot_lo, boot_hi = bootstrap_mean_ci(ratios, seed, draws)
    log_mean, log_lo, log_hi = paired_log_ci(ratios)
    return {
        "mean": mean,
        "boot_lo": boot_lo,
        "boot_hi": boot_hi,
        "log": log_mean,
        "log_lo": log_lo,
        "log_hi": log_hi,
        "upper": max(boot_hi, log_hi),
        "lower": min(boot_lo, log_lo),
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ok_rows(rows: list[dict[str, str]], cell: str, arm: str) -> list[dict[str, str]]:
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


def scored_calls(sample: list[dict[str, str]], budget: int) -> list[float]:
    scored = []
    for row in sample:
        if int(row["hit"]) == 1:
            scored.append(float(row["charged"]))
        else:
            scored.append(float(budget + 1))
    return scored


def values(sample: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in sample]


def identity_table(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    mismatches: list[dict[str, str]] = []
    summary: list[dict[str, str]] = []
    for spec in CELLS:
        left = {int(row["seed"]): row for row in ok_rows(rows, spec.cell, "B")}
        right = {int(row["seed"]): row for row in ok_rows(rows, spec.cell, IDENTITY_ARM)}
        differed = 0
        missing = 0
        for seed in spec.seeds:
            b_row = left.get(seed)
            b0_row = right.get(seed)
            if b_row is None or b0_row is None:
                missing += 1
                mismatches.append({
                    "cell": spec.cell,
                    "seed": str(seed),
                    "field": "missing",
                    "b": "" if b_row is None else "ok",
                    "b0": "" if b0_row is None else "ok",
                })
                continue
            for field in IDENTITY_FIELDS:
                if b_row.get(field, "") != b0_row.get(field, ""):
                    differed += 1
                    mismatches.append({
                        "cell": spec.cell,
                        "seed": str(seed),
                        "field": field,
                        "b": b_row.get(field, ""),
                        "b0": b0_row.get(field, ""),
                    })
        if differed:
            verdict = "FAIL"
        elif missing or len(left) != spec.n_seeds or len(right) != spec.n_seeds:
            verdict = "INCOMPLETE"
        else:
            verdict = "PASS"
        summary.append({
            "cell": spec.cell,
            "n_b": str(len(left)),
            "n_b0": str(len(right)),
            "expected": str(spec.n_seeds),
            "mismatch_fields": str(differed),
            "missing_seeds": str(missing),
            "verdict": verdict,
        })
    return mismatches, summary


def summarize_cell(
    spec,
    groups: dict[str, list[dict[str, str]]],
    cell_index: int,
    draws: int,
) -> dict[str, str]:
    n_expected = spec.n_seeds
    counts = {arm: len(groups[arm]) for arm in PANEL_ARMS}
    complete = all(
        [int(row["seed"]) for row in groups[arm]] == list(spec.seeds)
        for arm in PANEL_ARMS
    )
    hits = {arm: sum(int(row["hit"]) for row in groups[arm]) for arm in PANEL_ARMS}
    walks = {
        arm: sum(int(row["basis_walks"]) > 0 for row in groups[arm])
        for arm in PANEL_ARMS
    }
    raw_median = {
        arm: median_type7(values(groups[arm], "charged")) if groups[arm] else math.nan
        for arm in PANEL_ARMS
    }
    base = {
        "cell": spec.cell,
        "n_expected": str(n_expected),
        "n_a": str(counts["A"]),
        "n_a2": str(counts["A2"]),
        "n_b": str(counts["B"]),
        "hits_a": str(hits["A"]),
        "hits_a2": str(hits["A2"]),
        "hits_b": str(hits["B"]),
        "walks_a": str(walks["A"]),
        "walks_a2": str(walks["A2"]),
        "walks_b": str(walks["B"]),
        "verdict": "PARTIAL",
    }
    if not complete:
        return base
    by_seed = {
        arm: {int(row["seed"]): row for row in groups[arm]} for arm in PANEL_ARMS
    }
    ordered = {arm: [by_seed[arm][seed] for seed in spec.seeds] for arm in PANEL_ARMS}
    outside = ratio_stats(
        values(ordered["B"], "outside_per_charged_s"),
        values(ordered["A"], "outside_per_charged_s"),
        176_000 + cell_index * 17 + 2,
        draws,
    )
    outside_a2 = ratio_stats(
        values(ordered["A2"], "outside_per_charged_s"),
        values(ordered["A"], "outside_per_charged_s"),
        176_000 + cell_index * 17 + 3,
        draws,
    )
    wall = ratio_stats(
        values(ordered["B"], "wall_s"),
        values(ordered["A"], "wall_s"),
        176_000 + cell_index * 17 + 1,
        draws,
    )
    rss = ratio_stats(
        values(ordered["B"], "rss_anon_kb"),
        values(ordered["A"], "rss_anon_kb"),
        176_000 + cell_index * 17 + 4,
        draws,
    )
    rss_a2 = ratio_stats(
        values(ordered["A2"], "rss_anon_kb"),
        values(ordered["A"], "rss_anon_kb"),
        176_000 + cell_index * 17 + 5,
        draws,
    )
    calls = {arm: scored_calls(ordered[arm], spec.budget) for arm in ("A", "B")}
    mw_p, _mw_worse = mann_whitney_p(calls["B"], calls["A"])
    fisher = fisher_two_sided_p(hits["A"], n_expected, hits["B"], n_expected)
    adjusted = max(OUTSIDE_BAR, outside_a2["upper"])
    rss_limit = max(OUTSIDE_BAR, rss_a2["upper"])
    hit_fail = hits["B"] < hits["A"]
    outside_fail = outside["upper"] > OUTSIDE_BAR
    rss_fail = rss["lower"] > rss_limit
    verdict = "FAIL" if hit_fail or outside_fail or rss_fail else "PASS"
    base.update({
        "fisher_p": fmt(fisher),
        "mw_p": fmt(mw_p),
        "median_calls_a": fmt(median_type7(calls["A"])),
        "median_calls_b": fmt(median_type7(calls["B"])),
        "median_charged_a": fmt(raw_median["A"]),
        "median_charged_a2": fmt(raw_median["A2"]),
        "median_charged_b": fmt(raw_median["B"]),
        "outside_boot_lo": fmt(outside["boot_lo"]),
        "outside_boot_hi": fmt(outside["boot_hi"]),
        "outside_log_lo": fmt(outside["log_lo"]),
        "outside_log_hi": fmt(outside["log_hi"]),
        "outside_upper": fmt(outside["upper"]),
        "a2_outside_upper": fmt(outside_a2["upper"]),
        "outside_adjusted": fmt(adjusted),
        "wall_boot_lo": fmt(wall["boot_lo"]),
        "wall_boot_hi": fmt(wall["boot_hi"]),
        "wall_log_lo": fmt(wall["log_lo"]),
        "wall_log_hi": fmt(wall["log_hi"]),
        "rss_boot_lo": fmt(rss["boot_lo"]),
        "rss_boot_hi": fmt(rss["boot_hi"]),
        "rss_log_lo": fmt(rss["log_lo"]),
        "rss_log_hi": fmt(rss["log_hi"]),
        "rss_lower": fmt(rss["lower"]),
        "a2_rss_upper": fmt(rss_a2["upper"]),
        "rss_limit": fmt(rss_limit),
        "hit_bar": "FAIL" if hit_fail else "PASS",
        "outside_bar": "FAIL" if outside_fail else "PASS",
        "rss_bar": "FAIL" if rss_fail else "PASS",
        "verdict": verdict,
    })
    return base


def render(summary_rows: list[dict[str, str]], identity_summary: list[dict[str, str]]) -> str:
    lines = ["# Lazy basin gate", ""]
    lines.append(
        "| cell | n A/A2/B | hits A/A2/B | Fisher | MW | "
        "median calls A/B | median charged A/B | "
        "outside B/A boot | outside log | outside upper | A2/A upper | "
        "wall B/A boot | rss B/A boot | rss lower | rss limit | "
        "walks>0 A/A2/B | verdict |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for row in summary_rows:
        if row["verdict"] == "PARTIAL":
            lines.append(
                f"| {row['cell']} | {row['n_a']}/{row['n_a2']}/{row['n_b']} | "
                f"{row['hits_a']}/{row['hits_a2']}/{row['hits_b']} | | | | | | | | | | | | | "
                f"{row['walks_a']}/{row['walks_a2']}/{row['walks_b']} | PARTIAL |"
            )
            continue
        lines.append(
            f"| {row['cell']} | {row['n_a']}/{row['n_a2']}/{row['n_b']} | "
            f"{row['hits_a']}/{row['hits_a2']}/{row['hits_b']} | {row['fisher_p']} | "
            f"{row['mw_p']} | {row['median_calls_a']}/{row['median_calls_b']} | "
            f"{row['median_charged_a']}/{row['median_charged_b']} | "
            f"[{row['outside_boot_lo']}, {row['outside_boot_hi']}] | "
            f"[{row['outside_log_lo']}, {row['outside_log_hi']}] | "
            f"{row['outside_upper']} | {row['a2_outside_upper']} | "
            f"[{row['wall_boot_lo']}, {row['wall_boot_hi']}] | "
            f"[{row['rss_boot_lo']}, {row['rss_boot_hi']}] | "
            f"{row['rss_lower']} | {row['rss_limit']} | "
            f"{row['walks_a']}/{row['walks_a2']}/{row['walks_b']} | {row['verdict']} |"
        )
    lines.extend(["", "## Bit identity", ""])
    lines.append("| cell | n B/B0 | mismatches | missing | verdict |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in identity_summary:
        lines.append(
            f"| {row['cell']} | {row['n_b']}/{row['n_b0']} | "
            f"{row['mismatch_fields']} | {row['missing_seeds']} | {row['verdict']} |"
        )
    lines.append("")
    if any(row["verdict"] == "FAIL" for row in identity_summary):
        claim = "FAIL"
    elif any(row["verdict"] != "PASS" for row in identity_summary):
        claim = "INCOMPLETE"
    elif identity_summary:
        claim = "PASS"
    else:
        claim = "INCOMPLETE"
    lines.append(f"Bit-identity claim: {claim}.")
    lines.append("")
    finished = [row for row in summary_rows if row["verdict"] != "PARTIAL"]
    if len(finished) != len(CELLS):
        lines.append("Panel: PARTIAL.")
    elif any(row["verdict"] == "FAIL" for row in finished):
        lines.append("Panel: FAIL.")
    else:
        lines.append("Panel: PASS.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    rows = load_rows(Path(args.rows))
    summary_rows = []
    for index, spec in enumerate(CELLS):
        groups = {arm: ok_rows(rows, spec.cell, arm) for arm in PANEL_ARMS}
        summary_rows.append(summarize_cell(spec, groups, index, args.draws))
    mismatches, identity_summary = identity_table(rows)
    fields: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(Path(args.summary), fields, [
        {key: row.get(key, "") for key in fields} for row in summary_rows
    ])
    write_csv(
        Path(args.identity),
        ["cell", "seed", "field", "b", "b0"],
        mismatches,
    )
    id_path = Path(args.identity).with_name("identity-summary.csv")
    write_csv(id_path, list(identity_summary[0]) if identity_summary else ["cell"], identity_summary)
    durable_write_text(Path(args.report), render(summary_rows, identity_summary))
    failed = any(row["verdict"] == "FAIL" for row in summary_rows)
    failed = failed or any(row["verdict"] == "FAIL" for row in identity_summary)
    incomplete = any(row["verdict"] == "PARTIAL" for row in summary_rows)
    incomplete = incomplete or any(row["verdict"] == "INCOMPLETE" for row in identity_summary)
    return 1 if failed or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
