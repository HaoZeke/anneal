#!/usr/bin/env python3
"""Summarize the append-only CooperativeRun architecture rows."""

from __future__ import annotations

import csv
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "lj75/rows.csv"
REPORT = HERE / "lj75/REPORT.md"
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_123
EXPECTED_SEEDS = 12
SECOND_SEED_LOWER_BOUND_S = 1_826.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], statistic) -> tuple[float, float]:
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = [
        statistic(generator.choices(values, k=len(values))) for _ in range(RESAMPLES)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def interval(values: list[float], statistic=statistics.median) -> tuple[float, float, float]:
    low, high = bootstrap_ci(values, statistic)
    return statistic(values), low, high


def main() -> None:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no rows")

    hits = [float(row["hit"]) for row in rows]
    budget = int(rows[0]["call_budget"])
    calls = [
        float(row["calls_to_hit"]) if row["calls_to_hit"] else float(budget + 1)
        for row in rows
    ]
    walls = [float(row["wall_s"]) for row in rows]
    ledger = [float(row["ledger_named_sum"]) for row in rows]
    hit_value, hit_low, hit_high = interval(hits, statistics.mean)
    calls_value, calls_low, calls_high = interval(calls)
    wall_value, wall_low, wall_high = interval(walls)
    ledger_value, ledger_low, ledger_high = interval(ledger)

    mismatches = [
        row["seed"]
        for row in rows
        if not (
            row["ledger_named_sum"]
            == row["ensemble_total"]
            == row["coordinator_total"]
            == row["call_budget"]
        )
    ]
    mismatch_text = "none" if not mismatches else ", ".join(mismatches)
    completed = len(rows)
    noun = "seed" if completed == 1 else "seeds"
    projection = ""
    if completed < EXPECTED_SEEDS:
        projected_hours = (
            (walls[0] + SECOND_SEED_LOWER_BOUND_S) / 2 * EXPECTED_SEEDS / 3600
        )
        projection = f"""
Seed 2001 remained active after {SECOND_SEED_LOWER_BOUND_S:.0f} seconds.
Together with seed 2000, that gives a full-panel lower-bound projection
of {projected_hours:.2f} hours. The four-hour rule stopped the incomplete
seed; seeds 2001 through 2011 remain unmeasured.
"""
    report = f"""# CooperativeRun LJ75 report

The arm completed {completed} {noun} and hit the target on
{sum(int(value) for value in hits)} {noun}. Misses enter calls-to-hit as
{budget + 1}. Intervals are percentile bootstrap 95% confidence
intervals from {RESAMPLES} resamples with seed {BOOTSTRAP_SEED}.

| Measure | Estimate | 95% CI |
| --- | ---: | ---: |
| Hit rate | {hit_value:.3f} | [{hit_low:.3f}, {hit_high:.3f}] |
| Calls to hit, median | {calls_value:.0f} | [{calls_low:.0f}, {calls_high:.0f}] |
| Wall time, median seconds | {wall_value:.3f} | [{wall_low:.3f}, {wall_high:.3f}] |
| Ledger sum, median calls | {ledger_value:.0f} | [{ledger_low:.0f}, {ledger_high:.0f}] |

Named-term, worker-total, coordinator-total, and budget mismatches:
{mismatch_text}.
{projection}

Fisher's exact test and the Mann-Whitney U test are unavailable because
the requested data contain one arm and no comparison arm.
"""
    REPORT.write_text(report, newline="\n")


if __name__ == "__main__":
    main()
