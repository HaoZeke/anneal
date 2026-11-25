#!/usr/bin/env python3
"""Summarize the independent anneal and recorded CREST samples."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

BUDGET = 332_675
MISS_CALLS = BUDGET + 1
CREST_MIN = 134_305
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_123
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return centre - half / denominator, centre + half / denominator


def bootstrap_median(
    values: list[int], rng: np.random.Generator
) -> tuple[float, float]:
    sample = np.asarray(values)
    indices = rng.integers(0, len(sample), size=(BOOTSTRAP_SAMPLES, len(sample)))
    medians = np.median(sample[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975], method="linear")
    return float(low), float(high)


def fmt_ci(interval: tuple[float, float], digits: int = 3) -> str:
    return f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"


def main() -> None:
    with (HERE / "results.csv").open(newline="") as handle:
        anneal = list(csv.DictReader(handle))
    if len(anneal) != 12:
        raise SystemExit(f"expected 12 anneal rows, found {len(anneal)}")
    with (ROOT / "bench/nh3-octamer/results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    crest = [
        row
        for row in history
        if row["arm"] == "crest" and 17_000 <= int(row["seed"]) <= 17_005
    ]
    if len(crest) != 6:
        raise SystemExit(f"expected 6 CREST rows, found {len(crest)}")

    anneal_hits = sum(row["hit"] == "yes" for row in anneal)
    crest_hits = sum(row["hit"] == "yes" for row in crest)
    anneal_calls = [int(row["calls_to_hit"]) for row in anneal]
    crest_calls = [
        int(row["calls"]) if row["hit"] == "yes" else MISS_CALLS for row in crest
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    anneal_boot = bootstrap_median(anneal_calls, rng)
    crest_boot = bootstrap_median(crest_calls, rng)
    anneal_wilson = wilson(anneal_hits, len(anneal))
    crest_wilson = wilson(crest_hits, len(crest))
    fisher = fisher_exact(
        [
            [anneal_hits, len(anneal) - anneal_hits],
            [crest_hits, len(crest) - crest_hits],
        ],
        alternative="two-sided",
    )
    mwu = mannwhitneyu(anneal_calls, crest_calls, alternative="two-sided")
    early_hits = sum(
        row["hit"] == "yes" and int(row["calls_to_hit"]) <= CREST_MIN
        for row in anneal
    )

    summary = {
        "budget": BUDGET,
        "miss_calls": MISS_CALLS,
        "anneal": {
            "hits": anneal_hits,
            "n": len(anneal),
            "wilson_95": anneal_wilson,
            "median_calls": float(np.median(anneal_calls)),
            "bootstrap_median_95": anneal_boot,
            "within_crest_min_hits": early_hits,
            "within_crest_min_fraction": early_hits / len(anneal),
        },
        "crest": {
            "hits": crest_hits,
            "n": len(crest),
            "wilson_95": crest_wilson,
            "median_calls": float(np.median(crest_calls)),
            "bootstrap_median_95": crest_boot,
            "seeds": [int(row["seed"]) for row in crest],
            "calls": crest_calls,
        },
        "tests": {
            "fisher_odds_ratio": float(fisher.statistic),
            "fisher_p": float(fisher.pvalue),
            "mann_whitney_u": float(mwu.statistic),
            "mann_whitney_p": float(mwu.pvalue),
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "quantile_method": "linear",
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = "\n".join(
        "| {seed} | {hit} | {calls_to_hit} | {stop_reason} | "
        "{best_energy_Eh} | {wall_s} |".format(**row)
        for row in anneal
    )
    report = f"""# NH3 octamer matched-budget result

The anneal arm used seeds 2000--2011. The independent CREST comparison uses
seeds 17000--17005 from `bench/nh3-octamer/results.csv`; CREST was not rerun.
The common anneal budget, {BUDGET} charged calls, is the maximum of those six
recorded CREST call counts.

| arm | hits / n | Wilson 95% CI | median calls | bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| anneal | {anneal_hits} / {len(anneal)} | {fmt_ci(anneal_wilson)} | {np.median(anneal_calls):.1f} | {fmt_ci(anneal_boot, 1)} |
| CREST | {crest_hits} / {len(crest)} | {fmt_ci(crest_wilson)} | {np.median(crest_calls):.1f} | {fmt_ci(crest_boot, 1)} |

The two-sided Fisher exact test on hits gives an odds ratio of
{fisher.statistic:.6g} and p = {fisher.pvalue:.6g}. The two-sided
Mann--Whitney test on calls gives U = {mwu.statistic:.6g} and
p = {mwu.pvalue:.6g}. Misses are represented as {MISS_CALLS} calls.

{early_hits} / {len(anneal)} anneal seeds ({early_hits / len(anneal):.1%}) hit
within 134305 calls, the minimum recorded per-seed CREST call count. Anneal's
median calls-to-hit, with misses represented as {MISS_CALLS}, is
{np.median(anneal_calls):.1f}.

The CREST recorded call counts are {", ".join(str(value) for value in crest_calls)};
their median is {np.median(crest_calls):.1f}. These are full recorded CREST
search-call totals, while the anneal values stop at the first target hit.
Because the seed sets differ, all comparisons treat the arms as independent
samples.

The median intervals use {BOOTSTRAP_SAMPLES} resamples with random seed
{BOOTSTRAP_SEED}. The hit-rate intervals are Wilson score intervals.

## Anneal rows

| seed | hit | calls to hit | stop reason | best energy / Eh | wall / s |
| ---: | :---: | ---: | :--- | ---: | ---: |
{table}
"""
    (HERE / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
