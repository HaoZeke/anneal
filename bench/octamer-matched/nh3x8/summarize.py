#!/usr/bin/env python3
"""Summarize the NH3 octamer matched-budget anneal sample."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

BUDGET = 332_675
MISS_CALLS = BUDGET + 1
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_123
REFERENCE_EH = -35.470769324714
TOLERANCE_EH = 1.0e-6
AUTOEV = 27.21138505
TARGET_EV = (REFERENCE_EH + TOLERANCE_EH) * AUTOEV
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = hits / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return centre - half / denominator, centre + half / denominator


def bootstrap_median(values: list[int], rng: np.random.Generator) -> tuple[float, float]:
    sample = np.asarray(values)
    indices = rng.integers(0, len(sample), size=(BOOTSTRAP_SAMPLES, len(sample)))
    medians = np.median(sample[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975], method="linear")
    return float(low), float(high)


def field(text: str, name: str) -> str:
    prefix = name + ":"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"missing {name}")


def main() -> None:
    with (HERE / "results.csv").open(newline="") as handle:
        anneal = list(csv.DictReader(handle))
    seeds = [int(row["seed"]) for row in anneal]
    if seeds != list(range(2000, 2012)):
        raise SystemExit(f"expected seeds 2000-2011, found {seeds}")
    for row in anneal:
        for value in row.values():
            if value.startswith("/") or value.startswith("../"):
                raise SystemExit(f"row holds a non-relative path: {value}")

    with (ROOT / "bench/nh3-octamer/results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    crest = [
        row
        for row in history
        if row["arm"] == "crest" and 17_000 <= int(row["seed"]) <= 17_005
    ]
    crest.sort(key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in crest] != list(range(17000, 17006)):
        raise SystemExit("expected one CREST row for each of seeds 17000-17005")

    for row in anneal:
        calls = int(row["calls_to_hit"])
        if row["hit"] == "yes":
            if calls != BUDGET or row["stop_reason"] != "target":
                raise SystemExit(f"unexpected hit row {row['seed']}")
        elif row["hit"] == "no":
            if calls != MISS_CALLS or row["stop_reason"] != "budget":
                raise SystemExit(f"unexpected miss row {row['seed']}")
        else:
            raise SystemExit(f"unexpected hit flag {row['hit']}")

    for seed in (2000, 2001):
        if (HERE / "runs" / str(seed) / "run.txt").exists():
            raise SystemExit(f"seed {seed} unexpectedly has run.txt")

    splits = []
    for seed in range(2002, 2012):
        text = (HERE / "runs" / str(seed) / "run.txt").read_text()
        record = {
            "energy_calls": int(field(text, "energy_calls")),
            "search_calls": int(field(text, "search_calls")),
            "polish_calls": int(field(text, "polish_calls")),
            "polished_from_eV": float(field(text, "polished_from_eV")),
            "objective_best_eV": float(field(text, "objective_best_eV")),
        }
        if record["search_calls"] + record["polish_calls"] != record["energy_calls"]:
            raise SystemExit(f"seed {seed} call split does not add up")
        if record["polished_from_eV"] <= TARGET_EV:
            raise SystemExit(f"seed {seed} loose seat is already at the target")
        row = anneal[seed - 2000]
        final_hit = record["objective_best_eV"] <= TARGET_EV
        if final_hit != (row["hit"] == "yes"):
            raise SystemExit(f"seed {seed} csv hit disagrees with objective_best_eV")
        splits.append(record)

    search_calls = sorted({item["search_calls"] for item in splits})
    polish_calls = sorted({item["polish_calls"] for item in splits})
    energy_calls = sorted({item["energy_calls"] for item in splits})
    if search_calls != [331_075] or polish_calls != [1_600] or energy_calls != [BUDGET]:
        raise SystemExit(
            f"unexpected call split {search_calls} {polish_calls} {energy_calls}"
        )

    anneal_hits = sum(row["hit"] == "yes" for row in anneal)
    crest_hits = sum(row["hit"] == "yes" for row in crest)
    anneal_calls = [int(row["calls_to_hit"]) for row in anneal]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = bootstrap_median(anneal_calls, rng)
    wilson_ci = wilson(anneal_hits, len(anneal))
    table = [
        [anneal_hits, len(anneal) - anneal_hits],
        [crest_hits, len(crest) - crest_hits],
    ]
    fisher = fisher_exact(table, alternative="two-sided")
    median = float(np.median(np.asarray(anneal_calls)))
    odds_infinite = bool(math.isinf(fisher.statistic))
    summary = {
        "budget": BUDGET,
        "miss_calls": MISS_CALLS,
        "target_eV": TARGET_EV,
        "anneal": {
            "hits": anneal_hits,
            "n": len(anneal),
            "wilson_95": list(wilson_ci),
            "median_calls_to_hit": median,
            "bootstrap_median_95": list(boot),
        },
        "crest": {
            "hits": crest_hits,
            "n": len(crest),
            "seeds": [int(row["seed"]) for row in crest],
            "hit_definition": "hit column, reference within tolerance",
            "calls_are": "full-run totals",
        },
        "tests": {
            "table_rows": ["anneal hits, anneal misses", "CREST hits, CREST misses"],
            "table": table,
            "fisher_odds_ratio": None if odds_infinite else float(fisher.statistic),
            "fisher_odds_ratio_infinite": odds_infinite,
            "fisher_p": float(fisher.pvalue),
            "mann_whitney": "not_computed",
        },
        "call_accounting": {
            "search_calls": search_calls[0],
            "polish_calls": polish_calls[0],
            "energy_calls": energy_calls[0],
            "run_txt_seeds": "2002-2011",
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "quantile_method": "linear",
        },
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rendered = "\n".join(
        "| {seed} | {hit} | {calls_to_hit} | {stop_reason} | "
        "{best_energy_Eh} | {wall_s} |".format(**row)
        for row in anneal
    )
    odds = "inf" if odds_infinite else f"{fisher.statistic:.6g}"
    report = f"""# NH3 octamer matched-budget result

Twelve anneal seeds, 2000-2011, are in `results.csv`. CREST was not rerun.
The hit comparison uses seeds 17000-17005 from `bench/nh3-octamer/results.csv`.

Anneal hits {anneal_hits}/12. Wilson 95% CI [{wilson_ci[0]:.6f}, {wilson_ci[1]:.6f}].
Median calls_to_hit, with a miss stored as {MISS_CALLS}: {median:.1f}.
Bootstrap 95% CI of that median, {BOOTSTRAP_SAMPLES} resamples, seed {BOOTSTRAP_SEED}, linear quantiles: [{boot[0]:.1f}, {boot[1]:.1f}].

CREST hits {crest_hits}/6. The two-sided Fisher exact test on the hit table
[[{table[0][0]}, {table[0][1]}], [{table[1][0]}, {table[1][1]}]]
gives odds ratio {odds} and p = {fisher.pvalue:.6g}.

Every anneal hit has calls_to_hit {BUDGET} and stop_reason target.
That number is `energy_calls` from the run: the ledger total after the hop
and the final polish. Saved run.txt files for seeds 2002-2011 record
search_calls {search_calls[0]} and polish_calls {polish_calls[0]}.
Seeds 2000 and 2001 were recorded by the same harness and have no saved
run.txt in this tree. `README.md` names the lines that define the count.

The CREST calls column holds full-run totals. This summary does not compare
those totals with anneal calls_to_hit. A calls comparison needs CREST
first-hit counts, which are not recorded for NH3.

## Anneal rows

| seed | hit | calls to hit | stop reason | best energy / Eh | wall / s |
| ---: | :---: | ---: | :--- | ---: | ---: |
{rendered}
"""
    (HERE / "REPORT.md").write_text(report)
    print(f"HITS {anneal_hits}/12")
    print(f"WILSON {wilson_ci[0]:.10f} {wilson_ci[1]:.10f}")
    print(f"MEDIAN {median:.10f}")
    print(f"BOOT {boot[0]:.10f} {boot[1]:.10f}")
    print(f"FISHER_OR {fisher.statistic!r}")
    print(f"FISHER_P {fisher.pvalue!r}")
    print(f"TABLE {table}")
    print(f"SPLIT {search_calls[0]} {polish_calls[0]} {energy_calls[0]}")


if __name__ == "__main__":
    main()
