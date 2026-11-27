#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Combine the shard rows and compare anneal with CREST."""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TARGET_EH = -40.681336013361
TOL_EH = 1e-6
BUDGET = 151738
BOOT_DRAWS = 10000
BOOT_SEED = 20261123
CREST_TABLE = "bench/h2o8-poi-vs-crest/inputs.csv"
CREST_SEEDS = tuple(range(17000, 17012))
HEX64 = re.compile(r"[0-9a-fA-F]{64}")

COMBINED_FIELDS = [
    "shard",
    "seed",
    "budget",
    "calls_charged",
    "wall_s",
    "cpu_s",
    "best_Eh",
    "best_eV",
    "hit",
    "calls_to_first_hit",
    "revision",
    "host",
    "core",
    "cpu_model",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def is_hit(best_eh: str) -> bool:
    return abs(float(best_eh) - TARGET_EH) <= TOL_EH


def call_score(row: dict[str, str]) -> int:
    budget = int(row["budget"])
    if not is_hit(row["best_Eh"]):
        return budget + 1
    return int(row["calls_to_first_hit"])


def check_host(value: str) -> None:
    if HEX64.fullmatch(value):
        raise SystemExit("host field is 64 hex")


def combine(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> list[dict[str, str]]:
    combined = []
    for shard, rows, host_key in (
        ("A", rows_a, "host_id_hash"),
        ("B", rows_b, "host_id"),
    ):
        for raw in rows:
            check_host(raw[host_key])
            combined.append(
                {
                    "shard": shard,
                    "seed": raw["seed"],
                    "budget": raw["budget"],
                    "calls_charged": raw["calls_charged"],
                    "wall_s": raw["wall_s"],
                    "cpu_s": raw["cpu_s"],
                    "best_Eh": raw["best_Eh"],
                    "best_eV": raw.get("best_eV", ""),
                    "hit": raw["hit"],
                    "calls_to_first_hit": raw["calls_to_first_hit"],
                    "revision": raw["revision"],
                    "host": raw[host_key],
                    "core": raw["core"],
                    "cpu_model": raw["cpu_model"],
                }
            )
    seeds = [int(row["seed"]) for row in combined]
    if seeds != list(range(2000, 2012)):
        raise SystemExit(f"seeds {seeds}")
    return combined


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMBINED_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def arm_record(rows: list[dict[str, str]], shard: str) -> dict:
    selected = [row for row in rows if row["shard"] == shard]
    calls = [call_score(row) for row in selected]
    hits = sum(is_hit(row["best_Eh"]) for row in selected)
    n = len(selected)
    return {
        "arm": shard,
        "hits": hits,
        "median": float(np.median(np.asarray(calls, dtype=float))),
        "misses": n - hits,
        "n": n,
        "budget": BUDGET,
        "miss_score": BUDGET + 1,
        "seeds": [int(row["seed"]) for row in selected],
        "calls": calls,
        "target_Eh": TARGET_EH,
        "tolerance_Eh": TOL_EH,
    }


def load_crest() -> list[dict[str, str]]:
    rows = [row for row in read_csv(REPO / CREST_TABLE) if row["source"] == "crest"]
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(CREST_SEEDS):
        raise SystemExit(f"crest seeds {seeds}")
    return rows


def crest_call(row: dict[str, str]) -> int:
    if not is_hit(row["best_Eh"]):
        return BUDGET + 1
    return int(row["first_hit_call"])


def method_arm(name: str, rows: list[dict[str, str]], calls: list[int], source: str, call_column: str) -> dict:
    hits = sum(is_hit(row["best_Eh"]) for row in rows)
    n = len(rows)
    return {
        "arm": name,
        "hits": hits,
        "median": float(np.median(np.asarray(calls, dtype=float))),
        "misses": n - hits,
        "n": n,
        "budget": BUDGET,
        "miss_score": BUDGET + 1,
        "seeds": [int(row["seed"]) for row in rows],
        "calls": calls,
        "source": source,
        "call_column": call_column,
        "target_Eh": TARGET_EH,
        "tolerance_Eh": TOL_EH,
    }


def median_ratio_ci(anneal_calls: list[int], crest_calls: list[int]) -> dict:
    anneal = np.asarray(anneal_calls, dtype=float)
    crest = np.asarray(crest_calls, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    ratios = np.empty(BOOT_DRAWS, dtype=float)
    for index in range(BOOT_DRAWS):
        anneal_draw = rng.choice(anneal, size=anneal.size, replace=True)
        crest_draw = rng.choice(crest, size=crest.size, replace=True)
        ratios[index] = float(np.median(anneal_draw) / np.median(crest_draw))
    low, high = np.percentile(ratios, [2.5, 97.5], method="linear")
    return {
        "ratio": float(np.median(anneal) / np.median(crest)),
        "ci95": [float(low), float(high)],
        "percentile": "numpy.percentile linear",
        "resamples": BOOT_DRAWS,
        "rng": "numpy.random.default_rng",
        "seed": BOOT_SEED,
        "numerator": "anneal",
        "denominator": "crest",
    }


def compare_arms(left: dict, right: dict) -> dict:
    table = [
        [left["hits"], left["misses"]],
        [right["hits"], right["misses"]],
    ]
    odds_ratio, fisher_p = stats.fisher_exact(table, alternative="two-sided")
    odds_ratio = float(odds_ratio)
    mann = stats.mannwhitneyu(
        np.asarray(left["calls"], dtype=float),
        np.asarray(right["calls"], dtype=float),
        alternative="two-sided",
        method="exact",
    )
    return {
        "fisher_exact": {
            "alternative": "two-sided",
            "odds_ratio": None if not np.isfinite(odds_ratio) else odds_ratio,
            "odds_ratio_infinite": not bool(np.isfinite(odds_ratio)),
            "p_value": float(fisher_p),
            "table_hits_misses": table,
        },
        "mann_whitney": {
            "alternative": "two-sided",
            "method": "exact",
            "p_value": float(mann.pvalue),
            "u_anneal": float(mann.statistic),
        },
    }


def rows_b_vs_poi(rows_b: list[dict[str, str]]) -> dict:
    poi_path = REPO / "bench/octamer-matched/h2o8/poi/results.csv"
    poi = {int(row["seed"]): row for row in read_csv(poi_path)}
    pairs = []
    for raw in rows_b:
        seed = int(raw["seed"])
        stored = poi[seed]
        pairs.append(
            {
                "seed": seed,
                "rows_b_calls_to_first_hit": int(raw["calls_to_first_hit"]),
                "poi_calls_to_hit": int(stored["calls_to_hit"]),
                "rows_b_best_Eh": raw["best_Eh"],
                "poi_best_Eh": stored["best_Eh"],
            }
        )
    return {
        "poi_table": "bench/octamer-matched/h2o8/poi/results.csv",
        "poi_column": "calls_to_hit",
        "rows_b_column": "calls_to_first_hit",
        "seeds": pairs,
        "index_pairs_equal": all(
            item["rows_b_calls_to_first_hit"] == item["poi_calls_to_hit"] for item in pairs
        ),
        "energy_strings_equal": all(
            item["rows_b_best_Eh"] == item["poi_best_Eh"] for item in pairs
        ),
    }


def decimal_median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / Decimal(2)


def wall_summary(rows: list[dict[str, str]]) -> dict:
    summary = {}
    for arm in ("anneal", "crest"):
        selected = [row for row in rows if row["arm"] == arm]
        walls = [Decimal(row["wall_s"]) for row in selected]
        cpus = [Decimal(row["cpu_s"]) for row in selected]
        summary[arm] = {
            "n": len(selected),
            "wall_s_sum": format(sum(walls), "f"),
            "wall_s_median": format(decimal_median(walls), "f"),
            "cpu_s_sum": format(sum(cpus), "f"),
            "cpu_s_median": format(decimal_median(cpus), "f"),
            "host_id": sorted({row["host_id"] for row in selected}),
            "core": sorted({row["core"] for row in selected}),
        }
    return summary


def summarize() -> dict:
    rows_a = read_csv(HERE / "rows-A.csv")
    rows_b = read_csv(HERE / "rows-B.csv")
    combined = combine(rows_a, rows_b)
    write_csv(HERE / "rows.csv", combined)
    anneal_calls = [call_score(row) for row in combined]
    crest_rows = load_crest()
    crest_calls = [crest_call(row) for row in crest_rows]
    anneal = method_arm("anneal", combined, anneal_calls, "bench/pareto-h2o8/rows.csv", "calls_to_first_hit")
    crest = method_arm("crest", crest_rows, crest_calls, CREST_TABLE, "first_hit_call")
    compared = compare_arms(anneal, crest)
    abba = read_csv(HERE / "rows-abba.csv")
    return {
        "arms": [anneal, crest],
        "paired_by_seed": False,
        "fisher_exact": compared["fisher_exact"],
        "mann_whitney": compared["mann_whitney"],
        "median_ratio_anneal_over_crest": median_ratio_ci(anneal_calls, crest_calls),
        "seed_stability": {
            "label": "seed-stability check",
            "same_method": "anneal",
            "A": arm_record(combined, "A"),
            "B": arm_record(combined, "B"),
        },
        "abba_wall": wall_summary(abba),
        "rows_b_vs_poi": rows_b_vs_poi(rows_b),
    }


def main() -> None:
    result = summarize()
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
