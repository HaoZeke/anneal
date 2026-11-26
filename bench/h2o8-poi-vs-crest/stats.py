"""Unpaired comparison of stored (H2O)8 first-hit calls.

Anneal calls are `calls_to_hit` from the polish-on-improve record.
CREST calls are `first_hit_call` from the crest first-hit table.
The charged and full-run columns are copied into inputs.csv and are not
used as the call counts.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ANNEAL_TABLE = "bench/octamer-matched/h2o8/poi/results.csv"
CREST_TABLE = "bench/h2o8-crest-firsthit/rows.csv"
ANNEAL_COMMIT = "42e708289a718d3d916b46e5ae43f26cf0a677c1"
CREST_COMMIT = "7ad8eab67d61f1d5835a9ce7b67e7b2cb2b4ede9"
CREST_COMMIT_REQUESTED = "7ad8eab67d61f1d5835a9ce7b67e7b2cb2b4ede9"
ANNEAL_SEEDS = tuple(range(2000, 2012))
CREST_SEEDS = tuple(range(17000, 17012))
BOOT_DRAWS = 10000
BOOT_SEED = 20261123

# Contents of bench/h2o8-crest-firsthit/rows.csv at CREST_COMMIT.
CREST_CSV = """\
seed,first_hit_call,total_calls,hit,best_Eh,wall_s,cpu_s,version,revision
17000,131547,151538,1,-40.681335841728,1067.110000,1062.260000,3.0.2,6914a25
17001,113009,151738,1,-40.681335622591,1083.270000,1079.310000,3.0.2,6914a25
17002,66888,142740,1,-40.681335976117,978.890000,975.560000,3.0.2,6914a25
17003,76037,151738,1,-40.681335914197,1958.330000,1090.280000,3.0.2,6914a25
17004,81130,151738,1,-40.681335975453,3842.000000,1122.400000,3.0.2,6914a25
17005,151739,151738,0,-40.681245390898,2360.070000,1109.340000,3.0.2,6914a25
17006,91817,139177,1,-40.681335988082,1969.610000,1059.850000,3.0.2,6914a25
17007,151739,151738,0,-40.681245466346,1335.660000,1087.490000,3.0.2,6914a25
17008,151739,151738,0,-40.681245423782,1090.130000,1086.020000,3.0.2,6914a25
17009,151739,151738,0,-40.603591290126,731.630000,726.980000,3.0.2,6914a25
17010,151739,151738,0,-40.681245330059,1181.730000,1177.940000,3.0.2,6914a25
17011,151696,151738,1,-40.681335265487,1144.180000,1140.600000,3.0.2,6914a25
"""

INPUT_FIELDS = [
    "source",
    "source_commit",
    "seed",
    "hit",
    "calls_to_hit",
    "first_hit_call",
    "charged_calls",
    "polish_calls",
    "polish_on_improve_triggers",
    "best_Eh",
    "cpu_s",
    "wall_s",
    "engine_sha",
    "harness_sha",
    "config_sha256",
    "poi_triggers",
    "charged",
    "total_calls",
    "version",
    "revision",
]


def _blank_row() -> dict[str, str]:
    return {field: "" for field in INPUT_FIELDS}


def anneal_input_rows() -> list[dict[str, str]]:
    path = REPO / ANNEAL_TABLE
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            seed = int(raw["seed"])
            if seed not in ANNEAL_SEEDS:
                continue
            row = _blank_row()
            row["source"] = "anneal"
            row["source_commit"] = ANNEAL_COMMIT
            for field in INPUT_FIELDS:
                if field in raw:
                    row[field] = raw[field]
            rows.append(row)
    rows.sort(key=lambda item: int(item["seed"]))
    found = [int(item["seed"]) for item in rows]
    if found != list(ANNEAL_SEEDS):
        raise SystemExit(f"anneal seeds {found}")
    return rows


def crest_input_rows() -> list[dict[str, str]]:
    rows = []
    for raw in csv.DictReader(io.StringIO(CREST_CSV)):
        seed = int(raw["seed"])
        if seed not in CREST_SEEDS:
            continue
        row = _blank_row()
        row["source"] = "crest"
        row["source_commit"] = CREST_COMMIT
        for field in INPUT_FIELDS:
            if field in raw:
                row[field] = raw[field]
        rows.append(row)
    rows.sort(key=lambda item: int(item["seed"]))
    found = [int(item["seed"]) for item in rows]
    if found != list(CREST_SEEDS):
        raise SystemExit(f"crest seeds {found}")
    return rows


def write_inputs(path: Path) -> None:
    rows = anneal_input_rows() + crest_input_rows()
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_inputs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def call_value(row: dict[str, str]) -> int:
    if row["source"] == "anneal":
        return int(row["calls_to_hit"])
    if row["source"] == "crest":
        return int(row["first_hit_call"])
    raise ValueError(row["source"])


def wilson95(hits: int, n: int) -> list[float]:
    interval = stats.binomtest(hits, n).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(interval.low), float(interval.high)]


def arm_summary(rows: list[dict[str, str]], source: str) -> dict:
    selected = [row for row in rows if row["source"] == source]
    calls = [call_value(row) for row in selected]
    hits = sum(int(row["hit"]) for row in selected)
    n = len(selected)
    seeds = [int(row["seed"]) for row in selected]
    summary = {
        "call_column": "calls_to_hit" if source == "anneal" else "first_hit_call",
        "calls": calls,
        "hit_rate": hits / n,
        "hits": hits,
        "median": float(np.median(np.asarray(calls, dtype=float))),
        "misses": n - hits,
        "n": n,
        "seeds": seeds,
        "source_commit": selected[0]["source_commit"],
        "table": ANNEAL_TABLE if source == "anneal" else CREST_TABLE,
        "wilson95": wilson95(hits, n),
    }
    if source == "crest":
        present = set(seeds)
        summary["commit_requested"] = CREST_COMMIT_REQUESTED
        summary["seeds_absent"] = [seed for seed in CREST_SEEDS if seed not in present]
    return summary


def median_ratio_ci(anneal_calls: list[int], crest_calls: list[int]) -> dict:
    anneal = np.asarray(anneal_calls, dtype=float)
    crest = np.asarray(crest_calls, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    ratios = np.empty(BOOT_DRAWS, dtype=float)
    for index in range(BOOT_DRAWS):
        anneal_draw = rng.choice(anneal, size=anneal.size, replace=True)
        crest_draw = rng.choice(crest, size=crest.size, replace=True)
        ratios[index] = float(np.median(crest_draw) / np.median(anneal_draw))
    low, high = np.percentile(ratios, [2.5, 97.5], method="linear")
    return {
        "ci95": [float(low), float(high)],
        "percentile": "numpy.percentile linear",
        "ratio": float(np.median(crest) / np.median(anneal)),
        "resamples": BOOT_DRAWS,
        "rng": "numpy.random.default_rng",
        "seed": BOOT_SEED,
    }


def summarize(rows: list[dict[str, str]]) -> dict:
    anneal = arm_summary(rows, "anneal")
    crest = arm_summary(rows, "crest")
    table = [
        [anneal["hits"], anneal["misses"]],
        [crest["hits"], crest["misses"]],
    ]
    odds_ratio, fisher_p = stats.fisher_exact(table, alternative="two-sided")
    odds_ratio = float(odds_ratio)
    mann = stats.mannwhitneyu(
        np.asarray(anneal["calls"], dtype=float),
        np.asarray(crest["calls"], dtype=float),
        alternative="two-sided",
        method="exact",
    )
    return {
        "anneal": anneal,
        "crest": crest,
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
        "median_ratio_crest_over_anneal": median_ratio_ci(anneal["calls"], crest["calls"]),
    }


def _num(value: float) -> str:
    return f"{value:.12g}"


def write_report(path: Path, result: dict) -> None:
    anneal = result["anneal"]
    crest = result["crest"]
    fisher = result["fisher_exact"]
    mann = result["mann_whitney"]
    ratio = result["median_ratio_crest_over_anneal"]
    odds = "infinite" if fisher["odds_ratio_infinite"] else _num(fisher["odds_ratio"])
    crest_lines = [
        f"CREST rows are `{crest['table']}` at `{crest['source_commit']}`.",
        f"The table contains seeds {crest['seeds'][0]}-{crest['seeds'][-1]}, n={crest['n']}.",
    ]
    if crest["seeds_absent"]:
        absent = ", ".join(str(seed) for seed in crest["seeds_absent"])
        crest_lines.append(f"Seeds {absent} are not in the table.")
    crest_lines.append(
        f"The call column is `{crest['call_column']}`. Misses stay at the stored value 151739."
    )
    lines = [
        "# Polish-on-improve versus CREST, (H2O)8",
        "",
        f"Anneal rows are `{anneal['table']}` at `{anneal['source_commit']}`.",
        f"Seeds {anneal['seeds'][0]}-{anneal['seeds'][-1]}. The call column is `{anneal['call_column']}`.",
        f"Hits {anneal['hits']}/{anneal['n']}. Wilson 95% CI {_num(anneal['wilson95'][0])} to {_num(anneal['wilson95'][1])}.",
        f"Median {_num(anneal['median'])}.",
        "",
        *crest_lines,
        f"Hits {crest['hits']}/{crest['n']}. Wilson 95% CI {_num(crest['wilson95'][0])} to {_num(crest['wilson95'][1])}.",
        f"Median {_num(crest['median'])}.",
        "",
        "The hit counts use a two-sided Fisher exact test.",
        f"Table of hits and misses, anneal then CREST: {fisher['table_hits_misses']}.",
        f"Odds ratio {odds}. p = {_num(fisher['p_value'])}.",
        "",
        "The call counts use a two-sided exact Mann-Whitney U test, anneal against CREST.",
        f"U = {_num(mann['u_anneal'])}. p = {_num(mann['p_value'])}.",
        "",
        "The median ratio is the CREST median divided by the anneal median.",
        f"Ratio {_num(ratio['ratio'])}.",
        (
            f"Percentile bootstrap 95% CI {_num(ratio['ci95'][0])} to {_num(ratio['ci95'][1])}, "
            f"{ratio['resamples']} resamples, numpy default_rng seed {ratio['seed']}."
        ),
        "",
        "These figures are the values written by `bench/h2o8-poi-vs-crest/stats.py` from `inputs.csv`.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    inputs = HERE / "inputs.csv"
    write_inputs(inputs)
    result = summarize(read_inputs(inputs))
    (HERE / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(HERE / "REPORT.md", result)


if __name__ == "__main__":
    main()
