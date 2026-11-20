#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summary of the pre-registered GOFEE rerun.

Reads rows.csv and writes summary.csv. A miss is ranked at positive
infinity in the Mann-Whitney test. Fisher's exact test compares hit
counts with anneal. Both functions are the ones in
scripts/gmin_panel/run_panel.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gmin_panel"))
from run_panel import fisher, mann_whitney, type7  # noqa: E402

METHODS = ("anneal", "ase_mh", "agox_gofee")
FIELDS = [
    "system",
    "method",
    "n",
    "budget",
    "reference",
    "hits",
    "wall_stops",
    "best_median",
    "wall_median_s",
    "evals_to_hit_median",
    "fisher_p",
    "mannwhitney_u",
    "mannwhitney_p",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def calls_for_test(row: dict[str, str]) -> float:
    if row["hit"] == "1" and row["evals_to_hit"] != "":
        return float(row["evals_to_hit"])
    return math.inf


def median(values: list[float]) -> str:
    if not values:
        return ""
    return f"{type7(values, 0.5):.6f}"


def summarise(rows: list[dict[str, str]], systems: list[str], references: dict[str, float]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["system"], row["method"]), []).append(row)
    out: list[dict[str, str]] = []
    for system in systems:
        anneal = grouped.get((system, "anneal"), [])
        anneal_calls = [calls_for_test(row) for row in anneal]
        anneal_hits = sum(row["hit"] == "1" for row in anneal)
        for method in METHODS:
            group = grouped.get((system, method), [])
            if len(group) != 12:
                raise SystemExit(f"{system} {method} has {len(group)} rows, expected 12")
            hits = sum(row["hit"] == "1" for row in group)
            walls = [float(row["wall_s"]) for row in group]
            bests = [float(row["best"]) for row in group]
            hit_calls = [float(row["evals_to_hit"]) for row in group if row["hit"] == "1" and row["evals_to_hit"] != ""]
            record = {
                "system": system,
                "method": method,
                "n": "12",
                "budget": group[0]["budget"],
                "reference": f"{references[system]:.10f}",
                "hits": str(hits),
                "wall_stops": str(sum(row["stop"] == "wall" for row in group)),
                "best_median": median(bests),
                "wall_median_s": median(walls),
                "evals_to_hit_median": median(hit_calls),
                "fisher_p": "",
                "mannwhitney_u": "",
                "mannwhitney_p": "",
            }
            if method != "anneal":
                if len(anneal) != 12:
                    raise SystemExit(f"{system} anneal has {len(anneal)} rows, expected 12")
                record["fisher_p"] = f"{fisher(hits, 12, anneal_hits, 12):.6e}"
                u_stat, p_value = mann_whitney([calls_for_test(row) for row in group], anneal_calls)
                record["mannwhitney_u"] = f"{u_stat:.4f}"
                record["mannwhitney_p"] = f"{p_value:.6e}"
            out.append(record)
    return out


def write_summary(rows: list[dict[str, str]], path: Path, systems: list[str], references: dict[str, float]) -> None:
    records = summarise(rows, systems, references)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    from run_campaign import REFERENCES, SYSTEMS

    rows = load_rows(args.rows)
    expected = len(SYSTEMS) * 12 * 3
    if len(rows) != expected:
        raise SystemExit(f"{args.rows} has {len(rows)} rows, expected {expected}")
    write_summary(rows, args.out, [item["name"] for item in SYSTEMS], REFERENCES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
