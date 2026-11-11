#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise a seed-matched Lennard-Jones panel.

Reads the row file written by ``h2h_gmin_proof`` and prints one row per
system and phase. The median and the interquartile range use Hyndman and
Fan type 7. A miss is already stored as the full budget. Best evaluations
to a hit is the minimum of that column. Best energy is printf ``%.17g``.
"""

from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from decimal import Decimal


def type7(sorted_values: list[float], p: float) -> float:
    """Hyndman-Fan type 7 quantile. ``sorted_values`` is ascending."""
    n = len(sorted_values)
    if n < 1:
        raise SystemExit("type 7 needs a sample")
    if n == 1:
        return sorted_values[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return sorted_values[index]
    return sorted_values[index] * (1.0 - frac) + sorted_values[index + 1] * frac


def median_iqr(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    q1 = type7(ordered, 0.25)
    q3 = type7(ordered, 0.75)
    return type7(ordered, 0.5), q3 - q1


def g17(text: str) -> str:
    """printf ``%.17g`` of the decimal recorded in the row."""
    return format(Decimal(text), ".17g")


def load(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: OrderedDict[tuple[str, str], list[dict[str, str]]] = OrderedDict()
    for row in rows:
        groups.setdefault((row["system"], row["code"]), []).append(row)
    out = []
    for (system, code), group in groups.items():
        evals = [float(row["evals_to_hit"]) for row in group]
        walls = [float(row["wall_s"]) for row in group]
        hits = sum(row["hit"] in ("1", "true") for row in group)
        evals_median, evals_iqr = median_iqr(evals)
        wall_median, wall_iqr = median_iqr(walls)
        best_text = min(group, key=lambda row: Decimal(row["best"]))["best"]
        out.append(
            {
                "system": system,
                "code": code,
                "n": str(len(group)),
                "hits": str(hits),
                "hits_over_n": f"{hits}/{len(group)}",
                "budget": group[0]["budget"],
                "evals_median": f"{evals_median:.6f}",
                "evals_iqr": f"{evals_iqr:.6f}",
                "evals_best": str(int(min(evals)))
                if min(evals).is_integer()
                else f"{min(evals):.6f}",
                "wall_median": f"{wall_median:.6f}",
                "wall_iqr": f"{wall_iqr:.6f}",
                "wall_sum": f"{sum(walls):.6f}",
                "best_energy": g17(best_text),
                "head": group[0]["head"],
            }
        )
    return out


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    q1 = type7(sample, 0.25)
    mid = type7(sample, 0.5)
    q3 = type7(sample, 0.75)
    if abs(q1 - 2.25) > 1e-12 or abs(mid - 3.5) > 1e-12 or abs(q3 - 4.75) > 1e-12:
        raise SystemExit(f"type 7 mismatch {q1} {mid} {q3}")
    if g17("-582.086642") != "-582.086642":
        raise SystemExit(f"g17 mismatch {g17('-582.086642')}")
    if g17("-580.42126629") != "-580.42126629":
        raise SystemExit(f"g17 mismatch {g17('-580.42126629')}")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        print("ok")
        return 0
    if len(argv) != 3:
        print("usage: h2h_panel.py <rows.csv> <panel.tsv>", file=sys.stderr)
        return 2
    self_test()
    rows = load(argv[1])
    summary = summarise(rows)
    fields = [
        "system",
        "code",
        "n",
        "hits",
        "hits_over_n",
        "budget",
        "evals_median",
        "evals_iqr",
        "evals_best",
        "wall_median",
        "wall_iqr",
        "wall_sum",
        "best_energy",
        "head",
    ]
    with open(argv[2], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
