#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint the LJ75 ASE head-to-head.

Reads bench/ase-lj75/rows.csv. Writes bench/ase-lj75/summary.csv.
Misses are censored at the call budget. Does not start a search.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "bench" / "ase-lj75" / "rows.csv"
SUMMARY = ROOT / "bench" / "ase-lj75" / "summary.csv"
REFERENCE = -397.492331
SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "ase_bh", "ase_mh")
PAPER_BUDGETS = {75: 4_000_000}


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    return float(left * (1.0 - weight) + ordered[int(lower)] * weight)


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def load() -> dict[str, list[dict[str, str]]]:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {arm: [] for arm in ARMS}
    for row in rows:
        if row["system"] != "lj75" or row["budget"] != "4000000" or row["host"] != "bench":
            raise SystemExit(f"unexpected row {row}")
        grouped[row["arm"]].append(row)
    for arm in ARMS:
        seeds = tuple(sorted(int(row["seed"]) for row in grouped[arm]))
        if seeds != SEEDS:
            raise SystemExit(f"{arm} seeds {seeds}")
        grouped[arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def fisher(hits: int, base: int) -> float:
    return float(fisher_exact([[hits, 12 - hits], [base, 12 - base]])[1])


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(set(left + right)) == 1:
        return len(left) * len(right) / 2.0, 1.0
    result = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    grouped = load()
    anneal = grouped["anneal"]
    lines = [
        "arm,hits,n,median_best,gap_to_reference,median_wall_s,median_inside_s,median_outside_s,median_over_anneal,median_anneal_over,fisher_p,mann_whitney_U,mann_whitney_p,wall_stops,wall_cap_s",
    ]
    base_calls = [calls(row) for row in anneal]
    base_hits = sum(row["hit"] == "1" for row in anneal)
    caps = sorted({row["cap_s"] for row in grouped["ase_bh"] + grouped["ase_mh"] if row.get("cap_s")})
    print("LJ75 seeds 2000-2011, budget 4000000, host bench, reference -397.492331")
    if caps:
        print(f"ASE wall cap {caps[0]}")
    for arm in ARMS:
        group = grouped[arm]
        hits = sum(row["hit"] == "1" for row in group)
        best = median([float(row["best"]) for row in group])
        wall = median([float(row["wall_s"]) for row in group])
        inside = median([float(row["inside_s"]) for row in group])
        outside = median([float(row["outside_s"]) for row in group])
        wall_stops = sum(row.get("stop") == "wall" for row in group)
        cap = group[0].get("cap_s") or ""
        if arm == "anneal":
            over = back = fisher_p = 1.0
            u_stat, mw_p = len(base_calls) * len(base_calls) / 2.0, 1.0
            cap = ""
        else:
            ratios = [float(a["wall_s"]) / float(b["wall_s"]) for a, b in zip(group, anneal)]
            back_ratios = [float(b["wall_s"]) / float(a["wall_s"]) for a, b in zip(group, anneal)]
            over = median(ratios)
            back = median(back_ratios)
            fisher_p = fisher(hits, base_hits)
            u_stat, mw_p = mann_whitney([calls(row) for row in group], base_calls)
        print(
            f"{arm} hits {hits}/12 median_best {best:.8f} gap {best - REFERENCE:.8f} "
            f"wall {wall:.6f} inside {inside:.6f} outside {outside:.6f} "
            f"over {over:.6f} back {back:.6f} fisher {fisher_p:.6g} U {u_stat:.6g} mw {mw_p:.6g} "
            f"wall_stops {wall_stops}"
        )
        lines.append(
            f"{arm},{hits},12,{best:.8f},{best - REFERENCE:.8f},{wall:.6f},{inside:.6f},{outside:.6f},"
            f"{over:.6f},{back:.6f},{fisher_p:.6g},{u_stat:.6g},{mw_p:.6g},{wall_stops},{cap}"
        )
    SUMMARY.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
