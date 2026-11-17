#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the Sutton–Chen c-MCTS head-to-head.

Hits are a quenched energy within 1e-3 of the Cambridge global minimum.
A miss is stored as the shared budget, so the rank test includes it.
Medians are Hyndman–Fan type 7. Fisher is the two-sided exact test on
the hit counts. Mann–Whitney is the two-sided normal approximation with
the tie correction and a continuity correction of one half, on the call
counts, the wall, the time inside the potential and the time outside it.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

SUMMARY_FIELDS = [
    "system",
    "anneal_arm",
    "anneal_hits",
    "anneal_n",
    "cmcts_hits",
    "cmcts_n",
    "fisher_p",
    "mannwhitney_p_calls",
    "mannwhitney_p_wall",
    "mannwhitney_p_pot",
    "mannwhitney_p_outside",
    "anneal_calls_median",
    "cmcts_calls_median",
    "anneal_wall_median_s",
    "cmcts_wall_median_s",
    "anneal_pot_median_s",
    "anneal_outside_median_s",
    "cmcts_pot_median_s",
    "cmcts_outside_median_s",
    "calls_per_minimisation",
    "paper_anchor_calls",
    "bh_anchor_calls",
]


def type7(ordered: list[float], p: float) -> float:
    n = len(ordered)
    if n == 0:
        return float("nan")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    index = lower - 1
    if frac == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - frac) + ordered[index + 1] * frac


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    a, b = hits_a, n_a - hits_a
    c, d = hits_b, n_b - hits_b

    def prob(aa: int, bb: int, cc: int, dd: int) -> float:
        total = aa + bb + cc + dd
        return math.comb(aa + bb, aa) * math.comb(cc + dd, cc) / math.comb(total, aa + cc)

    base = prob(a, b, c, d)
    total = 0.0
    for aa in range(a + b + 1):
        cc = a + c - aa
        bb = a + b - aa
        dd = c + d - cc
        if min(aa, bb, cc, dd) < 0:
            continue
        p = prob(aa, bb, cc, dd)
        if p <= base + 1e-12:
            total += p
    return min(1.0, total)


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided tie-corrected Mann–Whitney p-value."""
    n1 = len(left)
    n2 = len(right)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = [(value, 0) for value in left] + [(value, 1) for value in right]
    pooled.sort(key=lambda item: item[0])
    ranks = [0.0] * len(pooled)
    i = 0
    tie_term = 0.0
    while i < len(pooled):
        j = i + 1
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        # Average rank, 1-based, over the tied block [i, j).
        average = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[k] = average
        block = j - i
        tie_term += block**3 - block
        i = j
    rank_1 = sum(rank for rank, (_, side) in zip(ranks, pooled) if side == 0)
    u = n1 * n2 + n1 * (n1 + 1) / 2.0 - rank_1
    mean = n1 * n2 / 2.0
    n = n1 + n2
    if n < 2:
        return 1.0
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0.0:
        return 1.0
    z_num = abs(u - mean) - 0.5
    if z_num <= 0.0:
        return 1.0
    z = z_num / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2.0))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _latest(rows: list[dict[str, str]], system: str, arm: str) -> list[dict[str, str]]:
    chosen = [row for row in rows if row["system"] == system and row["arm"] == arm]
    if not chosen:
        return []
    revision = chosen[-1]["code_revision"]
    return [row for row in chosen if row["code_revision"] == revision]


def _hits(rows: list[dict[str, str]]) -> int:
    return sum(row["hit"] == "1" for row in rows)


def _column(rows: list[dict[str, str]], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    systems = []
    for row in rows:
        if row["system"] not in systems:
            systems.append(row["system"])
    lines = []
    for system in systems:
        cmcts = _latest(rows, system, "cmcts")
        anneal_arms = []
        for row in rows:
            if row["system"] == system and row["arm"] not in ("cmcts",) and row["arm"] not in anneal_arms:
                anneal_arms.append(row["arm"])
        for arm in anneal_arms:
            anneal = _latest(rows, system, arm)
            shared = sorted(set(row["seed"] for row in anneal) & set(row["seed"] for row in cmcts))
            anneal_shared = [row for row in anneal if row["seed"] in shared]
            cmcts_shared = [row for row in cmcts if row["seed"] in shared]
            calls_per = anneal[0].get("calls_per_minimisation", "") if anneal else ""
            paper_anchor = anneal[0].get("paper_anchor_calls", "") if anneal else ""
            bh_anchor = anneal[0].get("bh_anchor_calls", "") if anneal else ""
            lines.append(
                {
                    "system": system,
                    "anneal_arm": arm,
                    "anneal_hits": str(_hits(anneal)),
                    "anneal_n": str(len(anneal)),
                    "cmcts_hits": str(_hits(cmcts_shared)),
                    "cmcts_n": str(len(cmcts_shared)),
                    "fisher_p": f"{fisher_two_sided(_hits(anneal_shared), len(anneal_shared), _hits(cmcts_shared), len(cmcts_shared)):.6g}",
                    "mannwhitney_p_calls": f"{mann_whitney(_column(anneal_shared, 'evals_to_hit'), _column(cmcts_shared, 'evals_to_hit')):.6g}",
                    "mannwhitney_p_wall": f"{mann_whitney(_column(anneal_shared, 'wall_s'), _column(cmcts_shared, 'wall_s')):.6g}",
                    "mannwhitney_p_pot": f"{mann_whitney(_column(anneal_shared, 'pot_s'), _column(cmcts_shared, 'pot_s')):.6g}",
                    "mannwhitney_p_outside": f"{mann_whitney(_column(anneal_shared, 'outside_s'), _column(cmcts_shared, 'outside_s')):.6g}",
                    "anneal_calls_median": f"{median(_column(anneal, 'evals_to_hit')):.6f}",
                    "cmcts_calls_median": f"{median(_column(cmcts, 'evals_to_hit')):.6f}" if cmcts else "",
                    "anneal_wall_median_s": f"{median(_column(anneal, 'wall_s')):.6f}",
                    "cmcts_wall_median_s": f"{median(_column(cmcts, 'wall_s')):.6f}" if cmcts else "",
                    "anneal_pot_median_s": f"{median(_column(anneal, 'pot_s')):.6f}",
                    "anneal_outside_median_s": f"{median(_column(anneal, 'outside_s')):.6f}",
                    "cmcts_pot_median_s": f"{median(_column(cmcts, 'pot_s')):.6f}" if cmcts else "",
                    "cmcts_outside_median_s": f"{median(_column(cmcts, 'outside_s')):.6f}" if cmcts else "",
                    "calls_per_minimisation": calls_per,
                    "paper_anchor_calls": paper_anchor,
                    "bh_anchor_calls": bh_anchor,
                }
            )
    return lines


def write_summary(seeds: Path, summary: Path) -> list[dict[str, str]]:
    lines = summarise(read_rows(seeds))
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)
    return lines


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    seeds = root / "bench" / "cmcts-au" / "seeds.csv"
    summary = root / "bench" / "cmcts-au" / "summary.csv"
    lines = write_summary(seeds, summary)
    for line in lines:
        print(
            "{system} {anneal_arm}: anneal {anneal_hits}/{anneal_n} cmcts {cmcts_hits}/{cmcts_n} "
            "fisher {fisher_p} mw_calls {mannwhitney_p_calls} mw_wall {mannwhitney_p_wall} "
            "calls {anneal_calls_median} {cmcts_calls_median} "
            "wall {anneal_wall_median_s} {cmcts_wall_median_s} "
            "pot {anneal_pot_median_s} {cmcts_pot_median_s} "
            "outside {anneal_outside_median_s} {cmcts_outside_median_s}".format(**line)
        )


if __name__ == "__main__":
    sys.exit(main())
