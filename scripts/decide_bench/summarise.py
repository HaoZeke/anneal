#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise bench/decide/rows.csv against a reference arm.

One line per (variant, system) and one ``total`` line per variant.
``--reference`` names the arm the others are tested against, ``develop``
by default; the other unswitched arm is left out. ``--revision`` keeps
only rows whose ``code_revision`` starts with one of the given
prefixes, so one session is not pooled with an earlier one. Hits use a two-sided
Fisher exact test against the reference; calls to
hit, with a miss at the full budget, use a two-sided Mann-Whitney U
with the normal approximation, tie and continuity corrections. Medians
and interquartile ranges are Hyndman-Fan type 7. ``pot`` is time inside
the potential and ``outside`` the rest of the wall.

The land criterion: no system loses hits at Fisher p < 0.05, the hit
total rises, and the median calls do not rise on any system.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.05

FIELDS = (
    "variant",
    "system",
    "n",
    "hits",
    "baseline_hits",
    "fisher_p",
    "calls_median",
    "calls_iqr",
    "baseline_calls_median",
    "mannwhitney_p",
    "pot_median_s",
    "outside_median_s",
    "code_revision",
    "host",
)


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lower = int(h)
    frac = h - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_two_sided(a_hits: int, a_n: int, b_hits: int, b_n: int) -> float:
    a, b, c, d = a_hits, a_n - a_hits, b_hits, b_n - b_hits
    row, col, total = a + b, a + c, a + b + c + d

    def prob(x: int) -> float:
        return (
            math.comb(row, x) * math.comb(total - row, col - x) / math.comb(total, col)
        )

    observed = prob(a)
    lo, hi = max(0, row + col - total), min(row, col)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def mann_whitney_two_sided(x: list[float], y: list[float]) -> float:
    n1, n2 = len(x), len(y)
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = [0.0] * len(pooled)
    ties = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = rank
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - mean) - 0.5) / math.sqrt(var)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0)))


def summarise(rows: list[dict[str, str]], reference: str = "develop") -> list[dict[str, str]]:
    others = {"baseline", "develop"} - {reference}
    rows = [row for row in rows if row["variant"] not in others]
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    systems: list[str] = []
    variants: list[str] = []
    for row in rows:
        groups.setdefault((row["variant"], row["system"]), []).append(row)
        if row["system"] not in systems:
            systems.append(row["system"])
        if row["variant"] not in variants:
            variants.append(row["variant"])
    if reference in variants:
        variants.remove(reference)
        variants.insert(0, reference)
    out = []
    for variant in variants:
        total_hits = total_base = total_n = 0
        for system in systems:
            mine = groups.get((variant, system))
            base = groups.get((reference, system))
            if not mine:
                continue
            seeds = {r["seed"] for r in mine}
            if base:
                base = [r for r in base if r["seed"] in seeds]
            hits = sum(r["hit"] == "1" for r in mine)
            calls = [float(r["evals_to_hit"]) for r in mine]
            line = {
                "variant": variant,
                "system": system,
                "n": str(len(mine)),
                "hits": str(hits),
                "calls_median": f"{type7(calls, 0.5):.1f}",
                "calls_iqr": f"{type7(calls, 0.75) - type7(calls, 0.25):.1f}",
                "pot_median_s": f"{type7([float(r['pot_s']) for r in mine], 0.5):.3f}",
                "outside_median_s": f"{type7([float(r['outside_s']) for r in mine], 0.5):.3f}",
                "code_revision": " ".join(sorted({r["code_revision"][:8] for r in mine})),
                "host": " ".join(sorted({r["host"] for r in mine})),
                "baseline_hits": "",
                "fisher_p": "",
                "baseline_calls_median": "",
                "mannwhitney_p": "",
            }
            if base and variant != reference:
                base_hits = sum(r["hit"] == "1" for r in base)
                base_calls = [float(r["evals_to_hit"]) for r in base]
                line["baseline_hits"] = str(base_hits)
                line["fisher_p"] = f"{fisher_two_sided(hits, len(mine), base_hits, len(base)):.4g}"
                line["baseline_calls_median"] = f"{type7(base_calls, 0.5):.1f}"
                line["mannwhitney_p"] = f"{mann_whitney_two_sided(calls, base_calls):.4g}"
                total_base += base_hits
            total_hits += hits
            total_n += len(mine)
            out.append(line)
        out.append(
            {
                "variant": variant,
                "system": "total",
                "n": str(total_n),
                "hits": str(total_hits),
                "baseline_hits": "" if variant == reference else str(total_base),
                "fisher_p": "",
                "calls_median": "",
                "calls_iqr": "",
                "baseline_calls_median": "",
                "mannwhitney_p": "",
                "pot_median_s": "",
                "outside_median_s": "",
                "code_revision": "",
                "host": "",
            }
        )
    return out


def lands(lines: list[dict[str, str]], variant: str) -> bool:
    mine = [l for l in lines if l["variant"] == variant and l["system"] != "total"]
    if not mine:
        return False
    for line in mine:
        if not line["baseline_hits"]:
            return False
        lost = int(line["hits"]) < int(line["baseline_hits"])
        if lost and float(line["fisher_p"]) < ALPHA:
            return False
        if float(line["calls_median"]) > float(line["baseline_calls_median"]):
            return False
    total = next(l for l in lines if l["variant"] == variant and l["system"] == "total")
    return int(total["hits"]) > int(total["baseline_hits"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=ROOT / "bench/decide/rows.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "bench/decide/summary.csv")
    parser.add_argument("--reference", default="develop")
    parser.add_argument("--revision", action="append", default=[])
    args = parser.parse_args(argv)
    rows = list(csv.DictReader(args.rows.open()))
    if args.revision:
        rows = [
            row
            for row in rows
            if any(row["code_revision"].startswith(prefix) for prefix in args.revision)
        ]
    lines = summarise(rows, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)
    for variant in sorted({l["variant"] for l in lines} - {args.reference}):
        print(f"{variant}: {'lands' if lands(lines, variant) else 'stays off'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
