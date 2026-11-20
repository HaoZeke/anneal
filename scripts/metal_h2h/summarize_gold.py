#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the Gupta gold rows in bench/gold-gupta/seeds.csv.

Medians are Hyndman–Fan type 7. Inside and outside microseconds are the
median of the per-seed rates, ``1e6 * seconds / charged``. ``code_revision``
is the tree the row was annealed on; older rows store that in ``revision``.
``host`` is the UTS hostname. Rows are never overwritten.

On ``351152695c97`` all twelve Au75 seeds, including 2007, reach
−263.27183577 and all twelve Au98 seeds reach −345.98489186. Routing a
physical bond off the reduced-unit fivefold SOAP residual and onto the
observed-cloud residual leaves those minima where they are and moves
the time outside the potential from about 12 s to about 551 s, so this
tree does not route bonds that way.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "bench" / "gold-gupta" / "seeds.csv"


def median(values: list[float]) -> float:
    """Hyndman–Fan type 7 median."""
    if not values:
        raise ValueError("median of an empty sample")
    ordered = sorted(values)
    n = len(ordered)
    position = 1.0 + (n - 1) * 0.5
    lower = int(position)
    fraction = position - lower
    index = lower - 1
    if fraction == 0.0:
        return ordered[index]
    return ordered[index] * (1.0 - fraction) + ordered[index + 1] * fraction


def revision_of(row: dict[str, str]) -> str:
    named = row.get("code_revision", "").strip()
    if named:
        return named
    return row.get("revision", "").strip()


def load() -> list[dict[str, str]]:
    with SEEDS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]]) -> str:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("code") != "anneal":
            continue
        groups[(revision_of(row), row["system"], row["code"])].append(row)
    lines = [
        "revision,system,n,hits,median_calls,median_wall_s,median_inside_s,"
        "median_outside_s,inside_us,outside_us,median_best,host"
    ]
    for key in sorted(groups):
        group = groups[key]
        charged = [max(int(row["charged"]), 1) for row in group]
        hits = sum(row["hit"] == "true" for row in group)
        hosts = sorted({row.get("host", "").strip() for row in group if row.get("host", "").strip()})
        lines.append(
            ",".join(
                [
                    key[0],
                    key[1],
                    str(len(group)),
                    f"{hits}/{len(group)}",
                    f"{median([float(row['evals_to_hit']) for row in group]):.1f}",
                    f"{median([float(row['wall_s']) for row in group]):.3f}",
                    f"{median([float(row['pot_s']) for row in group]):.3f}",
                    f"{median([float(row['outside_s']) for row in group]):.3f}",
                    f"{median([1e6 * float(row['pot_s']) / c for row, c in zip(group, charged)]):.3f}",
                    f"{median([1e6 * float(row['outside_s']) / c for row, c in zip(group, charged)]):.3f}",
                    f"{median([float(row['best']) for row in group]):.8f}",
                    ";".join(hosts),
                ]
            )
        )
    stuck = [
        row
        for row in rows
        if row.get("system") == "au75" and row.get("seed") == "2007" and row.get("code") == "anneal"
    ]
    lines.append("au75_seed_2007")
    for row in stuck:
        lines.append(
            f"{revision_of(row)},{row['best']},{row.get('host', '').strip()}"
        )
    return "\n".join(lines)


def main() -> int:
    text = summarise(load())
    sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
