#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize rows-tth.csv into REPORT-tth.md."""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROWS = HERE / "rows-tth.csv"
REPORT = HERE / "REPORT-tth.md"
SEED = 20261123
DRAWS = 10_000
WORKERS = (1, 2, 4, 8)

sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import fisher_exact  # noqa: E402


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return float("nan")
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap(values: list[float], rng: random.Random, statistic) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    estimates = [
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(DRAWS)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def wilson(hits: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    observed = hits / n
    denominator = 1.0 + z * z / n
    centre = (observed + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(observed * (1.0 - observed) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return centre - radius, centre + radius


def mannwhitney(left: list[float], right: list[float]) -> float:
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    ranks: dict[float, float] = {}
    ties = 0.0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2.0
        block = end - index + 1
        ties += block**3 - block
        index = end + 1
    u = sum(ranks[value] for value in left) - n1 * (n1 + 1) / 2.0
    total = n1 + n2
    variance = n1 * n2 / 12.0 * ((total + 1) - ties / (total * (total - 1)))
    if variance <= 0.0:
        return 1.0
    deviation = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return min(1.0, math.erfc(deviation / math.sqrt(2.0 * variance)))


def load() -> dict[str, dict[int, list[dict[str, str]]]]:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, dict[int, list[dict[str, str]]]] = {}
    for row in rows:
        grouped.setdefault(row["system"], {}).setdefault(int(row["workers"]), []).append(row)
    for system, cells in grouped.items():
        for workers, cell in cells.items():
            cell.sort(key=lambda row: int(row["seed"]))
            seeds = [int(row["seed"]) for row in cell]
            if seeds != list(range(2000, 2012)):
                raise SystemExit(f"{system} p={workers} seeds are {seeds}")
            for row in cell:
                values = {
                    int(row["oracle_calls"]),
                    int(row["ledger_charged"]),
                    int(row["ledger_sum"]),
                    int(row["objective_calls"]),
                }
                if len(values) != 1:
                    raise SystemExit(
                        f"{system} seed {row['seed']} p={workers} ledger disagreement"
                    )
    return grouped


def number(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.{digits}f}"


def ci(values: tuple[float, float], digits: int = 3) -> str:
    return f"[{number(values[0], digits)}, {number(values[1], digits)}]"


def efficiency_interval(
    base: list[dict[str, str]],
    cell: list[dict[str, str]],
    workers: int,
    rng: random.Random,
) -> tuple[float, float]:
    estimates = []
    for _ in range(DRAWS):
        indices = [rng.randrange(len(base)) for _ in base]
        left = [
            float(base[index]["wall_to_hit_s"])
            for index in indices
            if base[index]["wall_to_hit_s"]
        ]
        right = [
            float(cell[index]["wall_to_hit_s"])
            for index in indices
            if cell[index]["wall_to_hit_s"]
        ]
        if left and right:
            estimates.append(median(left) / (workers * median(right)))
    if not estimates:
        return float("nan"), float("nan")
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def write_report(groups: dict[str, dict[int, list[dict[str, str]]]]) -> None:
    rng = random.Random(SEED)
    names = {"lj38": "LJ38", "au38": "Au38 Gupta", "m55r6": "M55 rho=6 Morse"}
    lines = [
        "# Time-to-hit worker scaling",
        "",
        "Shared communicating ensemble, replicas as operating-system threads,",
        "seeds 2000-2011, and p = 1, 2, 4, 8. Runs stop at the first objective",
        "energy at or below the reference plus the panel tolerance of 0.001.",
        "Misses enter calls-to-hit as budget + 1. Wall medians use hits only.",
        "Intervals on calls, geometric means, wall time, and efficiency are",
        "bootstrap percentile 95% intervals from 10000 resamples with seed",
        "20261123. Hit intervals are Wilson 95% intervals. Tests are two-sided",
        "against p=1; MWU uses the tie-corrected normal approximation and",
        "continuity correction.",
        "",
        "References and sources:",
        "",
        "- LJ38: -173.928427; paper/jctc/manuscript.tex, Cambridge references.",
        "- Au38 Gupta: -130.54001708; examples/h2h_metal.rs, Au38 system specification.",
        "- M55 rho=6 Morse: -250.286609; paper/jctc/manuscript.tex, Cambridge references.",
        "- LJ38 budget: 400000; PAPER_BUDGETS in scripts/paper_0120/run_panel.py.",
        "",
    ]
    if "au38" in groups and set(groups["au38"]) == {1}:
        au_hits = sum(int(row["hit"]) for row in groups["au38"][1])
        lines.extend(
            [
                f"Au38 p=1 gate: {au_hits}/12 hits. It did not meet the 10/12 gate,",
                "so the complete second matrix uses M55 rho=6 Morse.",
                "",
            ]
        )
    for system in ("lj38", "au38", "m55r6"):
        if system not in groups:
            continue
        cells = groups[system]
        lines.extend(
            [
                f"## {names[system]}",
                "",
                "| p | hits (Wilson CI) | calls median (CI) | calls geometric mean (CI) | wall median s (CI) | efficiency (CI) | Fisher p | MWU p | ledger total |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        base = cells[1]
        base_calls = [float(row["calls_to_hit"]) for row in base]
        base_walls = [float(row["wall_to_hit_s"]) for row in base if row["wall_to_hit_s"]]
        base_wall = median(base_walls)
        for workers in sorted(cells):
            cell = cells[workers]
            hits = sum(int(row["hit"]) for row in cell)
            calls = [float(row["calls_to_hit"]) for row in cell]
            walls = [float(row["wall_to_hit_s"]) for row in cell if row["wall_to_hit_s"]]
            med = median(calls)
            med_ci = bootstrap(calls, rng, median)
            geo = geometric_mean(calls)
            geo_ci = bootstrap(calls, rng, geometric_mean)
            wall = median(walls)
            wall_ci = bootstrap(walls, rng, median)
            if workers == 1:
                efficiency = 1.0
                efficiency_ci = (1.0, 1.0)
                fisher = mwu = float("nan")
            elif walls and base_walls:
                efficiency = base_wall / (workers * wall)
                efficiency_ci = efficiency_interval(base, cell, workers, rng)
                fisher = fisher_exact(hits, len(cell), sum(int(row["hit"]) for row in base), len(base))
                mwu = mannwhitney(calls, base_calls)
            else:
                efficiency = float("nan")
                efficiency_ci = (float("nan"), float("nan"))
                fisher = fisher_exact(hits, len(cell), sum(int(row["hit"]) for row in base), len(base))
                mwu = mannwhitney(calls, base_calls)
            ledger_total = sum(int(row["oracle_calls"]) for row in cell)
            hit_ci = wilson(hits, len(cell))
            lines.append(
                f"| {workers} | {hits}/12 ({ci(hit_ci)}) | "
                f"{number(med)} ({ci(med_ci)}) | {number(geo)} ({ci(geo_ci)}) | "
                f"{number(wall)} ({ci(wall_ci)}) | "
                f"{number(efficiency, 4)} ({ci(efficiency_ci, 4)}) | "
                f"{number(fisher, 4)} | {number(mwu, 4)} | {ledger_total} |"
            )
        lines.extend(
            [
                "",
                "For every row, oracle_calls = ledger_charged = ledger_sum =",
                "objective_calls. The table reports their per-cell sum.",
                "",
            ]
        )
    revisions = sorted(
        {
            row["revision"]
            for cells in groups.values()
            for cell in cells.values()
            for row in cell
        }
    )
    lines.extend(["Measurement revision: " + " ".join(revisions), ""])
    REPORT.write_text("\n".join(lines))


def self_test() -> None:
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert abs(geometric_mean([1.0, 4.0]) - 2.0) < 1e-12
    assert wilson(0, 12)[0] >= 0.0
    assert 0.0 < mannwhitney([1.0, 2.0], [1.0, 2.0]) <= 1.0


if __name__ == "__main__":
    self_test()
    if ROWS.exists():
        write_report(load())
    elif "--check" not in sys.argv:
        raise SystemExit(f"{ROWS} does not exist")
