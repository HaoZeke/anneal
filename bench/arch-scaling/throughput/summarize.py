#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise throughput rows into REPORT-throughput.md.

Bootstrap intervals use one random.Random(20261123) stream and 10000
resamples. Systems are au55 then lj75. Within a system, worker counts
ascend. Within a count the stream draws the median wall, the geometric
mean wall, the median rate, then the geometric mean rate. Efficiency
intervals follow, in the same system order, for each non-oversubscribed
count above 1.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows-throughput.csv"
REPORT = HERE / "REPORT-throughput.md"
SEED = 20261123
DRAWS = 10000
SYSTEMS = ("au55", "lj75")
WORKERS = (1, 2, 4, 8)
BUDGET = 1_000_000
HEADER = (
    "system,arm,seed,workers,wall_s,oracle_calls,call_budget,"
    "calls_minus_budget,calls_per_s,ledger_objective,ledger_charged,"
    "ledger_ensemble,hit,best_energy,revision,nproc,cpu_model,"
    "pinned_cores,oversubscribed"
)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of an empty sample")
    if n % 2 == 1:
        return float(ordered[n // 2])
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean needs positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def percentile(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def bootstrap_stat(values: list[float], rng: random.Random, reduce) -> tuple[float, float]:
    drawn = []
    n = len(values)
    for _ in range(DRAWS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        drawn.append(reduce(sample))
    return percentile(drawn, 0.025), percentile(drawn, 0.975)


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def ci(bounds: tuple[float, float] | None, digits: int) -> str:
    if bounds is None:
        return ""
    return f"[{fmt(bounds[0], digits)}, {fmt(bounds[1], digits)}]"


def load_rows() -> list[dict[str, str]]:
    if not ROWS.exists():
        raise SystemExit(f"missing {ROWS.name}")
    with ROWS.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or ",".join(reader.fieldnames) != HEADER:
            raise SystemExit("header does not match the protocol")
        rows = list(reader)
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["arm"] != "comm":
            raise SystemExit(f"unexpected arm {row['arm']}")
        if row["system"] not in SYSTEMS:
            raise SystemExit(f"unexpected system {row['system']}")
        workers = int(row["workers"])
        if workers not in WORKERS:
            raise SystemExit(f"unexpected workers {workers}")
        seed = int(row["seed"])
        if seed < 2000 or seed > 2011:
            raise SystemExit(f"unexpected seed {seed}")
        key = (row["system"], row["arm"], row["seed"], row["workers"])
        if key in seen:
            raise SystemExit(f"duplicate cell {key}")
        seen.add(key)
        budget = int(row["call_budget"])
        calls = int(row["oracle_calls"])
        if budget != BUDGET:
            raise SystemExit(f"budget {budget} is not {BUDGET}")
        if int(row["calls_minus_budget"]) != calls - budget:
            raise SystemExit(f"deviation field disagrees for {key}")
        ledgers = (
            int(row["ledger_objective"]),
            int(row["ledger_charged"]),
            int(row["ledger_ensemble"]),
        )
        if len(set(ledgers)) != 1 or ledgers[0] != calls:
            raise SystemExit(f"ledgers disagree for {key}")
        wall = float(row["wall_s"])
        if wall < 0.05:
            raise SystemExit(f"wall under 50 ms for {key}")
        rate = calls / wall
        if abs(float(row["calls_per_s"]) - rate) > max(1e-3, rate * 1e-6):
            raise SystemExit(f"calls per second disagree for {key}")
    return rows


def usable(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if int(row["calls_minus_budget"]) == 0]


def walls(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["wall_s"]) for row in rows]


def rates(rows: list[dict[str, str]]) -> list[float]:
    return [int(row["oracle_calls"]) / float(row["wall_s"]) for row in rows]


def by_seed(rows: list[dict[str, str]]) -> dict[int, float]:
    return {int(row["seed"]): float(row["wall_s"]) for row in rows}


def efficiency_interval(
    base: list[dict[str, str]],
    other: list[dict[str, str]],
    workers: int,
    rng: random.Random,
) -> tuple[float, tuple[float, float]]:
    left = by_seed(base)
    right = by_seed(other)
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("efficiency needs paired seeds")
    point = median([left[seed] for seed in seeds]) / (
        workers * median([right[seed] for seed in seeds])
    )
    drawn = []
    n = len(seeds)
    for _ in range(DRAWS):
        pick = [seeds[rng.randrange(n)] for _ in range(n)]
        t1 = median([left[seed] for seed in pick])
        tp = median([right[seed] for seed in pick])
        drawn.append(t1 / (workers * tp))
    return point, (percentile(drawn, 0.025), percentile(drawn, 0.975))


def cell_map(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    out: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row["system"], int(row["workers"])), []).append(row)
    for group in out.values():
        group.sort(key=lambda row: int(row["seed"]))
    return out


def write_report(rows: list[dict[str, str]]) -> None:
    kept = usable(rows)
    grouped = cell_map(kept)
    deviations = [row for row in rows if int(row["calls_minus_budget"]) != 0]
    rng = random.Random(SEED)
    stats: dict[tuple[str, int], dict[str, object]] = {}
    for system in SYSTEMS:
        for workers in WORKERS:
            sample = grouped.get((system, workers), [])
            if not sample:
                continue
            wall = walls(sample)
            rate = rates(sample)
            stats[(system, workers)] = {
                "n": len(sample),
                "hits": sum(int(row["hit"]) for row in sample),
                "wall_median": median(wall),
                "wall_median_ci": bootstrap_stat(wall, rng, median),
                "wall_geomean": geomean(wall),
                "wall_geomean_ci": bootstrap_stat(wall, rng, geomean),
                "rate_median": median(rate),
                "rate_median_ci": bootstrap_stat(rate, rng, median),
                "rate_geomean": geomean(rate),
                "rate_geomean_ci": bootstrap_stat(rate, rng, geomean),
                "oversubscribed": sample[0]["oversubscribed"] == "1",
                "pinned": sample[0]["pinned_cores"],
            }
    for system in SYSTEMS:
        base = grouped.get((system, 1), [])
        for workers in WORKERS:
            if workers == 1 or (system, workers) not in stats:
                continue
            item = stats[(system, workers)]
            if item["oversubscribed"] or not base:
                item["efficiency"] = None
                item["efficiency_ci"] = None
                continue
            point, bounds = efficiency_interval(base, grouped[(system, workers)], workers, rng)
            item["efficiency"] = point
            item["efficiency_ci"] = bounds

    revisions = sorted({row["revision"] for row in rows})
    nprocs = sorted({row["nproc"] for row in rows})
    cpus = sorted({row["cpu_model"] for row in rows})
    lines = [
        "# Throughput scaling",
        "",
        "Shared communicating ensemble, one cell at a time, budget",
        f"{BUDGET} oracle calls spent to exhaustion. Replicas are operating-system",
        "threads from `std::thread::scope`, not a rayon pool. Bootstrap intervals",
        f"use 10000 resamples from `random.Random({SEED})`.",
        "",
        f"Revision `{revisions[0] if len(revisions) == 1 else revisions}`.",
        f"nproc {', '.join(nprocs)}. CPU {', '.join(cpus)}.",
        "",
        "| system | workers | n | wall median s | wall median CI | wall geomean s | wall geomean CI | calls/s median | calls/s median CI | calls/s geomean | calls/s geomean CI | efficiency | efficiency CI | hits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system in SYSTEMS:
        for workers in WORKERS:
            item = stats.get((system, workers))
            if item is None:
                lines.append(
                    f"| {system} | {workers} | 0 |  |  |  |  |  |  |  |  |  |  |  |"
                )
                continue
            if workers == 1:
                efficiency = "1"
                efficiency_ci = ""
            elif item["oversubscribed"]:
                efficiency = "omitted"
                efficiency_ci = ""
            elif item.get("efficiency") is None:
                efficiency = ""
                efficiency_ci = ""
            else:
                efficiency = fmt(float(item["efficiency"]), 4)
                efficiency_ci = ci(item["efficiency_ci"], 4)  # type: ignore[arg-type]
            lines.append(
                "| {system} | {workers} | {n} | {wall} | {wall_ci} | {geo} | {geo_ci} | {rate} | {rate_ci} | {rgeo} | {rgeo_ci} | {eff} | {eff_ci} | {hits}/{n} |".format(
                    system=system,
                    workers=workers,
                    n=item["n"],
                    wall=fmt(float(item["wall_median"]), 3),
                    wall_ci=ci(item["wall_median_ci"], 3),  # type: ignore[arg-type]
                    geo=fmt(float(item["wall_geomean"]), 3),
                    geo_ci=ci(item["wall_geomean_ci"], 3),  # type: ignore[arg-type]
                    rate=fmt(float(item["rate_median"]), 1),
                    rate_ci=ci(item["rate_median_ci"], 1),  # type: ignore[arg-type]
                    rgeo=fmt(float(item["rate_geomean"]), 1),
                    rgeo_ci=ci(item["rate_geomean_ci"], 1),  # type: ignore[arg-type]
                    eff=efficiency,
                    eff_ci=efficiency_ci,
                    hits=item["hits"],
                )
            )
    lines.extend(["", "## Deviations", ""])
    if deviations:
        for row in deviations:
            lines.append(
                f"- {row['system']} seed {row['seed']} workers {row['workers']}: "
                f"oracle_calls {row['oracle_calls']} minus budget {row['calls_minus_budget']}"
            )
    else:
        lines.append("Every stored cell spent the budget. The three ledger totals agree on each row.")
    lines.extend(["", "## Pins and hits", ""])
    for system in SYSTEMS:
        for workers in WORKERS:
            item = stats.get((system, workers))
            if item is None:
                lines.append(f"- {system} p={workers}: no row")
                continue
            lines.append(
                f"- {system} p={workers}: pin `{item['pinned']}`, "
                f"hits {item['hits']}/{item['n']}, "
                f"oversubscribed {'yes' if item['oversubscribed'] else 'no'}"
            )
    lines.extend(
        [
            "",
            "Hits use the reference plus `1e-3` and do not enter the intervals.",
            "Efficiency is `T1 / (p Tp)` from median wall times of paired seeds.",
            "A cell with `p` above nproc is omitted from efficiency.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def self_test() -> None:
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert abs(geomean([1.0, 4.0]) - 2.0) < 1e-12
    assert abs(percentile([1.0, 2.0, 3.0, 4.0], 0.0) - 1.0) < 1e-12
    assert abs(percentile([1.0, 2.0, 3.0, 4.0], 1.0) - 4.0) < 1e-12
    rng = random.Random(SEED)
    first = bootstrap_stat([1.0, 2.0, 3.0, 4.0], rng, median)
    again = bootstrap_stat([1.0, 2.0, 3.0, 4.0], random.Random(SEED), median)
    assert first == again
    assert first[0] <= 2.5 <= first[1]
    point, bounds = efficiency_interval(
        [{"seed": "1", "wall_s": "10"}, {"seed": "2", "wall_s": "14"}],
        [{"seed": "1", "wall_s": "4"}, {"seed": "2", "wall_s": "6"}],
        2,
        random.Random(SEED),
    )
    assert abs(point - (12.0 / (2.0 * 5.0))) < 1e-12
    assert bounds[0] <= point <= bounds[1]


def main() -> None:
    self_test()
    if "--check" in sys.argv:
        return
    write_report(load_rows())


if __name__ == "__main__":
    main()
