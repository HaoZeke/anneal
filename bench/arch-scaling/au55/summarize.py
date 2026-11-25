#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarise bench/arch-scaling/au55/rows.csv into REPORT.md.

Bootstrap intervals use one random.Random(20261123) stream and 10000
resamples. Worker counts are visited in ascending order. Within a count
the stream draws the hit count, then the calls-to-hit median, then the
wall-to-hit median. Efficiency intervals follow, for each
non-oversubscribed count above 1, ascending.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / "rows.csv"
REPORT = HERE / "REPORT.md"
ROOT = HERE.parents[2]
SEED = 20261123
DRAWS = 10000
WORKERS = (1, 2, 4, 8, 16)

sys.path.insert(0, str(ROOT / "scripts"))
from paired_stats import fisher_exact  # noqa: E402


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of an empty sample")
    if n % 2 == 1:
        return float(ordered[n // 2])
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def percentile(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = p * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def mannwhitney(left: list[float], right: list[float]) -> float:
    """Two-sided tie-corrected normal approximation with continuity correction."""
    if not left or not right:
        return float("nan")
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        block = j - i + 1
        ties += block**3 - block
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return min(1.0, math.erfc(dev / math.sqrt(2.0 * var)))


def bootstrap_stat(values: list[float], rng: random.Random, reduce) -> tuple[float, float]:
    drawn = []
    n = len(values)
    for _ in range(DRAWS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        drawn.append(reduce(sample))
    return percentile(drawn, 0.025), percentile(drawn, 0.975)


def load_rows() -> list[dict[str, str]]:
    with ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("rows.csv has no data")
    return rows


def group(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row["system"] != "au55" or row["arm"] != "comm":
            raise SystemExit(f"unexpected row {row['system']} {row['arm']}")
        workers = int(row["workers"])
        out.setdefault(workers, []).append(row)
    for workers, group_rows in out.items():
        seeds = [int(row["seed"]) for row in group_rows]
        if seeds != list(range(2000, 2012)):
            raise SystemExit(f"workers {workers} seeds are {seeds}")
        if len({row["call_budget"] for row in group_rows}) != 1:
            raise SystemExit(f"workers {workers} budgets differ")
        if len({row["oversubscribed"] for row in group_rows}) != 1:
            raise SystemExit(f"workers {workers} oversubscribed flag differs")
    return out


def hit_count(rows: list[dict[str, str]]) -> int:
    return sum(int(row["hit"]) for row in rows)


def calls(rows: list[dict[str, str]]) -> list[float]:
    return [float(row["calls_to_hit"]) for row in rows]


def walls(rows: list[dict[str, str]]) -> list[float | None]:
    out: list[float | None] = []
    for row in rows:
        text = row["wall_to_hit_s"].strip()
        out.append(float(text) if text else None)
    return out


def fmt_num(value: float) -> str:
    if math.isnan(value):
        return "na"
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}"


def interval(lo: float, hi: float) -> str:
    return f"[{fmt_num(lo)}, {fmt_num(hi)}]"


def write_report(groups: dict[int, list[dict[str, str]]]) -> None:
    present = [workers for workers in WORKERS if workers in groups]
    if not present:
        raise SystemExit("no ladder rungs in rows.csv")
    budget = int(groups[present[0]][0]["call_budget"])
    for workers in present:
        if int(groups[workers][0]["call_budget"]) != budget:
            raise SystemExit("call budgets differ across rungs")
    revision = groups[present[0]][0]["revision"]
    nproc = groups[present[0]][0]["nproc"]
    cpu = groups[present[0]][0]["cpu_model"]
    rng = random.Random(SEED)
    stats = []
    for workers in present:
        rows = groups[workers]
        hits = hit_count(rows)
        hit_lo, hit_hi = bootstrap_stat(
            [float(int(row["hit"])) for row in rows], rng, sum
        )
        call_values = calls(rows)
        call_med = median(call_values)
        call_lo, call_hi = bootstrap_stat(call_values, rng, median)
        hit_walls = [value for value in walls(rows) if value is not None]
        if hit_walls:
            wall_med = median(hit_walls)
            wall_lo, wall_hi = bootstrap_stat(hit_walls, rng, median)
        else:
            wall_med = wall_lo = wall_hi = float("nan")
        base = groups.get(1)
        if base is None or workers == 1:
            fisher = float("nan")
            mwu = float("nan")
        else:
            fisher = fisher_exact(hits, len(rows), hit_count(base), len(base))
            mwu = mannwhitney(call_values, calls(base))
        over = rows[0]["oversubscribed"] == "1"
        stats.append(
            {
                "workers": workers,
                "hits": hits,
                "n": len(rows),
                "hit_lo": hit_lo,
                "hit_hi": hit_hi,
                "call_med": call_med,
                "call_lo": call_lo,
                "call_hi": call_hi,
                "wall_med": wall_med,
                "wall_lo": wall_lo,
                "wall_hi": wall_hi,
                "fisher": fisher,
                "mwu": mwu,
                "over": over,
                "pinned": rows[0]["pinned_cores"],
                "wall_n": len(hit_walls),
            }
        )
    by_workers = {item["workers"]: item for item in stats}
    base_walls = walls(groups[1]) if 1 in groups else None
    for item in stats:
        workers = item["workers"]
        if item["over"] or workers == 1 or base_walls is None:
            item["eff"] = float("nan")
            item["eff_lo"] = float("nan")
            item["eff_hi"] = float("nan")
            item["eff_kept"] = 0
            continue
        rung_walls = walls(groups[workers])
        point_a = [value for value in base_walls if value is not None]
        point_b = [value for value in rung_walls if value is not None]
        if not point_a or not point_b:
            item["eff"] = float("nan")
            item["eff_lo"] = float("nan")
            item["eff_hi"] = float("nan")
            item["eff_kept"] = 0
            continue
        item["eff"] = median(point_a) / (workers * median(point_b))
        drawn = []
        n = len(base_walls)
        for _ in range(DRAWS):
            idx = [rng.randrange(n) for _ in range(n)]
            sample_a = [base_walls[i] for i in idx if base_walls[i] is not None]
            sample_b = [rung_walls[i] for i in idx if rung_walls[i] is not None]
            if not sample_a or not sample_b:
                continue
            drawn.append(median(sample_a) / (workers * median(sample_b)))
        item["eff_kept"] = len(drawn)
        if drawn:
            item["eff_lo"] = percentile(drawn, 0.025)
            item["eff_hi"] = percentile(drawn, 0.975)
        else:
            item["eff_lo"] = float("nan")
            item["eff_hi"] = float("nan")

    lines = [
        "# Au55 worker scaling",
        "",
        f"Shared communicating ensemble, budget {budget} oracle calls,",
        f"seeds 2000-2011. nproc {nproc}. CPU model {cpu}.",
        f"Revision {revision}.",
        "",
        "Wall time to hit is the median over hits. Calls to hit enter a",
        "miss as budget + 1. Efficiency is T1/(p Tp) from the wall-to-hit",
        "medians. It is undefined when a rung has no hit, and it is omitted",
        "when oversubscribed is true. Intervals are bootstrap 95%",
        "percentiles, 10000 resamples, seed 20261123. A single hit makes",
        "that interval a point. Fisher and Mann-Whitney compare each rung",
        "with 1 worker.",
        "",
        "| workers | hits | hit CI | calls median | calls CI | wall median s | wall CI | efficiency | efficiency CI | Fisher p | MWU p | oversubscribed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in stats:
        if item["workers"] == 1 and not item["over"]:
            eff = "1"
            eff_ci = ""
        elif item["over"]:
            eff = "omitted"
            eff_ci = ""
        elif math.isnan(item["eff"]):
            eff = "undefined"
            eff_ci = ""
        else:
            eff = f"{item['eff']:.4f}"
            eff_ci = f"[{item['eff_lo']:.4f}, {item['eff_hi']:.4f}]"
        wall = "" if math.isnan(item["wall_med"]) else fmt_num(item["wall_med"])
        wall_ci = "" if math.isnan(item["wall_lo"]) else interval(item["wall_lo"], item["wall_hi"])
        lines.append(
            "| {workers} | {hits}/{n} | {hit_ci} | {calls} | {call_ci} | {wall} | {wall_ci} | {eff} | {eff_ci} | {fisher} | {mwu} | {over} |".format(
                workers=item["workers"],
                hits=item["hits"],
                n=item["n"],
                hit_ci=interval(item["hit_lo"], item["hit_hi"]),
                calls=fmt_num(item["call_med"]),
                call_ci=interval(item["call_lo"], item["call_hi"]),
                wall=wall,
                wall_ci=wall_ci,
                eff=eff,
                eff_ci=eff_ci,
                fisher="" if math.isnan(item["fisher"]) else f"{item['fisher']:.4g}",
                mwu="" if math.isnan(item["mwu"]) else f"{item['mwu']:.4g}",
                over="yes" if item["over"] else "no",
            )
        )
    lines.extend(["", "Pins read back from the affinity mask:", ""])
    for item in stats:
        lines.append(
            f"- {item['workers']} workers: cores {item['pinned']}, "
            f"wall-to-hit samples {item['wall_n']}"
        )
    missing = [workers for workers in WORKERS if workers not in by_workers]
    if missing:
        lines.extend(
            [
                "",
                "Rungs not stored: " + ", ".join(str(workers) for workers in missing) + ".",
            ]
        )
    lines.append("")
    text = "\n".join(lines)
    if "\r" in text:
        raise SystemExit("report would contain CR")
    REPORT.write_text(text)
    print(text)


def self_test() -> None:
    if median([1.0, 2.0, 3.0]) != 2.0 or median([1.0, 2.0, 3.0, 4.0]) != 2.5:
        raise SystemExit("median")
    if abs(percentile([1.0, 2.0, 3.0, 4.0], 0.0) - 1.0) > 1e-12:
        raise SystemExit("percentile")
    if fisher_exact(12, 12, 0, 12) <= 0.0:
        raise SystemExit("fisher")
    if not 0.0 < mannwhitney([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) <= 1.0:
        raise SystemExit("mwu ties")
    if mannwhitney([1.0, 2.0, 3.0], [10.0, 11.0, 12.0]) > 0.1:
        raise SystemExit("mwu separation")
    rng = random.Random(SEED)
    lo, hi = bootstrap_stat([0.0, 1.0, 1.0, 1.0], rng, median)
    if not lo <= 1.0 <= hi:
        raise SystemExit("bootstrap")


if __name__ == "__main__":
    self_test()
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("ok")
    else:
        write_report(group(load_rows()))
