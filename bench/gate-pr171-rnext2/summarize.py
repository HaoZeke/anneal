#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Apply the preregistered PR 171 r-next2 gate."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as gate

FIELDS = [
    "cell",
    "n",
    "hits_d",
    "hits_d2",
    "hits_g",
    "hits_g2",
    "fisher_p",
    "mwu_p",
    "calls_ratio_gd",
    "calls_ratio_gd_lo",
    "calls_ratio_gd_hi",
    "calls_ratio_d2d",
    "calls_ratio_d2d_lo",
    "calls_ratio_d2d_hi",
    "time_ratio_gd",
    "time_ratio_gd_lo",
    "time_ratio_gd_hi",
    "time_ratio_d2d",
    "time_ratio_d2d_lo",
    "time_ratio_d2d_hi",
    "rss_ratio_gd",
    "rss_ratio_gd_lo",
    "rss_ratio_gd_hi",
    "rss_ratio_d2d",
    "rss_ratio_d2d_lo",
    "rss_ratio_d2d_hi",
    "rss_fail_floor",
    "probe_offered_g",
    "probe_hit_g",
    "probe_calls_g",
    "hit_rule",
    "time_rule",
    "rss_rule",
    "path_rule",
    "stop_runs_d",
    "stop_runs_g",
    "best_split_d",
    "best_split_d2",
    "best_split_g",
    "best_split_g2",
    "best_rule",
    "verdict",
]


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    if lower + 1 == len(ordered):
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[lower + 1] * fraction


def fisher_two_sided(d_hit: int, g_hit: int, n: int) -> float:
    total_hits = d_hit + g_hit
    low = max(0, total_hits - n)
    high = min(n, total_hits)

    def mass(value: int) -> float:
        return (
            math.comb(n, value)
            * math.comb(n, total_hits - value)
            / math.comb(2 * n, total_hits)
        )

    observed = mass(d_hit)
    return min(
        1.0,
        sum(mass(value) for value in range(low, high + 1) if mass(value) <= observed + 1e-15),
    )


def average_ranks_twice(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        rank_twice = start + end + 2
        for index in order[start : end + 1]:
            ranks[index] = rank_twice
        start = end + 1
    return ranks


def mann_whitney_greater(d: list[float], g: list[float]) -> float:
    """Exact conditional permutation probability that G ranks higher."""
    ranks = average_ranks_twice(d + g)
    observed = sum(ranks[len(d) :])
    choose = len(g)
    ways: dict[tuple[int, int], int] = {(0, 0): 1}
    for rank in ranks:
        updated = dict(ways)
        for (count, total), number in ways.items():
            if count < choose:
                key = (count + 1, total + rank)
                updated[key] = updated.get(key, 0) + number
        ways = updated
    extreme = sum(
        number
        for (count, total), number in ways.items()
        if count == choose and total >= observed
    )
    return extreme / math.comb(len(ranks), choose)


def geometric_ratio(left: list[float], right: list[float]) -> float:
    logs = [math.log(b / a) for a, b in zip(left, right)]
    return math.exp(sum(logs) / len(logs))


def bootstrap_ratio(
    left: list[float], right: list[float], rng: random.Random
) -> tuple[float, float, float]:
    point = geometric_ratio(left, right)
    count = len(left)
    draws = []
    for _ in range(gate.BOOT_DRAWS):
        selected = [rng.randrange(count) for _ in range(count)]
        draws.append(
            geometric_ratio(
                [left[index] for index in selected],
                [right[index] for index in selected],
            )
        )
    return point, type7(draws, 0.025), type7(draws, 0.975)


def fmt(value: float) -> str:
    return f"{value:.6g}"


def load(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    found = {}
    for row in rows:
        key = (row["cell"], row["arm"], int(row["seed"]))
        if key in found:
            raise SystemExit(f"duplicate row {key}")
        found[key] = row
    return found


def values(found, cell: gate.Cell, arm: str, field: str) -> list[float]:
    result = []
    for seed in cell.seeds:
        row = found[(cell.name, arm, seed)]
        if int(row["exit_code"]) != 0:
            raise ValueError(f"{cell.name} {arm} {seed} is inconclusive")
        result.append(float(row[field]))
    return result


def calls(found, cell: gate.Cell, arm: str) -> list[float]:
    result = []
    for seed in cell.seeds:
        row = found[(cell.name, arm, seed)]
        if int(row["exit_code"]) != 0:
            raise ValueError(f"{cell.name} {arm} {seed} is inconclusive")
        result.append(
            float(row["calls"]) if row["hit"] == "1" else float(cell.budget + 1)
        )
    return result


def hits(found, cell: gate.Cell, arm: str) -> int:
    return sum(found[(cell.name, arm, seed)]["hit"] == "1" for seed in cell.seeds)


def evaluate(found) -> tuple[list[dict[str, str]], str]:
    rng = random.Random(gate.BOOT_SEED)
    output = []
    for cell in gate.CELLS:
        complete = all(
            (cell.name, arm, seed) in found
            and found[(cell.name, arm, seed)]["exit_code"] == "0"
            for arm in ("D", "D2", "G", "G2")
            for seed in cell.seeds
        )
        if not complete:
            output.append(
                {field: ("INCONCLUSIVE" if field == "verdict" else "") for field in FIELDS}
                | {"cell": cell.name, "n": "0"}
            )
            continue
        hit = {arm: hits(found, cell, arm) for arm in ("D", "D2", "G", "G2")}
        call = {arm: calls(found, cell, arm) for arm in ("D", "D2", "G")}
        timing = {
            arm: values(found, cell, arm, "outside_s")
            for arm in ("D", "D2", "G")
        }
        rss = {
            arm: values(found, cell, arm, "rss_anon_peak")
            for arm in ("D", "D2", "G")
        }
        fisher = fisher_two_sided(hit["D"], hit["G"], len(cell.seeds))
        mwu = mann_whitney_greater(call["D"], call["G"])
        call_g = bootstrap_ratio(call["D"], call["G"], rng)
        call_aa = bootstrap_ratio(call["D"], call["D2"], rng)
        time_g = bootstrap_ratio(timing["D"], timing["G"], rng)
        time_aa = bootstrap_ratio(timing["D"], timing["D2"], rng)
        rss_g = bootstrap_ratio(rss["D"], rss["G"], rng)
        rss_aa = bootstrap_ratio(rss["D"], rss["D2"], rng)
        offered = sum(
            int(found[(cell.name, "G", seed)]["probe_offered"])
            for seed in cell.seeds
        )
        probe_hit = sum(
            int(found[(cell.name, "G", seed)]["probe_hit"])
            for seed in cell.seeds
        )
        probe_calls = sum(
            int(found[(cell.name, "G", seed)]["probe_calls"])
            for seed in cell.seeds
        )
        hit_ok = not (
            (hit["G"] < hit["D"] and fisher < 0.05)
            or mwu < 0.05
        )
        time_ok = time_g[2] <= gate.TIME_CAP
        rss_floor = max(gate.RSS_CAP, rss_aa[2])
        rss_ok = rss_g[1] <= rss_floor
        if cell.name == gate.TARGET_CELL:
            path_ok = offered > 0 and probe_calls > 0
        elif cell.name == gate.MOLECULAR_CELL:
            path_ok = offered == 0
        else:
            path_ok = True
        stop = {
            arm: [
                found[(cell.name, arm, seed)]
                for seed in cell.seeds
                if found[(cell.name, arm, seed)]["stop_run"] == "1"
            ]
            for arm in ("D", "D2", "G", "G2")
        }
        split = {
            arm: sum(row["best_is_lowest"] != "1" for row in stop[arm])
            for arm in stop
        }
        if cell.name == gate.TARGET_CELL:
            best_ok = split["G"] == 0 and split["G2"] == 0
        else:
            best_ok = True
        verdict = (
            "PASS"
            if hit_ok and time_ok and rss_ok and path_ok and best_ok
            else "FAIL"
        )
        row = {
            "cell": cell.name,
            "n": str(len(cell.seeds)),
            "hits_d": str(hit["D"]),
            "hits_d2": str(hit["D2"]),
            "hits_g": str(hit["G"]),
            "hits_g2": str(hit["G2"]),
            "fisher_p": fmt(fisher),
            "mwu_p": fmt(mwu),
            "rss_fail_floor": fmt(rss_floor),
            "probe_offered_g": str(offered),
            "probe_hit_g": str(probe_hit),
            "probe_calls_g": str(probe_calls),
            "hit_rule": "1" if hit_ok else "0",
            "time_rule": "1" if time_ok else "0",
            "rss_rule": "1" if rss_ok else "0",
            "path_rule": "1" if path_ok else "0",
            "stop_runs_d": str(len(stop["D"])),
            "stop_runs_g": str(len(stop["G"])),
            "best_split_d": str(split["D"]),
            "best_split_d2": str(split["D2"]),
            "best_split_g": str(split["G"]),
            "best_split_g2": str(split["G2"]),
            "best_rule": "1" if best_ok else "0",
            "verdict": verdict,
        }
        for prefix, triple in (
            ("calls_ratio_gd", call_g),
            ("calls_ratio_d2d", call_aa),
            ("time_ratio_gd", time_g),
            ("time_ratio_d2d", time_aa),
            ("rss_ratio_gd", rss_g),
            ("rss_ratio_d2d", rss_aa),
        ):
            row[prefix] = fmt(triple[0])
            row[prefix + "_lo"] = fmt(triple[1])
            row[prefix + "_hi"] = fmt(triple[2])
        output.append(row)
    overall = (
        "PASS"
        if output and all(row["verdict"] == "PASS" for row in output)
        else "INCONCLUSIVE"
        if any(row["verdict"] == "INCONCLUSIVE" for row in output)
        else "FAIL"
    )
    return output, overall


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, rows: list[dict[str, str]], verdict: str) -> None:
    lines = [
        "# PR 171 r-next2 gate",
        "",
        f"Overall verdict: **{verdict}**.",
        "",
        "| cell | hits D/D2/G | Fisher p | MWU p | time G/D (95% CI) | "
        "RSS G/D (95% CI) | probe offered/hit/calls | best splits D/G | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["verdict"] == "INCONCLUSIVE":
            lines.append(
                f"| {row['cell']} | - | - | - | - | - | - | - | INCONCLUSIVE |"
            )
            continue
        lines.append(
            f"| {row['cell']} | {row['hits_d']}/{row['hits_d2']}/{row['hits_g']} "
            f"| {row['fisher_p']} | {row['mwu_p']} "
            f"| {row['time_ratio_gd']} "
            f"({row['time_ratio_gd_lo']}, {row['time_ratio_gd_hi']}) "
            f"| {row['rss_ratio_gd']} "
            f"({row['rss_ratio_gd_lo']}, {row['rss_ratio_gd_hi']}) "
            f"| {row['probe_offered_g']}/{row['probe_hit_g']}/"
            f"{row['probe_calls_g']} "
            f"| {row['best_split_d']}/{row['best_split_g']} "
            f"| {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "The protocol, arm definitions, thresholds, and statistical rules are",
            "in `PROTOCOL.md`. Raw rows are in `rows.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", newline="\n")


def self_test() -> None:
    assert abs(fisher_two_sided(12, 6, 12) - 0.0137307) < 1e-6
    assert mann_whitney_greater([1, 2], [3, 4]) < 0.2
    rng = random.Random(1)
    point, low, high = bootstrap_ratio([1, 1], [2, 2], rng)
    assert point == low == high == 2.0
    print("summarize ok")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()
    rows, verdict = evaluate(load(args.rows))
    write_csv(args.summary, rows)
    write_readme(args.readme, rows, verdict)
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
