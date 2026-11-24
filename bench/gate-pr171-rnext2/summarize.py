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
    "hits_a",
    "hits_a2",
    "hits_b",
    "hits_b2",
    "fisher_p",
    "mwu_p",
    "calls_ratio_ba",
    "calls_ratio_ba_lo",
    "calls_ratio_ba_hi",
    "calls_ratio_a2a",
    "calls_ratio_a2a_lo",
    "calls_ratio_a2a_hi",
    "time_total_ratio_ba",
    "time_total_ratio_ba_lo",
    "time_total_ratio_ba_hi",
    "time_total_ratio_a2a",
    "time_total_ratio_a2a_lo",
    "time_total_ratio_a2a_hi",
    "time_per_call_ratio_ba",
    "time_per_call_ratio_ba_lo",
    "time_per_call_ratio_ba_hi",
    "time_per_call_ratio_a2a",
    "time_per_call_ratio_a2a_lo",
    "time_per_call_ratio_a2a_hi",
    "time_alt_cap",
    "rss_ratio_ba",
    "rss_ratio_ba_lo",
    "rss_ratio_ba_hi",
    "rss_ratio_a2a",
    "rss_ratio_a2a_lo",
    "rss_ratio_a2a_hi",
    "rss_fail_floor",
    "probe_offered_b",
    "probe_hit_b",
    "probe_calls_b",
    "hit_rule",
    "time_rule",
    "rss_rule",
    "path_rule",
    "stop_runs_a",
    "stop_runs_b",
    "best_split_a",
    "best_split_a2",
    "best_split_b",
    "best_split_b2",
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


def mann_whitney_greater(a: list[float], b: list[float]) -> float:
    """Exact conditional permutation probability that B ranks higher."""
    ranks = average_ranks_twice(a + b)
    observed = sum(ranks[len(a) :])
    choose = len(b)
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
            for arm in ("A", "A2", "B", "B2")
            for seed in cell.seeds
        )
        if not complete:
            output.append(
                {field: ("INCONCLUSIVE" if field == "verdict" else "") for field in FIELDS}
                | {"cell": cell.name, "n": "0"}
            )
            continue
        hit = {arm: hits(found, cell, arm) for arm in ("A", "A2", "B", "B2")}
        call = {arm: calls(found, cell, arm) for arm in ("A", "A2", "B")}
        timing = {
            arm: values(found, cell, arm, "outside_s")
            for arm in ("A", "A2", "B")
        }
        timing_per_call = {
            arm: values(found, cell, arm, "outside_per_call_s")
            for arm in ("A", "A2", "B")
        }
        rss = {
            arm: values(found, cell, arm, "rss_anon_peak")
            for arm in ("A", "A2", "B")
        }
        fisher = fisher_two_sided(hit["A"], hit["B"], len(cell.seeds))
        mwu = mann_whitney_greater(call["A"], call["B"])
        call_b = bootstrap_ratio(call["A"], call["B"], rng)
        call_aa = bootstrap_ratio(call["A"], call["A2"], rng)
        time_b = bootstrap_ratio(timing["A"], timing["B"], rng)
        time_aa = bootstrap_ratio(timing["A"], timing["A2"], rng)
        time_call_b = bootstrap_ratio(
            timing_per_call["A"], timing_per_call["B"], rng
        )
        time_call_aa = bootstrap_ratio(
            timing_per_call["A"], timing_per_call["A2"], rng
        )
        rss_b = bootstrap_ratio(rss["A"], rss["B"], rng)
        rss_aa = bootstrap_ratio(rss["A"], rss["A2"], rng)
        offered = sum(
            int(found[(cell.name, "B", seed)]["probe_offered"])
            for seed in cell.seeds
        )
        probe_hit = sum(
            int(found[(cell.name, "B", seed)]["probe_hit"])
            for seed in cell.seeds
        )
        probe_calls = sum(
            int(found[(cell.name, "B", seed)]["probe_calls"])
            for seed in cell.seeds
        )
        hit_ok = not (
            (hit["B"] < hit["A"] and fisher < 0.05)
            or mwu < 0.05
        )
        time_ok = time_b[2] <= gate.TIME_CAP
        time_alt_cap = max(gate.TIME_CAP, time_aa[2])
        rss_floor = max(gate.RSS_CAP, rss_aa[2])
        rss_ok = rss_b[1] <= rss_floor
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
            for arm in ("A", "A2", "B", "B2")
        }
        split = {
            arm: sum(row["best_is_lowest"] != "1" for row in stop[arm])
            for arm in stop
        }
        if cell.name == gate.TARGET_CELL:
            best_ok = split["B"] == 0 and split["B2"] == 0
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
            "hits_a": str(hit["A"]),
            "hits_a2": str(hit["A2"]),
            "hits_b": str(hit["B"]),
            "hits_b2": str(hit["B2"]),
            "fisher_p": fmt(fisher),
            "mwu_p": fmt(mwu),
            "rss_fail_floor": fmt(rss_floor),
            "time_alt_cap": fmt(time_alt_cap),
            "probe_offered_b": str(offered),
            "probe_hit_b": str(probe_hit),
            "probe_calls_b": str(probe_calls),
            "hit_rule": "1" if hit_ok else "0",
            "time_rule": "1" if time_ok else "0",
            "rss_rule": "1" if rss_ok else "0",
            "path_rule": "1" if path_ok else "0",
            "stop_runs_a": str(len(stop["A"])),
            "stop_runs_b": str(len(stop["B"])),
            "best_split_a": str(split["A"]),
            "best_split_a2": str(split["A2"]),
            "best_split_b": str(split["B"]),
            "best_split_b2": str(split["B2"]),
            "best_rule": "1" if best_ok else "0",
            "verdict": verdict,
        }
        for prefix, triple in (
            ("calls_ratio_ba", call_b),
            ("calls_ratio_a2a", call_aa),
            ("time_total_ratio_ba", time_b),
            ("time_total_ratio_a2a", time_aa),
            ("time_per_call_ratio_ba", time_call_b),
            ("time_per_call_ratio_a2a", time_call_aa),
            ("rss_ratio_ba", rss_b),
            ("rss_ratio_a2a", rss_aa),
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
        "# PR 171 restack gate",
        "",
        f"Overall verdict: **{verdict}**.",
        "",
        "| cell | hits A/A2/B/B2 | Fisher p | MWU p | calls B/A | "
        "outside total B/A (95% CI) | outside/call B/A (95% CI) | "
        "outside A2/A upper / alt cap | RSS B/A (95% CI) / floor | "
        "probe offered/hit/calls | best splits A/B | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["verdict"] == "INCONCLUSIVE":
            lines.append(
                f"| {row['cell']} | - | - | - | - | - | - | - | INCONCLUSIVE |"
            )
            continue
        lines.append(
            f"| {row['cell']} | {row['hits_a']}/{row['hits_a2']}/"
            f"{row['hits_b']}/{row['hits_b2']} "
            f"| {row['fisher_p']} | {row['mwu_p']} "
            f"| {row['calls_ratio_ba']} "
            f"| {row['time_total_ratio_ba']} "
            f"({row['time_total_ratio_ba_lo']}, "
            f"{row['time_total_ratio_ba_hi']}) "
            f"| {row['time_per_call_ratio_ba']} "
            f"({row['time_per_call_ratio_ba_lo']}, "
            f"{row['time_per_call_ratio_ba_hi']}) "
            f"| {row['time_total_ratio_a2a_hi']} / {row['time_alt_cap']} "
            f"| {row['rss_ratio_ba']} "
            f"({row['rss_ratio_ba_lo']}, {row['rss_ratio_ba_hi']}) / "
            f"{row['rss_fail_floor']} "
            f"| {row['probe_offered_b']}/{row['probe_hit_b']}/"
            f"{row['probe_calls_b']} "
            f"| {row['best_split_a']}/{row['best_split_b']} "
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
