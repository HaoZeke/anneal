#!/usr/bin/env python3
"""Score the r-next2 gate from the per-cell row files."""

from __future__ import annotations

import csv
import math
import random
import statistics
import struct
from pathlib import Path

import run_gate as gate

HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 21102
TIME_LIMIT = 1.10
IDENTITY = ("energy_hash", "best_bits", "hops", "basins", "visits", "unconverged")
SUMMARY_FIELDS = (
    "cell",
    "n",
    "identical_b",
    "identical_a2",
    "unconverged_a",
    "unconverged_a2",
    "unconverged_b",
    "hits_a",
    "hits_a2",
    "hits_b",
    "hit_loss",
    "fisher_a_greater_p",
    "mannwhitney_b_slower_p",
    "outside_median_b_a",
    "outside_b_a_ci_low",
    "outside_b_a_ci_high",
    "outside_median_a2_a",
    "outside_a2_a_ci_low",
    "outside_a2_a_ci_high",
    "time_limit",
    "adjusted_time_limit",
    "time_clause",
    "adjusted_time_clause",
    "verdict",
    "first_diff",
)


def type7(values: list[float], probability: float = 0.5) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median(
    values: list[float],
    generator: random.Random,
) -> tuple[float, float, float]:
    estimate = type7(values)
    size = len(values)
    samples = [
        type7([values[generator.randrange(size)] for _ in range(size)])
        for _ in range(BOOTSTRAPS)
    ]
    return estimate, type7(samples, 0.025), type7(samples, 0.975)


def fisher_a_greater(hits_a: int, hits_b: int, n: int) -> float:
    total_hits = hits_a + hits_b
    ways = math.comb(2 * n, total_hits)

    def probability(a_hits: int) -> float:
        b_hits = total_hits - a_hits
        if not 0 <= b_hits <= n:
            return 0.0
        return math.comb(n, a_hits) * math.comb(n, b_hits) / ways

    return min(1.0, sum(probability(value) for value in range(hits_a, n + 1)))


def mannwhitney_b_slower(a: list[float], b: list[float]) -> float:
    pooled = sorted(a + b)
    ranks: dict[float, float] = {}
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1] == pooled[index]:
            end += 1
        ranks[pooled[index]] = (index + end + 2) / 2
        width = end - index + 1
        tie_term += width**3 - width
        index = end + 1
    n_a, n_b = len(a), len(b)
    u_b = sum(ranks[value] for value in b) - n_b * (n_b + 1) / 2
    mean = n_a * n_b / 2
    count = n_a + n_b
    variance = n_a * n_b / 12 * (count + 1 - tie_term / (count * (count - 1)))
    if variance <= 0:
        return 1.0
    z_score = (u_b - mean - 0.5) / math.sqrt(variance)
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def fmt(value: float) -> str:
    return f"{value:.8g}"


def matches(left: dict[str, str], right: dict[str, str]) -> bool:
    return all(left[field] == right[field] for field in IDENTITY)


def bits_to_float(word: str) -> float:
    return struct.unpack(">d", bytes.fromhex(word))[0]


def first_energy_difference(left: str, right: str, other: str) -> str:
    left_words = left.split()
    right_words = right.split()
    limit = min(len(left_words), len(right_words))
    for index in range(limit):
        if left_words[index] != right_words[index]:
            return (
                f"record {index} A={left_words[index]} "
                f"({bits_to_float(left_words[index]):.16e}) "
                f"{other}={right_words[index]} "
                f"({bits_to_float(right_words[index]):.16e})"
            )
    if len(left_words) != len(right_words):
        return f"record length A={len(left_words)} {other}={len(right_words)}"
    return "energies match"


def describe_difference(a: dict[str, str], other: dict[str, str], label: str) -> str:
    energy = first_energy_difference(a["energies"], other["energies"], label)
    fields = [field for field in IDENTITY if a[field] != other[field]]
    a_unconv = int(a["unconverged"])
    other_unconv = int(other["unconverged"])
    if other_unconv > a_unconv:
        refusal = (
            f"refusal counter rose by {other_unconv - a_unconv} "
            f"(A {a_unconv}, {label} {other_unconv})"
        )
    elif other_unconv == a_unconv:
        refusal = f"refusal counter unchanged at {a_unconv}"
    else:
        refusal = f"refusal counter fell from {a_unconv} to {other_unconv}"
    field_text = ",".join(fields) if fields else "none"
    return f"fields {field_text}; {energy}; {refusal}"


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cell in gate.CELLS:
        path = HERE / f"rows-{cell['name']}.csv"
        with path.open(newline="") as handle:
            loaded = list(csv.DictReader(handle))
        if len(loaded) != len(gate.SEEDS) * 4:
            raise RuntimeError(f"{cell['name']} has {len(loaded)} rows")
        rows.extend(loaded)
    return rows


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    generator = random.Random(BOOTSTRAP_SEED)
    by_cell: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_cell.setdefault(row["cell"], []).append(row)
    for cell in gate.CELLS:
        name = str(cell["name"])
        budget = int(cell["budget"])
        grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in by_cell[name]:
            grouped.setdefault((row["arm"], int(row["seed"])), []).append(row)
        ratios_ba: list[float] = []
        ratios_a2a: list[float] = []
        scores = {"A": [], "A2": [], "B": []}
        hits = {"A": 0, "A2": 0, "B": 0}
        identical_b = 0
        identical_a2 = 0
        unconverged = {"A": 0, "A2": 0, "B": 0}
        hit_loss = 0
        hits_agree = True
        calls_agree = True
        first_diff = ""
        for seed in gate.SEEDS:
            a_rows = grouped[("A", seed)]
            a2_rows = grouped[("A2", seed)]
            b_rows = grouped[("B", seed)]
            if len(a_rows) != 1 or len(a2_rows) != 1 or len(b_rows) != 2:
                raise RuntimeError(f"incomplete rows for {name} seed {seed}")
            a, a2 = a_rows[0], a2_rows[0]
            b_rows = sorted(b_rows, key=lambda row: int(row["position"]))
            if matches(a, a2):
                identical_a2 += 1
            elif not first_diff:
                first_diff = f"seed {seed} A2 control: " + describe_difference(a, a2, "A2")
            both_b = all(matches(a, row) for row in b_rows)
            if both_b:
                identical_b += 1
            else:
                differing = next(row for row in b_rows if not matches(a, row))
                b_text = (
                    f"seed {seed} position {differing['position']} "
                    f"repeat {differing['arm_repeat']}: "
                    + describe_difference(a, differing, "B")
                )
                if first_diff.startswith("seed") and " A2 control:" in first_diff:
                    first_diff = b_text
                elif not first_diff:
                    first_diff = b_text
            for row in (a, a2, *b_rows):
                unconverged[row["arm"]] += int(row["unconverged"])
            b_hits = [int(row["hit"]) for row in b_rows]
            a_hit = int(a["hit"])
            a2_hit = int(a2["hit"])
            hits["A"] += a_hit
            hits["A2"] += a2_hit
            agreed_b = b_hits[0] if b_hits[0] == b_hits[1] else 0
            hits["B"] += agreed_b
            if a_hit and (b_hits[0] == 0 or b_hits[1] == 0):
                hit_loss += 1
            if b_hits[0] != b_hits[1]:
                hits_agree = False
                if not first_diff:
                    first_diff = f"seed {seed} B hit disagreement {b_hits[0]} vs {b_hits[1]}"
            b_calls = [int(row["calls"]) for row in b_rows]
            if len(set(b_calls)) != 1:
                calls_agree = False
                if not first_diff:
                    first_diff = f"seed {seed} B call disagreement"
            outside_b = statistics.fmean(float(row["outside_s"]) for row in b_rows)
            outside_a = float(a["outside_s"])
            outside_a2 = float(a2["outside_s"])
            if min(outside_a, outside_a2, outside_b) <= 0:
                raise RuntimeError(f"non-positive outside time for {name} seed {seed}")
            ratios_ba.append(outside_b / outside_a)
            ratios_a2a.append(outside_a2 / outside_a)
            for arm, selected, hit in (
                ("A", a, a_hit),
                ("A2", a2, a2_hit),
                ("B", b_rows[0], agreed_b),
            ):
                calls = float(selected["calls"])
                scores[arm].append(calls if hit else float(budget + 1))
        median_ba, low_ba, high_ba = bootstrap_median(ratios_ba, generator)
        median_a2, low_a2, high_a2 = bootstrap_median(ratios_a2a, generator)
        adjusted = max(TIME_LIMIT, high_a2)
        time_pass = high_ba <= TIME_LIMIT
        adjusted_pass = high_ba <= adjusted
        identity_pass = identical_b == len(gate.SEEDS) and identical_a2 == len(gate.SEEDS)
        loss_pass = hit_loss == 0 and hits_agree and calls_agree
        passed = identity_pass and loss_pass and time_pass
        summaries.append(
            {
                "cell": name,
                "n": str(len(gate.SEEDS)),
                "identical_b": str(identical_b),
                "identical_a2": str(identical_a2),
                "unconverged_a": str(unconverged["A"]),
                "unconverged_a2": str(unconverged["A2"]),
                "unconverged_b": str(unconverged["B"]),
                "hits_a": str(hits["A"]),
                "hits_a2": str(hits["A2"]),
                "hits_b": str(hits["B"]),
                "hit_loss": str(hit_loss),
                "fisher_a_greater_p": fmt(fisher_a_greater(hits["A"], hits["B"], len(gate.SEEDS))),
                "mannwhitney_b_slower_p": fmt(mannwhitney_b_slower(scores["A"], scores["B"])),
                "outside_median_b_a": fmt(median_ba),
                "outside_b_a_ci_low": fmt(low_ba),
                "outside_b_a_ci_high": fmt(high_ba),
                "outside_median_a2_a": fmt(median_a2),
                "outside_a2_a_ci_low": fmt(low_a2),
                "outside_a2_a_ci_high": fmt(high_a2),
                "time_limit": fmt(TIME_LIMIT),
                "adjusted_time_limit": fmt(adjusted),
                "time_clause": "PASS" if time_pass else "FAIL",
                "adjusted_time_clause": "PASS" if adjusted_pass else "FAIL",
                "verdict": "PASS" if passed else "FAIL",
                "first_diff": first_diff,
            }
        )
        del b_deterministic
    return summaries


def write_report(summaries: list[dict[str, str]]) -> None:
    overall = "PASS" if all(row["verdict"] == "PASS" for row in summaries) else "FAIL"
    lines = [
        "# PR 211 r-next2 gate",
        "",
        f"Overall verdict: {overall}.",
        "",
        "| Cell | B ident | A2 ident | Hits A/A2/B | Loss | Fisher | MWU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['identical_b']}/12 | {row['identical_a2']}/12 | "
            f"{row['hits_a']}/{row['hits_a2']}/{row['hits_b']} | {row['hit_loss']} | "
            f"{row['fisher_a_greater_p']} | {row['mannwhitney_b_slower_p']} |"
        )
    lines += [
        "",
        "| Cell | B/A median [CI] | A2/A median [CI] | Adjusted limit | Time | Verdict |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['cell']} | {row['outside_median_b_a']} "
            f"[{row['outside_b_a_ci_low']}, {row['outside_b_a_ci_high']}] | "
            f"{row['outside_median_a2_a']} "
            f"[{row['outside_a2_a_ci_low']}, {row['outside_a2_a_ci_high']}] | "
            f"{row['adjusted_time_limit']} | {row['time_clause']} | {row['verdict']} |"
        )
    lines += [
        "",
        "Unconverged sums are A, A2, and both B repeats.",
        "",
    ]
    for row in summaries:
        lines.append(
            f"- {row['cell']}: unconverged {row['unconverged_a']}, "
            f"{row['unconverged_a2']}, {row['unconverged_b']}. "
            f"{row['first_diff'] or 'No differing row.'}"
        )
    lines.append("")
    (HERE / "SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    rows = load_rows()
    summaries = aggregate(rows)
    with (HERE / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    write_report(summaries)


if __name__ == "__main__":
    main()
