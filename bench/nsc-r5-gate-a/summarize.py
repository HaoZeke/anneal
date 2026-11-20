#!/usr/bin/env python3
"""Summarize the no-size-constants r5 A/B gate."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ALPHA = 0.05
SUMMARY_FIELDS = (
    "panel",
    "system",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_exact_p",
    "median_calls_a",
    "median_calls_b",
    "mann_whitney_u",
    "mann_whitney_p",
    "median_wall_s_a",
    "median_wall_s_b",
    "median_pot_s_a",
    "median_pot_s_b",
    "median_outside_s_a",
    "median_outside_s_b",
    "median_outside_per_call_s_a",
    "median_outside_per_call_s_b",
    "outside_per_call_ratio_b_over_a",
    "hit_loss",
    "call_loss",
    "outside_per_call_loss",
    "verdict",
)
PANEL_FIELDS = (
    "panel",
    "systems",
    "exact_cmh_p",
    "van_elteren_z",
    "van_elteren_p",
)
CHANGE_FIELDS = ("panel", "system", "seed", "hit_a", "hit_b", "direction")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.8g}"


def fisher_exact(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    total = hits_a + hits_b
    denominator = math.comb(n_a + n_b, total)

    def probability(a_hits: int) -> float:
        b_hits = total - a_hits
        if not 0 <= a_hits <= n_a or not 0 <= b_hits <= n_b:
            return 0.0
        return math.comb(n_a, a_hits) * math.comb(n_b, b_hits) / denominator

    observed = probability(hits_a)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(max(0, total - n_b), min(n_a, total) + 1)
            if probability(value) <= observed * (1.0 + 1e-12)
        ),
    )


def ranked(values: list[float]) -> tuple[list[float], float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    tie_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        size = end - start
        tie_sum += size**3 - size
        start = end
    return ranks, tie_sum


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n_left, n_right = len(left), len(right)
    values = left + right
    ranks, tie_sum = ranked(values)
    u = sum(ranks[:n_left]) - n_left * (n_left + 1) / 2.0
    count = len(values)
    variance = n_left * n_right / 12.0 * (
        count + 1 - tie_sum / (count * (count - 1))
    )
    if variance <= 0:
        return u, 1.0
    deviation = max(abs(u - n_left * n_right / 2.0) - 0.5, 0.0)
    return u, math.erfc(deviation / math.sqrt(2.0 * variance))


def exact_cmh(strata: list[tuple[int, int, int, int]]) -> float:
    distribution = {0: 1.0}
    observed = 0
    expectation = 0.0
    for hits_a, n_a, hits_b, n_b in strata:
        total = hits_a + hits_b
        observed += hits_b
        expectation += n_b * total / (n_a + n_b)
        denominator = math.comb(n_a + n_b, total)
        mass = {
            b_hits: math.comb(n_b, b_hits) * math.comb(n_a, total - b_hits) / denominator
            for b_hits in range(max(0, total - n_a), min(n_b, total) + 1)
        }
        combined: dict[int, float] = defaultdict(float)
        for prior_total, prior_probability in distribution.items():
            for b_hits, probability in mass.items():
                combined[prior_total + b_hits] += prior_probability * probability
        distribution = dict(combined)
    distance = abs(observed - expectation)
    return min(
        1.0,
        sum(probability for value, probability in distribution.items() if abs(value - expectation) >= distance - 1e-12),
    )


def van_elteren(strata: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    score = 0.0
    variance = 0.0
    for a_values, b_values in strata:
        values = b_values + a_values
        ranks, tie_sum = ranked(values)
        n_b, n_a = len(b_values), len(a_values)
        count = n_a + n_b
        u = sum(ranks[:n_b]) - n_b * (n_b + 1) / 2.0
        mean = n_a * n_b / 2.0
        var = n_a * n_b / 12.0 * (
            count + 1 - tie_sum / (count * (count - 1))
        )
        weight = 1.0 / (count + 1.0)
        score += weight * (u - mean)
        variance += weight * weight * var
    if variance <= 0:
        return 0.0, 1.0
    z_value = score / math.sqrt(variance)
    return z_value, math.erfc(abs(z_value) / math.sqrt(2.0))


def grouped_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[dict[str, str]]]]:
    groups: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"A": [], "B": []}
    )
    for row in rows:
        groups[(row["panel"], row["system"])][row["arm"]].append(row)
    for arms in groups.values():
        for arm in arms:
            arms[arm].sort(key=lambda row: int(row["seed"]))
    return dict(groups)


def verify(rows: list[dict[str, str]]) -> None:
    if len(rows) != 288:
        raise RuntimeError(f"expected 288 rows, found {len(rows)}")
    groups = grouped_rows(rows)
    if len(groups) != 12:
        raise RuntimeError(f"expected 12 cells, found {len(groups)}")
    for key, arms in groups.items():
        for arm in ("A", "B"):
            seeds = [int(row["seed"]) for row in arms[arm]]
            if seeds != list(range(2000, 2012)):
                raise RuntimeError(f"{key} arm {arm}: seeds {seeds}")
        counts = {
            (slot, arm): sum(
                int(row["slot"]) == slot and row["arm"] == arm
                for row in arms["A"] + arms["B"]
            )
            for slot in (1, 2)
            for arm in ("A", "B")
        }
        if set(counts.values()) != {6}:
            raise RuntimeError(f"{key}: unbalanced schedule {counts}")


def median(rows: list[dict[str, str]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def generate_readme(
    path: Path,
    summary: list[dict[str, str]],
    panels: list[dict[str, str]],
    changes: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> None:
    losing = [row["system"] for row in summary if row["verdict"] == "FIX"]
    verdict = "FIX" if losing else "PASS"
    first = rows[0]
    arm_data = {
        arm: next(row for row in rows if row["arm"] == arm)
        for arm in ("A", "B")
    }
    lines = [
        "# No-size-constants r5 gate A",
        "",
        f"Verdict: **{verdict}**" + (f" ({', '.join(losing)})" if losing else ""),
        "",
        "The rule was fixed before execution: FIX if any cell has fewer B hits, "
        "if B loses on calls at two-sided p < 0.05, or if the B/A ratio of "
        "median outside-potential time per charged call exceeds 1.10. Otherwise PASS.",
        "",
        "The campaign used seeds 2000–2011 at the paper budgets in one serial "
        "session. Adjacent seeds used AB then BA order, giving six observations "
        "for every (slot, arm) pair in every cell. No budget override was set.",
        "",
        "## Provenance",
        "",
        "| item | value |",
        "|---|---|",
        f"| arm A revision | `{arm_data['A']['code_revision']}` |",
        f"| arm A tree | `{arm_data['A']['code_tree']}` |",
        f"| arm B revision | `{arm_data['B']['code_revision']}` |",
        f"| arm B tree | `{arm_data['B']['code_tree']}` |",
        f"| runner blob | `{first['runner_blob']}` |",
        f"| summarizer blob | `{first['summarizer_blob']}` |",
        f"| machine identity hash | `{first['machine_id_sha256']}` |",
        "",
        "Binary hashes are recorded on every row in `records.csv`.",
        "",
        "## Cell results",
        "",
        "| panel | cell | hits A/B | Fisher p | calls A/B | MW p | wall A/B (s) | "
        "potential A/B (s) | outside A/B (s) | outside/call B/A | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['panel']} | {row['system']} | {row['hits_a']}/{row['hits_b']} | "
            f"{row['fisher_exact_p']} | {row['median_calls_a']}/{row['median_calls_b']} | "
            f"{row['mann_whitney_p']} | {row['median_wall_s_a']}/{row['median_wall_s_b']} | "
            f"{row['median_pot_s_a']}/{row['median_pot_s_b']} | "
            f"{row['median_outside_s_a']}/{row['median_outside_s_b']} | "
            f"{row['outside_per_call_ratio_b_over_a']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Calls use misses at the paper budget. Mann–Whitney is tie-corrected "
            "with continuity correction. Times and calls are medians. "
            "Outside-per-call uses charged calls.",
            "",
            "## Multi-system panel tests",
            "",
            "| panel | cells | exact CMH p | van Elteren z | van Elteren p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in panels:
        lines.append(
            f"| {row['panel']} | {row['systems']} | {row['exact_cmh_p']} | "
            f"{row['van_elteren_z']} | {row['van_elteren_p']} |"
        )
    lines.extend(["", "## Hit-status changes", ""])
    if changes:
        lines.extend(["| panel | cell | seed | direction |", "|---|---|---:|---|"])
        for row in changes:
            lines.append(
                f"| {row['panel']} | {row['system']} | {row['seed']} | {row['direction']} |"
            )
    else:
        lines.append("None.")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--changes", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = read_csv(args.rows)
    verify(rows)
    groups = grouped_rows(rows)
    summary_rows: list[dict[str, str]] = []
    change_rows: list[dict[str, str]] = []
    for (panel, system), arms in sorted(groups.items()):
        a_rows, b_rows = arms["A"], arms["B"]
        hits_a = sum(row["hit"] == "1" for row in a_rows)
        hits_b = sum(row["hit"] == "1" for row in b_rows)
        calls_a = [float(row["evals_to_hit"]) for row in a_rows]
        calls_b = [float(row["evals_to_hit"]) for row in b_rows]
        u_value, u_p = mann_whitney(calls_b, calls_a)
        ratio = median(b_rows, "outside_per_call_s") / median(a_rows, "outside_per_call_s")
        hit_loss = hits_b < hits_a
        call_loss = statistics.median(calls_b) > statistics.median(calls_a) and u_p < ALPHA
        outside_loss = ratio > 1.10
        fix = hit_loss or call_loss or outside_loss
        summary_rows.append(
            {
                "panel": panel,
                "system": system,
                "n_a": str(len(a_rows)),
                "n_b": str(len(b_rows)),
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "fisher_exact_p": fmt(fisher_exact(hits_a, len(a_rows), hits_b, len(b_rows))),
                "median_calls_a": fmt(statistics.median(calls_a)),
                "median_calls_b": fmt(statistics.median(calls_b)),
                "mann_whitney_u": fmt(u_value),
                "mann_whitney_p": fmt(u_p),
                "median_wall_s_a": fmt(median(a_rows, "wall_s")),
                "median_wall_s_b": fmt(median(b_rows, "wall_s")),
                "median_pot_s_a": fmt(median(a_rows, "pot_s")),
                "median_pot_s_b": fmt(median(b_rows, "pot_s")),
                "median_outside_s_a": fmt(median(a_rows, "outside_s")),
                "median_outside_s_b": fmt(median(b_rows, "outside_s")),
                "median_outside_per_call_s_a": fmt(median(a_rows, "outside_per_call_s")),
                "median_outside_per_call_s_b": fmt(median(b_rows, "outside_per_call_s")),
                "outside_per_call_ratio_b_over_a": fmt(ratio),
                "hit_loss": str(int(hit_loss)),
                "call_loss": str(int(call_loss)),
                "outside_per_call_loss": str(int(outside_loss)),
                "verdict": "FIX" if fix else "PASS",
            }
        )
        by_seed = {
            arm: {int(row["seed"]): row for row in arms[arm]}
            for arm in ("A", "B")
        }
        for seed in range(2000, 2012):
            a_hit = by_seed["A"][seed]["hit"]
            b_hit = by_seed["B"][seed]["hit"]
            if a_hit != b_hit:
                change_rows.append(
                    {
                        "panel": panel,
                        "system": system,
                        "seed": str(seed),
                        "hit_a": a_hit,
                        "hit_b": b_hit,
                        "direction": "miss-to-hit" if b_hit == "1" else "hit-to-miss",
                    }
                )

    panel_rows: list[dict[str, str]] = []
    for panel in sorted({key[0] for key in groups}):
        panel_groups = [
            (system, groups[(group_panel, system)])
            for group_panel, system in sorted(groups)
            if group_panel == panel
        ]
        if len(panel_groups) < 2:
            continue
        hit_strata = []
        call_strata = []
        for _system, arms in panel_groups:
            a_rows, b_rows = arms["A"], arms["B"]
            hit_strata.append(
                (
                    sum(row["hit"] == "1" for row in a_rows),
                    len(a_rows),
                    sum(row["hit"] == "1" for row in b_rows),
                    len(b_rows),
                )
            )
            call_strata.append(
                (
                    [float(row["evals_to_hit"]) for row in a_rows],
                    [float(row["evals_to_hit"]) for row in b_rows],
                )
            )
        z_value, van_p = van_elteren(call_strata)
        panel_rows.append(
            {
                "panel": panel,
                "systems": str(len(panel_groups)),
                "exact_cmh_p": fmt(exact_cmh(hit_strata)),
                "van_elteren_z": fmt(z_value),
                "van_elteren_p": fmt(van_p),
            }
        )

    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.panels, PANEL_FIELDS, panel_rows)
    write_csv(args.changes, CHANGE_FIELDS, change_rows)
    generate_readme(args.readme, summary_rows, panel_rows, change_rows, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
