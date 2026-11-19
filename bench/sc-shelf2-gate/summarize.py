#!/usr/bin/env python3
"""Validate, summarize and decide the three-arm shelf gate."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

from run import FIELDS, SEEDS, SYSTEMS


EXPECTED_C_TREE = "b17735cfa8c81727c596d6e20a696b5b9678fb50"
ARMS = ("A", "B", "C")
SUMMARY_FIELDS = (
    "system",
    "arm",
    "hits",
    "n",
    "hits_over_n",
    "median_calls",
    "median_wall_s",
    "median_pot_s",
    "median_outside_s",
    "code_revision",
    "host",
)
COMPARISON_FIELDS = (
    "system",
    "comparison",
    "c_hits",
    "other_hits",
    "n",
    "fisher_p",
    "c_median_calls",
    "other_median_calls",
    "mannwhitney_u",
    "mannwhitney_p",
    "hit_drop",
    "call_increase",
    "significant_loss",
)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    hits = hits_a + hits_b
    denominator = math.comb(n_a + n_b, hits)

    def probability(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / denominator

    observed = probability(hits_a)
    low = max(0, hits - n_b)
    high = min(n_a, hits)
    return min(
        1.0,
        sum(
            probability(k)
            for k in range(low, high + 1)
            if probability(k) <= observed * (1.0 + 1.0e-12)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    combined = [(value, 0) for value in left] + [(value, 1) for value in right]
    combined.sort(key=lambda item: item[0])
    rank_sum = 0.0
    ties = []
    index = 0
    while index < len(combined):
        stop = index + 1
        while stop < len(combined) and combined[stop][0] == combined[index][0]:
            stop += 1
        rank = (index + 1 + stop) / 2.0
        rank_sum += rank * sum(group == 0 for _, group in combined[index:stop])
        ties.append(stop - index)
        index = stop
    n_left = len(left)
    n_right = len(right)
    u = rank_sum - n_left * (n_left + 1) / 2.0
    total = n_left + n_right
    tie_term = sum(size**3 - size for size in ties)
    variance = n_left * n_right / 12.0 * (
        total + 1.0 - tie_term / (total * (total - 1.0))
    )
    if variance <= 0.0:
        return u, 1.0
    mean = n_left * n_right / 2.0
    distance = max(0.0, abs(u - mean) - 0.5)
    z = distance / math.sqrt(variance)
    return u, math.erfc(z / math.sqrt(2.0))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def validate(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    if not rows or tuple(rows[0]) != FIELDS:
        raise SystemExit("rows have an unexpected header")
    expected_systems = {system.name: system for system in SYSTEMS}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    keys = set()
    for row in rows:
        if row["system"] not in expected_systems or row["arm"] not in ARMS:
            raise SystemExit(f"unexpected row {row['system']} {row['arm']}")
        key = (row["system"], row["arm"], int(row["seed"]))
        if key in keys:
            raise SystemExit(f"duplicate row {key}")
        keys.add(key)
        if row["status"] != "ok":
            raise SystemExit(f"failed row {key}: {row['status']}")
        if row["host"] != "bench":
            raise SystemExit(f"row {key} has host {row['host']}")
        system = expected_systems[row["system"]]
        if int(row["budget"]) != system.budget:
            raise SystemExit(f"row {key} has budget {row['budget']}")
        grouped.setdefault((row["system"], row["arm"]), []).append(row)
    for system in SYSTEMS:
        for arm in ARMS:
            group = grouped.get((system.name, arm), [])
            seeds = sorted(int(row["seed"]) for row in group)
            if seeds != list(SEEDS):
                raise SystemExit(f"{system.name} {arm} has seeds {seeds}")
            group.sort(key=lambda row: int(row["seed"]))
    return grouped


def summarize(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    summaries = []
    comparisons = []
    loses = False
    for system in SYSTEMS:
        for arm in ARMS:
            group = grouped[(system.name, arm)]
            hits = sum(row["hit"] == "1" for row in group)
            summaries.append(
                {
                    "system": system.name,
                    "arm": arm,
                    "hits": str(hits),
                    "n": str(len(group)),
                    "hits_over_n": f"{hits}/{len(group)}",
                    "median_calls": fmt(
                        statistics.median(float(row["calls"]) for row in group), 1
                    ),
                    "median_wall_s": fmt(
                        statistics.median(float(row["wall_s"]) for row in group)
                    ),
                    "median_pot_s": fmt(
                        statistics.median(float(row["pot_s"]) for row in group)
                    ),
                    "median_outside_s": fmt(
                        statistics.median(float(row["outside_s"]) for row in group)
                    ),
                    "code_revision": group[0]["code_revision"],
                    "host": group[0]["host"],
                }
            )
        c_group = grouped[(system.name, "C")]
        c_hits = sum(row["hit"] == "1" for row in c_group)
        c_calls = [float(row["calls"]) for row in c_group]
        for other in ("A", "B"):
            other_group = grouped[(system.name, other)]
            other_hits = sum(row["hit"] == "1" for row in other_group)
            other_calls = [float(row["calls"]) for row in other_group]
            fisher_p = fisher(c_hits, len(c_group), other_hits, len(other_group))
            u, mann_p = mann_whitney(c_calls, other_calls)
            c_median = statistics.median(c_calls)
            other_median = statistics.median(other_calls)
            hit_drop = c_hits < other_hits
            call_increase = c_median > other_median
            significant = (hit_drop and fisher_p < 0.05) or (
                call_increase and mann_p < 0.05
            )
            loses = loses or significant
            comparisons.append(
                {
                    "system": system.name,
                    "comparison": f"C-vs-{other}",
                    "c_hits": str(c_hits),
                    "other_hits": str(other_hits),
                    "n": str(len(c_group)),
                    "fisher_p": fmt(fisher_p),
                    "c_median_calls": fmt(c_median, 1),
                    "other_median_calls": fmt(other_median, 1),
                    "mannwhitney_u": fmt(u, 1),
                    "mannwhitney_p": fmt(mann_p),
                    "hit_drop": str(hit_drop).lower(),
                    "call_increase": str(call_increase).lower(),
                    "significant_loss": str(significant).lower(),
                }
            )
    return summaries, comparisons, "LOSES (opt-in only)" if loses else "PASS"


def markdown(
    summaries: list[dict[str, str]],
    comparisons: list[dict[str, str]],
    verdict: str,
    c_tree: str,
) -> str:
    by_summary = {(row["system"], row["arm"]): row for row in summaries}
    revisions = {arm: by_summary[(SYSTEMS[0].name, arm)]["code_revision"] for arm in ARMS}
    lines = [
        "# Shelf gate",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "A, B and C are respectively develop, the shelf branch, and the shelf branch plus the four supplied patches.",
        f"C's tree after applying the patches is `{c_tree}` (expected `{EXPECTED_C_TREE}`).",
        f"Revisions: A `{revisions['A']}`, B `{revisions['B']}`, C `{revisions['C']}`.",
        "",
        "All systems use seeds 2000–2011 (n=12), paper call budgets, the recommended preset, one process at a time on core 0, and one thread. Arms rotate A/B/C, B/C/A, C/A/B by seed. Every row was recorded under the synthetic host `bench`.",
        "",
        "## Per-arm results",
        "",
        "| system | A hits | A calls | A inside / outside (s) | B hits | B calls | B inside / outside (s) | C hits | C calls | C inside / outside (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        cells = [system.name]
        for arm in ARMS:
            row = by_summary[(system.name, arm)]
            cells.extend(
                [
                    row["hits_over_n"],
                    row["median_calls"],
                    f"{row['median_pot_s']} / {row['median_outside_s']}",
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Pairwise gate",
            "",
            "| system | comparison | hits C / other | Fisher p | median calls C / other | Mann–Whitney p | hit drop | call increase |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in comparisons:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["system"],
                    row["comparison"],
                    f"{row['c_hits']} / {row['other_hits']}",
                    row["fisher_p"],
                    f"{row['c_median_calls']} / {row['other_median_calls']}",
                    row["mannwhitney_p"],
                    row["hit_drop"],
                    row["call_increase"],
                ]
            )
            + " |"
        )
    worse = [
        row
        for row in comparisons
        if row["hit_drop"] == "true" or row["call_increase"] == "true"
    ]
    lines.extend(["", "## Every regression direction", ""])
    if not worse:
        lines.append("None.")
    else:
        for row in worse:
            directions = []
            if row["hit_drop"] == "true":
                directions.append(
                    f"hits {row['c_hits']} against {row['other_hits']} (Fisher p {row['fisher_p']})"
                )
            if row["call_increase"] == "true":
                directions.append(
                    f"median calls {row['c_median_calls']} against {row['other_median_calls']} "
                    f"(Mann–Whitney p {row['mannwhitney_p']})"
                )
            lines.append(f"- {row['system']} {row['comparison']}: " + "; ".join(directions) + ".")
    lines.extend(
        [
            "",
            "The verdict changes to LOSES only for a hit drop with two-sided Fisher p < 0.05 or a median call increase with tie-corrected two-sided Mann–Whitney p < 0.05.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--comparisons", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--c-tree", required=True)
    args = parser.parse_args()
    if args.c_tree != EXPECTED_C_TREE:
        raise SystemExit(f"C tree {args.c_tree} != {EXPECTED_C_TREE}")
    with args.rows.open(newline="") as handle:
        grouped = validate(list(csv.DictReader(handle)))
    summaries, comparisons, verdict = summarize(grouped)
    write_csv(args.summary, SUMMARY_FIELDS, summaries)
    write_csv(args.comparisons, COMPARISON_FIELDS, comparisons)
    args.verdict.write_text(verdict + "\n")
    args.readme.write_text(markdown(summaries, comparisons, verdict, args.c_tree))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
