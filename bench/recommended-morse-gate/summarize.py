#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Exact tests for the recommended Morse gate.

Hits use a two-sided Fisher exact test. Evals to hit and wall time to hit
are the hitting runs only, compared with a two-sided exact Mann-Whitney
test. Medians are the ordinary median, which matches Hyndman-Fan type 7
at probability 1/2. The four Morse systems are one family: each reported
p-value is also given as min(1, 4p).

A cell is a B loss when B has fewer hits, or a higher median eval count
among hits, and the adjusted p-value is below 0.05. Wall time is reported
and is not part of that rule.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

ALPHA = 0.05
BONFERRONI_M = 4
SYSTEMS = ("m38r6", "m38r14", "m55r6", "m55r14")
AA_SYSTEMS = ("m38r14", "m55r6")
SEEDS = tuple(range(2000, 2012))
SUMMARY_FIELDS = [
    "comparison",
    "system",
    "arm_a",
    "arm_b",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_p",
    "fisher_p_adj",
    "n_hit_a",
    "n_hit_b",
    "median_evals_a",
    "median_evals_b",
    "mw_u_evals",
    "mw_p_evals",
    "mw_p_evals_adj",
    "median_wall_a",
    "median_wall_b",
    "mw_u_wall",
    "mw_p_wall",
    "mw_p_wall_adj",
    "bit_identical",
    "bit_identical_over_n",
    "bonferroni_m",
    "lose_hits",
    "lose_evals",
    "status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    table = [[hits_a, n_a - hits_a], [hits_b, n_b - hits_b]]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def exact_mw(left: list[float], right: list[float]) -> tuple[float, float]:
    result = mannwhitneyu(left, right, alternative="two-sided", method="exact")
    return float(result.statistic), float(result.pvalue)


def adjust(p: float, m: int) -> float:
    if not math.isfinite(p):
        return p
    return min(1.0, p * m)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return format(value, ".16g")


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def record_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["hit"], row["evals_to_hit"], row["charged"], row["best"])


def index_rows(rows: list[dict[str, str]], band: str, system: str, arm: str) -> dict[int, dict[str, str]]:
    found: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["band"] == band and row["system"] == system and row["arm"] == arm:
            found[int(row["seed"])] = row
    return found


def hit_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values = []
    for row in rows:
        if row["hit"] != "1" or row[key] == "":
            continue
        values.append(float(row[key]))
    return values


def compare(
    name: str,
    system: str,
    left_name: str,
    right_name: str,
    left: dict[int, dict[str, str]],
    right: dict[int, dict[str, str]],
    *,
    m: int | None,
) -> dict[str, str]:
    seeds = sorted(set(left) & set(right))
    status = "ok"
    if sorted(left) != list(SEEDS) or sorted(right) != list(SEEDS):
        status = f"incomplete:{len(left)}:{len(right)}"
    identical = 0
    for seed in seeds:
        identical += int(record_key(left[seed]) == record_key(right[seed]))
    left_rows = [left[seed] for seed in sorted(left)]
    right_rows = [right[seed] for seed in sorted(right)]
    hits_a = sum(row["hit"] == "1" for row in left_rows)
    hits_b = sum(row["hit"] == "1" for row in right_rows)
    fisher_p = fisher(hits_a, len(left_rows), hits_b, len(right_rows)) if left_rows and right_rows else float("nan")
    evals_a = hit_values(left_rows, "evals_to_hit")
    evals_b = hit_values(right_rows, "evals_to_hit")
    wall_a = hit_values(left_rows, "wall_s")
    wall_b = hit_values(right_rows, "wall_s")
    if evals_a and evals_b:
        u_evals, p_evals = exact_mw(evals_a, evals_b)
    else:
        u_evals, p_evals = float("nan"), float("nan")
    if wall_a and wall_b:
        u_wall, p_wall = exact_mw(wall_a, wall_b)
    else:
        u_wall, p_wall = float("nan"), float("nan")
    med_a = median(evals_a)
    med_b = median(evals_b)
    med_wall_a = median(wall_a)
    med_wall_b = median(wall_b)
    family = m if m is not None else 0
    fisher_adj = adjust(fisher_p, family) if family else float("nan")
    evals_adj = adjust(p_evals, family) if family else float("nan")
    wall_adj = adjust(p_wall, family) if family else float("nan")
    lose_hits = int(
        status == "ok" and family > 0 and hits_b < hits_a and math.isfinite(fisher_adj) and fisher_adj < ALPHA
    )
    lose_evals = int(
        status == "ok"
        and family > 0
        and math.isfinite(med_a)
        and math.isfinite(med_b)
        and med_b > med_a
        and math.isfinite(evals_adj)
        and evals_adj < ALPHA
    )
    return {
        "comparison": name,
        "system": system,
        "arm_a": left_name,
        "arm_b": right_name,
        "n_a": str(len(left_rows)),
        "n_b": str(len(right_rows)),
        "hits_a": str(hits_a),
        "hits_b": str(hits_b),
        "fisher_p": fmt(fisher_p),
        "fisher_p_adj": fmt(fisher_adj) if family else "",
        "n_hit_a": str(len(evals_a)),
        "n_hit_b": str(len(evals_b)),
        "median_evals_a": fmt(med_a),
        "median_evals_b": fmt(med_b),
        "mw_u_evals": fmt(u_evals),
        "mw_p_evals": fmt(p_evals),
        "mw_p_evals_adj": fmt(evals_adj) if family else "",
        "median_wall_a": fmt(med_wall_a),
        "median_wall_b": fmt(med_wall_b),
        "mw_u_wall": fmt(u_wall),
        "mw_p_wall": fmt(p_wall),
        "mw_p_wall_adj": fmt(wall_adj) if family else "",
        "bit_identical": str(identical),
        "bit_identical_over_n": f"{identical}/{len(seeds)}" if seeds else "",
        "bonferroni_m": str(family) if family else "",
        "lose_hits": str(lose_hits),
        "lose_evals": str(lose_evals),
        "status": status,
    }


def pooled_call_samples(rows: list[dict[str, str]], system: str) -> tuple[list[float], list[float], list[float], list[float]]:
    reference = index_rows(rows, "AB", system, "B")
    null_arms = [("AA", "B2"), ("PL", "end64"), ("PL", "start256"), ("PL", "align1024")]
    ref_calls = hit_values([reference[seed] for seed in sorted(reference)], "evals_to_hit")
    ref_wall = hit_values([reference[seed] for seed in sorted(reference)], "wall_s")
    null_calls: list[float] = []
    null_wall: list[float] = []
    for band, arm in null_arms:
        found = index_rows(rows, band, system, arm)
        null_calls.extend(hit_values([found[seed] for seed in sorted(found)], "evals_to_hit"))
        null_wall.extend(hit_values([found[seed] for seed in sorted(found)], "wall_s"))
    return ref_calls, null_calls, ref_wall, null_wall


def analyze(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for system in SYSTEMS:
        summary.append(
            compare(
                "AB",
                system,
                "A",
                "B",
                index_rows(rows, "AB", system, "A"),
                index_rows(rows, "AB", system, "B"),
                m=BONFERRONI_M,
            )
        )
    for system in AA_SYSTEMS:
        summary.append(
            compare(
                "AA",
                system,
                "B",
                "B2",
                index_rows(rows, "AA", system, "B"),
                index_rows(rows, "AA", system, "B2"),
                m=None,
            )
        )
        reference = index_rows(rows, "AB", system, "B")
        for kind in ("end64", "start256", "align1024"):
            summary.append(
                compare(
                    kind,
                    system,
                    "B",
                    kind,
                    reference,
                    index_rows(rows, "PL", system, kind),
                    m=None,
                )
            )
        ref_calls, null_calls, ref_wall, null_wall = pooled_call_samples(rows, system)
        if ref_calls and null_calls:
            u_calls, p_calls = exact_mw(ref_calls, null_calls)
        else:
            u_calls, p_calls = float("nan"), float("nan")
        if ref_wall and null_wall:
            u_wall, p_wall = exact_mw(ref_wall, null_wall)
        else:
            u_wall, p_wall = float("nan"), float("nan")
        ratios = []
        for item in summary:
            if item["system"] == system and item["comparison"] in {"AA", "end64", "start256", "align1024"}:
                if item["median_evals_a"] and item["median_evals_b"]:
                    ratios.append(float(item["median_evals_b"]) / float(item["median_evals_a"]))
        summary.append(
            {
                "comparison": "POOL",
                "system": system,
                "arm_a": "B",
                "arm_b": "B2+end64+start256+align1024",
                "n_a": str(len(ref_calls)),
                "n_b": str(len(null_calls)),
                "hits_a": "",
                "hits_b": "",
                "fisher_p": "",
                "fisher_p_adj": "",
                "n_hit_a": str(len(ref_calls)),
                "n_hit_b": str(len(null_calls)),
                "median_evals_a": fmt(median(ref_calls)),
                "median_evals_b": fmt(median(null_calls)),
                "mw_u_evals": fmt(u_calls),
                "mw_p_evals": fmt(p_calls),
                "mw_p_evals_adj": "",
                "median_wall_a": fmt(median(ref_wall)),
                "median_wall_b": fmt(median(null_wall)),
                "mw_u_wall": fmt(u_wall),
                "mw_p_wall": fmt(p_wall),
                "mw_p_wall_adj": "",
                "bit_identical": "",
                "bit_identical_over_n": "",
                "bonferroni_m": "",
                "lose_hits": "0",
                "lose_evals": "0",
                "status": "ratio_band=" + (fmt(min(ratios)) + ":" + fmt(max(ratios)) if ratios else ""),
            }
        )
    return summary


def verdict_text(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> str:
    ab = [row for row in summary if row["comparison"] == "AB"]
    incomplete = [row for row in ab if row["status"] != "ok"]
    losses = [row for row in ab if row["lose_hits"] == "1" or row["lose_evals"] == "1"]
    if incomplete:
        head = "INCOMPLETE"
    elif losses:
        head = "LOSE"
    else:
        head = "PASS"
    lines = [
        f"verdict: {head}",
        "rule: B loses a cell when it has fewer hits, or a higher median eval count on hitting runs, and the Bonferroni adjusted p at m=4 is below 0.05",
        "family: m38r6 m38r14 m55r6 m55r14",
        "evals: hitting runs only",
        "wall: hitting runs only, reported, not in the loss rule",
        "median: ordinary median (Hyndman-Fan type 7 at 1/2)",
        "fisher: two-sided exact",
        "mannwhitney: two-sided exact, U is the statistic of arm A",
        "",
        "cell hits_a hits_b fisher_p fisher_p_adj median_evals_a median_evals_b mw_p_evals mw_p_evals_adj median_wall_a median_wall_b mw_p_wall mw_p_wall_adj bit_identical lose_hits lose_evals",
    ]
    for row in ab:
        lines.append(
            " ".join(
                [
                    row["system"],
                    f"{row['hits_a']}/{row['n_a']}",
                    f"{row['hits_b']}/{row['n_b']}",
                    row["fisher_p"],
                    row["fisher_p_adj"],
                    row["median_evals_a"],
                    row["median_evals_b"],
                    row["mw_p_evals"],
                    row["mw_p_evals_adj"],
                    row["median_wall_a"],
                    row["median_wall_b"],
                    row["mw_p_wall"],
                    row["mw_p_wall_adj"],
                    row["bit_identical_over_n"],
                    row["lose_hits"],
                    row["lose_evals"],
                ]
            )
        )
    lines.extend(["", "per-seed final energy and evals"])
    lines.append("system seed energy_a evals_a charged_a energy_b evals_b charged_b identical")
    for system in SYSTEMS:
        left = index_rows(rows, "AB", system, "A")
        right = index_rows(rows, "AB", system, "B")
        for seed in SEEDS:
            a = left.get(seed)
            b = right.get(seed)
            if a is None or b is None:
                lines.append(f"{system} {seed} missing")
                continue
            same = int(record_key(a) == record_key(b))
            lines.append(
                f"{system} {seed} {a['best']} {a['evals_to_hit'] or '-'} {a['charged']} "
                f"{b['best']} {b['evals_to_hit'] or '-'} {b['charged']} {same}"
            )
    lines.extend(["", "A/A and placement, pooled on m38r14 and m55r6"])
    for row in summary:
        if row["comparison"] == "AB":
            continue
        lines.append(
            " ".join(
                [
                    row["comparison"],
                    row["system"],
                    f"hits {row['hits_a']}/{row['n_a']} vs {row['hits_b']}/{row['n_b']}" if row["hits_a"] else f"n {row['n_a']} vs {row['n_b']}",
                    f"fisher {row['fisher_p']}" if row["fisher_p"] else "fisher",
                    f"evals {row['median_evals_a']} {row['median_evals_b']} p {row['mw_p_evals']}",
                    f"wall {row['median_wall_a']} {row['median_wall_b']} p {row['mw_p_wall']}",
                    f"identical {row['bit_identical_over_n']}" if row["bit_identical_over_n"] else row["status"],
                ]
            )
        )
    m38 = next(row for row in ab if row["system"] == "m38r14")
    trigger = m38["status"] == "ok" and (m38["lose_hits"] == "1" or m38["lose_evals"] == "1")
    lines.extend(
        [
            "",
            f"m38r14_b_loss: {int(trigger)}",
            "bisect: pending" if trigger else "bisect: not_required",
        ]
    )
    return "\n".join(lines) + "\n"


def self_check() -> None:
    twelve = fisher(12, 12, 0, 12)
    five = fisher(5, 12, 0, 12)
    if abs(twelve - 7.39602301050679e-07) > 1e-18:
        raise SystemExit(f"fisher 12 vs 0 is {twelve}")
    if abs(five - 0.03726708074534161) > 1e-15:
        raise SystemExit(f"fisher 5 vs 0 is {five}")
    if fisher(12, 12, 12, 12) != 1.0:
        raise SystemExit("fisher 12 vs 12 is not 1")
    tied = mannwhitneyu([1, 2, 3, 4], [1, 2, 3, 4], alternative="two-sided", method="exact")
    if float(tied.pvalue) != 1.0:
        raise SystemExit("tied exact Mann-Whitney is not 1")
    small = mannwhitneyu([10, 20, 30], [11, 21, 40], alternative="two-sided", method="exact")
    if abs(float(small.statistic) - 3.0) > 1e-12 or abs(float(small.pvalue) - 0.7) > 1e-12:
        raise SystemExit("small exact Mann-Whitney drifted")
    if adjust(0.02, 4) != 0.08 or adjust(0.4, 4) != 1.0:
        raise SystemExit("Bonferroni adjustment drifted")
    if statistics.median([1, 2, 3, 4]) != 2.5:
        raise SystemExit("median is not the type-7 midpoint")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--verdict", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    self_check()
    if args.self_check and args.rows is None:
        return 0
    if args.rows is None or args.summary is None or args.verdict is None:
        raise SystemExit("rows, summary, and verdict are required")
    rows = read_csv(args.rows)
    summary = analyze(rows)
    write_csv(args.summary, SUMMARY_FIELDS, summary)
    args.verdict.write_text(verdict_text(rows, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
