#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Add trajectory and three-rule analysis without running searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import run_gate as gate
import summarize

HERE = Path(__file__).resolve().parent
TRAJECTORY_FIELDS = (
    "cell", "seed", "hit_equal", "calls_equal", "charged_equal",
    "best_equal", "trace_sha256_a", "trace_sha256_b", "trace_equal",
    "geometry_sha256_a", "geometry_sha256_b", "geometry_equal",
    "all_equal",
)
DIVERGENCE_FIELDS = (
    "cell", "seed", "trace_record", "call_a", "energy_a", "call_b",
    "energy_b", "difference",
)
RULE_FIELDS = (
    "cell", "n", "a_only_hits_b", "b_only_hits_c", "mcnemar_exact_p",
    "hit_clause", "outside_ratio_b_a", "outside_ratio_a2_a",
    "outside_limit", "cost_clause", "a2_identity_count",
    "non_rigid_identity_count", "non_rigid_clause", "verdict",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def mcnemar_exact(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(b, c) + 1)
    ) / 2**discordant
    return min(1.0, 2 * tail)


def trace_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def first_divergence(
    a_path: Path,
    b_path: Path,
) -> tuple[str, str, str, str, str, str]:
    a_rows = trace_rows(a_path)
    b_rows = trace_rows(b_path)
    for index in range(max(len(a_rows), len(b_rows))):
        a = a_rows[index] if index < len(a_rows) else None
        b = b_rows[index] if index < len(b_rows) else None
        if a == b:
            continue
        if a is None or b is None:
            return (
                str(index + 1),
                a["calls"] if a else "",
                a["energy"] if a else "",
                b["calls"] if b else "",
                b["energy"] if b else "",
                "one trace ended",
            )
        differs = [
            field for field in ("calls", "energy")
            if a[field] != b[field]
        ]
        return (
            str(index + 1),
            a["calls"],
            a["energy"],
            b["calls"],
            b["energy"],
            " and ".join(differs),
        )
    return ("", "", "", "", "", "trace bytes differ outside parsed rows")


def trajectory_analysis(
    rows: list[dict[str, str]],
    logs: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    by = {
        (row["cell"], row["arm"], int(row["seed"])): row
        for row in rows
    }
    identities = []
    divergences = []
    summaries = []
    for size in range(15, 22):
        cell = f"tip4p{size}"
        count = 0
        for seed in gate.TIP4P_SEEDS:
            a = by[(cell, "A", seed)]
            b = by[(cell, "B", seed)]
            stem = f"{cell}-s{seed}"
            trace_a_path = logs / f"{stem}-a.trace.csv"
            trace_b_path = logs / f"{stem}-b.trace.csv"
            geometry_a_path = logs / f"{stem}-a.xyz"
            geometry_b_path = logs / f"{stem}-b.xyz"
            trace_a = sha256_file(trace_a_path)
            trace_b = sha256_file(trace_b_path)
            geometry_a = sha256_file(geometry_a_path)
            geometry_b = sha256_file(geometry_b_path)
            if geometry_a != a["final_sha256"]:
                raise RuntimeError(f"{cell} {seed} A geometry digest drift")
            if geometry_b != b["final_sha256"]:
                raise RuntimeError(f"{cell} {seed} B geometry digest drift")
            equal = {
                "hit": str(int(
                    summarize.energy_hit(a, float(a["reference"]))
                    == summarize.energy_hit(b, float(b["reference"]))
                )),
                "calls": str(int(a["calls"] == b["calls"])),
                "charged": str(int(a["charged"] == b["charged"])),
                "best": str(int(a["energy"] == b["energy"])),
                "trace": str(int(trace_a == trace_b)),
                "geometry": str(int(geometry_a == geometry_b)),
            }
            all_equal = all(value == "1" for value in equal.values())
            count += all_equal
            identities.append({
                "cell": cell,
                "seed": str(seed),
                "hit_equal": equal["hit"],
                "calls_equal": equal["calls"],
                "charged_equal": equal["charged"],
                "best_equal": equal["best"],
                "trace_sha256_a": trace_a,
                "trace_sha256_b": trace_b,
                "trace_equal": equal["trace"],
                "geometry_sha256_a": geometry_a,
                "geometry_sha256_b": geometry_b,
                "geometry_equal": equal["geometry"],
                "all_equal": str(int(all_equal)),
            })
            if all_equal:
                continue
            if trace_a != trace_b:
                record, call_a, energy_a, call_b, energy_b, difference = (
                    first_divergence(trace_a_path, trace_b_path)
                )
            else:
                record = call_a = energy_a = call_b = energy_b = ""
                difference = "trace equal; final recorded fields differ"
            divergences.append({
                "cell": cell,
                "seed": str(seed),
                "trace_record": record,
                "call_a": call_a,
                "energy_a": energy_a,
                "call_b": call_b,
                "energy_b": energy_b,
                "difference": difference,
            })
        summaries.append({
            "cell": cell,
            "n": str(len(gate.TIP4P_SEEDS)),
            "identical": str(count),
        })
    return identities, divergences, summaries


def result_identity(a: dict[str, str], b: dict[str, str]) -> bool:
    reference = float(a["reference"])
    return (
        summarize.energy_hit(a, reference)
        == summarize.energy_hit(b, reference)
        and a["calls"] == b["calls"]
        and a["charged"] == b["charged"]
        and a["energy_bits"] == b["energy_bits"] != ""
    )


def three_rule_analysis(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by = {
        (row["cell"], row["arm"], int(row["seed"])): row
        for row in rows
    }
    output = []
    for cell, _budget, reference, seeds in gate.cells():
        a = [by[(cell, "A", seed)] for seed in seeds]
        a2 = [by[(cell, "A2", seed)] for seed in seeds]
        b_rows = [by[(cell, "B", seed)] for seed in seeds]
        a_hits = [summarize.energy_hit(row, reference) for row in a]
        b_hits = [
            summarize.energy_hit(row, reference) for row in b_rows
        ]
        b = sum(left and not right for left, right in zip(a_hits, b_hits))
        c = sum(right and not left for left, right in zip(a_hits, b_hits))
        mcnemar = mcnemar_exact(b, c)
        hit_clause = not (b > c and mcnemar < 0.05)
        outside_a = [float(row["per_call_s"]) for row in a]
        outside_a2 = [float(row["per_call_s"]) for row in a2]
        outside_b = [float(row["per_call_s"]) for row in b_rows]
        ratio_b = summarize.ratio(outside_a, outside_b)
        ratio_a2 = summarize.ratio(outside_a, outside_a2)
        limit = max(gate.MAX_OUTSIDE_RATIO, ratio_a2)
        cost_clause = ratio_b <= limit
        a2_identity = sum(
            result_identity(left, right)
            for left, right in zip(a, a2)
        )
        non_rigid = not cell.startswith("tip4p")
        non_rigid_identity = sum(
            result_identity(left, right)
            for left, right in zip(a, b_rows)
        ) if non_rigid else 0
        non_rigid_clause = (
            non_rigid_identity == len(seeds) if non_rigid else True
        )
        verdict = hit_clause and cost_clause and non_rigid_clause
        output.append({
            "cell": cell,
            "n": str(len(seeds)),
            "a_only_hits_b": str(b),
            "b_only_hits_c": str(c),
            "mcnemar_exact_p": summarize.fmt(mcnemar),
            "hit_clause": "PASS" if hit_clause else "FAIL",
            "outside_ratio_b_a": summarize.fmt(ratio_b),
            "outside_ratio_a2_a": summarize.fmt(ratio_a2),
            "outside_limit": summarize.fmt(limit),
            "cost_clause": "PASS" if cost_clause else "FAIL",
            "a2_identity_count": str(a2_identity),
            "non_rigid_identity_count": (
                str(non_rigid_identity) if non_rigid else ""
            ),
            "non_rigid_clause": (
                "PASS" if non_rigid_clause else "FAIL"
            ) if non_rigid else "N/A",
            "verdict": "PASS" if verdict else "FAIL",
        })
    return output


def write_report(
    trajectories: list[dict[str, str]],
    divergences: list[dict[str, str]],
    trajectory_summaries: list[dict[str, str]],
    rules: list[dict[str, str]],
) -> None:
    trajectory_pass = all(
        row["all_equal"] == "1" for row in trajectories
    )
    hit_pass = all(row["hit_clause"] == "PASS" for row in rules)
    cost_pass = all(row["cost_clause"] == "PASS" for row in rules)
    non_rigid_pass = all(
        row["non_rigid_clause"] in ("PASS", "N/A") for row in rules
    )
    verdict = (
        "PASS" if hit_pass and cost_pass and non_rigid_pass else "FAIL"
    )
    lines = [
        "# Added trajectory and three-rule analysis",
        "",
        "No search was rerun or replaced for this analysis.",
        "",
        f"Three-rule verdict: **{verdict}**.",
        f"McNemar no-loss rule: **{'PASS' if hit_pass else 'FAIL'}**.",
        f"Rebuild-band outside rule: **{'PASS' if cost_pass else 'FAIL'}**.",
        f"Non-rigid identity rule: "
        f"**{'PASS' if non_rigid_pass else 'FAIL'}**.",
        "",
        "## TIP4P trajectory identity",
        "",
        f"All requested trajectories identical: "
        f"**{'yes' if trajectory_pass else 'no'}**.",
        "",
        "| Cell | A==B |",
        "| --- | ---: |",
    ]
    for row in trajectory_summaries:
        lines.append(
            f"| {row['cell']} | {row['identical']}/{row['n']} |"
        )
    lines += [
        "",
        f"Non-identical seeds: {len(divergences)}.",
        "Their first observable trace differences are recorded in",
        "`trajectory-divergence.csv`.",
        "",
        "## Three-rule cell results",
        "",
        "| Cell | b | c | McNemar p | B/A outside | A2/A | Limit | "
        "A2 identity | Non-rigid identity | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- |",
    ]
    for row in rules:
        non_rigid = (
            f"{row['non_rigid_identity_count']}/{row['n']}"
            if row["non_rigid_identity_count"] else "N/A"
        )
        lines.append(
            f"| {row['cell']} | {row['a_only_hits_b']} | "
            f"{row['b_only_hits_c']} | {row['mcnemar_exact_p']} | "
            f"{row['outside_ratio_b_a']} | "
            f"{row['outside_ratio_a2_a']} | {row['outside_limit']} | "
            f"{row['a2_identity_count']}/{row['n']} | {non_rigid} | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "`b` is A-only hits and `c` is B-only hits. The no-loss rule fails",
        "only when `b > c` and the exact two-sided McNemar p-value is below",
        "0.05. The outside limit is `max(1.10, A2/A)` per cell.",
        "",
    ]
    (HERE / "ADDED_ANALYSIS.md").write_text("\n".join(lines))
    print(verdict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    with (HERE / "rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities, divergences, trajectory_summaries = trajectory_analysis(
        rows, args.logs
    )
    rules = three_rule_analysis(rows)
    write_csv(
        HERE / "trajectory-identity.csv",
        TRAJECTORY_FIELDS,
        identities,
    )
    write_csv(
        HERE / "trajectory-divergence.csv",
        DIVERGENCE_FIELDS,
        divergences,
    )
    write_csv(
        HERE / "trajectory-summary.csv",
        ("cell", "n", "identical"),
        trajectory_summaries,
    )
    write_csv(HERE / "three-rule-summary.csv", RULE_FIELDS, rules)
    write_report(identities, divergences, trajectory_summaries, rules)


if __name__ == "__main__":
    main()
