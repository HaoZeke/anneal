#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Analyze the no-builders14 gate rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import run_gate  # noqa: E402

FIELDS = (
    "family",
    "system",
    "arm",
    "n",
    "hits_a",
    "hits_candidate",
    "fisher_p",
    "calls_mwu",
    "calls_mwu_p",
    "outside_a_s_per_call",
    "outside_candidate_s_per_call",
    "outside_ratio",
    "outside_ci_low",
    "outside_ci_high",
    "outside_aa_ratio",
    "outside_aa_ci_low",
    "outside_aa_ci_high",
    "rss_a_kib",
    "rss_candidate_kib",
    "rss_ratio",
    "rss_ci_low",
    "rss_ci_high",
    "rss_aa_ratio",
    "rss_aa_ci_low",
    "rss_aa_ci_high",
    "ru_maxrss_a_kib",
    "ru_maxrss_candidate_kib",
    "vm_hwm_a_kib",
    "vm_hwm_candidate_kib",
    "hit_verdict",
    "outside_verdict",
    "rss_verdict",
    "verdict",
    "reason",
)


def f(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.8g}"


def expected_keys(system: str) -> set[tuple[str, str, str, str, str]]:
    out = set()
    for seed in common.SEEDS:
        for block, arm, repeat, _slot in run_gate.arm_plan(seed):
            out.add((system, str(seed), block, str(repeat), arm))
    return out


def seed_values(
    rows: list[dict[str, str]],
    arm: str,
    column: str,
    *,
    seeds: tuple[int, ...] = common.SEEDS,
    block: str = "main",
    per_call: bool = False,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in seeds:
        selected = [
            row
            for row in rows
            if row["arm"] == arm
            and int(row["seed"]) == seed
            and row["block"] == block
        ]
        values = []
        for row in selected:
            value = float(row[column])
            if per_call:
                value /= max(float(row["charged"]), 1.0)
            values.append(value)
        if values:
            out[seed] = common.median7(values)
    return out


def seed_hits(
    rows: list[dict[str, str]], arm: str
) -> tuple[dict[int, int], str]:
    out: dict[int, int] = {}
    for seed in common.SEEDS:
        values = {
            row["hit"]
            for row in rows
            if row["arm"] == arm
            and int(row["seed"]) == seed
            and row["block"] == "main"
        }
        if len(values) != 1:
            return {}, f"{arm} seed {seed} has inconsistent hits"
        out[seed] = int(values.pop())
    return out, ""


def seed_calls(rows: list[dict[str, str]], arm: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for seed in common.SEEDS:
        values = [
            float(row["calls_to_hit"])
            for row in rows
            if row["arm"] == arm
            and int(row["seed"]) == seed
            and row["block"] == "main"
        ]
        if values:
            out[seed] = common.median7(values)
    return out


def paired_ratio(
    top: dict[int, float], bottom: dict[int, float]
) -> float:
    seeds = sorted(set(top) & set(bottom))
    return common.ratio(
        common.median7([top[seed] for seed in seeds]),
        common.median7([bottom[seed] for seed in seeds]),
    )


def bootstrap_ratio(
    top: dict[int, float],
    bottom: dict[int, float],
    tag: str,
    resamples: int = 10_000,
) -> tuple[float, float]:
    seeds = sorted(set(top) & set(bottom))
    digest = hashlib.sha256(tag.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    samples = []
    for _ in range(resamples):
        draw = [rng.choice(seeds) for _ in seeds]
        ratio = common.ratio(
            common.median7([top[seed] for seed in draw]),
            common.median7([bottom[seed] for seed in draw]),
        )
        if math.isfinite(ratio):
            samples.append(ratio)
    return common.quantile7(samples, 0.025), common.quantile7(
        samples, 0.975
    )


def med_all(rows: list[dict[str, str]], arm: str, column: str) -> float:
    return common.median7(
        [float(row[column]) for row in rows if row["arm"] == arm]
    )


def incomplete_row(
    family: str, system: str, arm: str, reason: str
) -> dict[str, str]:
    row = {field: "" for field in FIELDS}
    row.update(
        {
            "family": family,
            "system": system,
            "arm": arm,
            "hit_verdict": "INCONCLUSIVE",
            "outside_verdict": "INCONCLUSIVE",
            "rss_verdict": "INCONCLUSIVE",
            "verdict": "INCONCLUSIVE",
            "reason": reason,
        }
    )
    return row


def analyze_cell(
    family: str,
    system: str,
    rows: list[dict[str, str]],
    arm: str,
) -> dict[str, str]:
    a_hits, hit_error = seed_hits(rows, "A")
    c_hits, candidate_error = seed_hits(rows, arm)
    if hit_error or candidate_error:
        return incomplete_row(
            family, system, arm, hit_error or candidate_error
        )
    calls_a = seed_calls(rows, "A")
    calls_c = seed_calls(rows, arm)
    u, mw_p = common.mann_whitney(
        [calls_c[seed] for seed in common.SEEDS],
        [calls_a[seed] for seed in common.SEEDS],
    )
    hits_a = sum(a_hits.values())
    hits_c = sum(c_hits.values())
    fisher_p = common.fisher(hits_a, len(a_hits), hits_c, len(c_hits))

    outside_a = seed_values(
        rows, "A", "outside_s", per_call=True
    )
    outside_c = seed_values(
        rows, arm, "outside_s", per_call=True
    )
    six = tuple(range(2000, 2006))
    outside_aa_a = seed_values(
        rows, "A", "outside_s", seeds=six, block="aa", per_call=True
    )
    outside_aa_a2 = seed_values(
        rows, "A2", "outside_s", seeds=six, block="aa", per_call=True
    )
    out_ratio = paired_ratio(outside_c, outside_a)
    out_ci = bootstrap_ratio(
        outside_c, outside_a, f"{system}:{arm}:outside"
    )
    out_aa = paired_ratio(outside_aa_a2, outside_aa_a)
    out_aa_ci = bootstrap_ratio(
        outside_aa_a2, outside_aa_a, f"{system}:aa:outside"
    )

    rss_a = seed_values(rows, "A", "rss_anon_kib")
    rss_c = seed_values(rows, arm, "rss_anon_kib")
    rss_aa_a = seed_values(
        rows, "A", "rss_anon_kib", seeds=six, block="aa"
    )
    rss_aa_a2 = seed_values(
        rows, "A2", "rss_anon_kib", seeds=six, block="aa"
    )
    rss_ratio = paired_ratio(rss_c, rss_a)
    rss_ci = bootstrap_ratio(rss_c, rss_a, f"{system}:{arm}:rss")
    rss_aa = paired_ratio(rss_aa_a2, rss_aa_a)
    rss_aa_ci = bootstrap_ratio(
        rss_aa_a2, rss_aa_a, f"{system}:aa:rss"
    )

    hit_verdict = "PASS" if hits_c >= hits_a else "FAIL"
    outside_limit = max(1.10, out_aa_ci[1])
    outside_verdict = (
        "PASS"
        if out_ratio <= 1.10 and out_ci[1] <= outside_limit
        else "FAIL"
    )
    rss_limit = max(1.10, rss_aa_ci[1])
    rss_verdict = "PASS" if rss_ratio <= rss_limit else "FAIL"
    verdict = (
        "PASS"
        if hit_verdict == outside_verdict == rss_verdict == "PASS"
        else "FAIL"
    )
    return {
        "family": family,
        "system": system,
        "arm": arm,
        "n": str(len(common.SEEDS)),
        "hits_a": str(hits_a),
        "hits_candidate": str(hits_c),
        "fisher_p": f(fisher_p),
        "calls_mwu": f(u),
        "calls_mwu_p": f(mw_p),
        "outside_a_s_per_call": f(common.median7(outside_a.values())),
        "outside_candidate_s_per_call": f(
            common.median7(outside_c.values())
        ),
        "outside_ratio": f(out_ratio),
        "outside_ci_low": f(out_ci[0]),
        "outside_ci_high": f(out_ci[1]),
        "outside_aa_ratio": f(out_aa),
        "outside_aa_ci_low": f(out_aa_ci[0]),
        "outside_aa_ci_high": f(out_aa_ci[1]),
        "rss_a_kib": f(common.median7(rss_a.values())),
        "rss_candidate_kib": f(common.median7(rss_c.values())),
        "rss_ratio": f(rss_ratio),
        "rss_ci_low": f(rss_ci[0]),
        "rss_ci_high": f(rss_ci[1]),
        "rss_aa_ratio": f(rss_aa),
        "rss_aa_ci_low": f(rss_aa_ci[0]),
        "rss_aa_ci_high": f(rss_aa_ci[1]),
        "ru_maxrss_a_kib": f(med_all(rows, "A", "ru_maxrss_kib")),
        "ru_maxrss_candidate_kib": f(
            med_all(rows, arm, "ru_maxrss_kib")
        ),
        "vm_hwm_a_kib": f(med_all(rows, "A", "vm_hwm_kib")),
        "vm_hwm_candidate_kib": f(med_all(rows, arm, "vm_hwm_kib")),
        "hit_verdict": hit_verdict,
        "outside_verdict": outside_verdict,
        "rss_verdict": rss_verdict,
        "verdict": verdict,
        "reason": "",
    }


def render_report(summary: list[dict[str, str]], machines: set[str]) -> str:
    overall = (
        "PASS"
        if summary
        and all(row["verdict"] == "PASS" for row in summary)
        else (
            "INCONCLUSIVE"
            if any(row["verdict"] == "INCONCLUSIVE" for row in summary)
            else "FAIL"
        )
    )
    lines = [
        "# no-builders14 gate report",
        "",
        f"Overall verdict: **{overall}**.",
        "",
        f"Rows were recorded on {len(machines)} hashed machine identity.",
        "G was enabled exactly as `H2H_COMPLETION=fitted`.",
        "",
        "| Cell | Arm | Hits A/candidate | Fisher p | Calls MW p | "
        "Outside ratio [95% CI] | A2/A band | RSS ratio [95% CI] | "
        "Hit | Outside | RSS | Verdict |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | "
        "--- | --- | --- |",
    ]
    for row in summary:
        if row["verdict"] == "INCONCLUSIVE":
            lines.append(
                f"| {row['system']} | {row['arm']} | - | - | - | - | - | "
                f"- | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | "
                f"INCONCLUSIVE: {row['reason']} |"
            )
            continue
        lines.append(
            f"| {row['system']} | {row['arm']} | "
            f"{row['hits_a']}/{row['hits_candidate']} | "
            f"{row['fisher_p']} | {row['calls_mwu_p']} | "
            f"{row['outside_ratio']} "
            f"[{row['outside_ci_low']}, {row['outside_ci_high']}] | "
            f"{row['outside_aa_ratio']} "
            f"[{row['outside_aa_ci_low']}, "
            f"{row['outside_aa_ci_high']}] | "
            f"{row['rss_ratio']} [{row['rss_ci_low']}, "
            f"{row['rss_ci_high']}] | {row['hit_verdict']} | "
            f"{row['outside_verdict']} | {row['rss_verdict']} | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "Peak RssAnon is the memory gate. ru_maxrss and VmHWM are present "
        "in `summary.csv` for reporting only.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = common.read_rows(args.rows)
    machines = {row["machine"] for row in rows}
    summary = []
    for family, system, _driver, _budget, _reference in common.CELLS:
        cell = [row for row in rows if row["system"] == system]
        keys = [run_gate.row_key(row) for row in cell]
        reason = ""
        if len(keys) != len(set(keys)):
            reason = "duplicate planned row"
        elif set(keys) != expected_keys(system):
            reason = "missing or unexpected planned row"
        elif any(
            row["status"] != "ok" or row["exit_code"] != "0"
            for row in cell
        ):
            reason = "non-zero or malformed invocation"
        elif len({row["machine"] for row in cell}) != 1:
            reason = "more than one machine"
        for arm in ("B", "G"):
            summary.append(
                incomplete_row(family, system, arm, reason)
                if reason
                else analyze_cell(family, system, cell, arm)
            )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    args.report.write_text(render_report(summary, machines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
