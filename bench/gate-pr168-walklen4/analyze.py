#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Verdict for the LJ98 walklen4 gate, shard A.

The rules below are the bar. A result is not renamed after it is computed.

PASS requires every one of these, on a complete valid measurement:
- no hit loss: hits(B) >= hits(A)
- no significant calls-to-hit loss: two-sided Mann-Whitney p >= 0.05,
  or B is not the slower sample. Misses are scored as budget + 1.
- outside-time ratio: the 95 percent bootstrap upper bound of the
  seed-paired geometric mean of B/A outside-potential CPU time is
  at most 1.10. Only seeds where both arms hit enter that ratio.
  The same bootstrap is reported for calls-to-hit and is not a bar.

FIX when the measurement is valid and any of these is clear:
- hits(B) < hits(A) and two-sided Fisher exact p < 0.05
- two-sided Mann-Whitney p < 0.05 and B's calls are the larger sample
- the outside upper bound is above 1.10 and either the lower bound is
  above 1 or the geometric mean itself is above 1.10

UNRESOLVED otherwise. That includes an invalid measurement, a hit
count that falls without a significant Fisher test, an outside
interval whose upper bound exceeds 1.10 while the geometric mean
stays at or below 1.10 and the interval still reaches 1 or below,
and a ratio that cannot be computed.
"""

from __future__ import annotations

import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

from protocol import (
    ALPHA,
    ARM_A_REV,
    ARM_A_TREE,
    ARM_B_TIP,
    ARM_B_TREE,
    BOOT_DRAWS,
    BOOT_SEED,
    BUDGET,
    CORE,
    MISS_CALLS,
    N_SEEDS,
    OUTSIDE_CAP,
    SEEDS,
    schedule,
)

HERE = Path(__file__).resolve().parent


def geomean_ci(ratios: np.ndarray) -> tuple[float, float, float] | None:
    """Geometric mean and percentile bootstrap interval. Own stream per call."""
    if ratios.size < 2:
        return None
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        return None
    logs = np.log(ratios.astype(float))
    point = float(np.exp(logs.mean()))
    rng = np.random.default_rng(BOOT_SEED)
    drawn = rng.integers(0, ratios.size, size=(BOOT_DRAWS, ratios.size))
    means = np.exp(logs[drawn].mean(axis=1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def _num(text: str) -> float:
    return float(text)


def measurement_problems(rows: list[dict]) -> list[str]:
    problems = []
    jobs = schedule(SEEDS)
    if len(rows) != len(jobs):
        problems.append(f"row count {len(rows)} != {len(jobs)}")
        return problems
    by_seq = {}
    seen = set()
    for row in rows:
        key = (row["arm"], int(row["seed"]))
        if key in seen:
            problems.append(f"duplicate {key}")
        seen.add(key)
        by_seq[int(row["seq"])] = row
    if len(by_seq) != len(jobs):
        problems.append("seq is not a permutation of the schedule")
        return problems
    for job in jobs:
        row = by_seq[job["seq"]]
        if row["arm"] != job["arm"] or int(row["seed"]) != job["seed"]:
            problems.append(f"seq {job['seq']} is not the ABBA job")
            break
        if int(row["exit_code"]) != 0:
            problems.append(f"seq {job['seq']} exit {row['exit_code']}")
        if int(row["budget"]) != BUDGET:
            problems.append(f"seq {job['seq']} budget")
        if int(row["core"]) != CORE or row["core_observed"] != str(CORE):
            problems.append(f"seq {job['seq']} core")
        if int(row["best_agrees"]) != 1 or int(row["calls_agrees"]) != 1:
            problems.append(f"seq {job['seq']} score disagrees with the harness")
        hit = int(row["hit"])
        calls = int(row["calls_to_hit"])
        if hit not in (0, 1):
            problems.append(f"seq {job['seq']} hit flag")
        elif hit == 0 and calls != MISS_CALLS:
            problems.append(f"seq {job['seq']} miss calls")
        elif hit == 1 and not (1 <= calls <= BUDGET):
            problems.append(f"seq {job['seq']} hit calls")
        for name in ("outside_cpu_s", "cpu_s", "wall_s", "pot_s"):
            value = _num(row[name])
            if not math.isfinite(value):
                problems.append(f"seq {job['seq']} {name}")
        if _num(row["cpu_s"]) < 0 or _num(row["wall_s"]) < 0:
            problems.append(f"seq {job['seq']} negative time")
        if int(row["peak_rss_kb"]) <= 0:
            problems.append(f"seq {job['seq']} rss")
    arms = {arm: [r for r in rows if r["arm"] == arm] for arm in ("A", "B")}
    if {r["code_revision"] for r in arms["A"]} != {ARM_A_REV}:
        problems.append("arm A revision")
    if {r["code_tree"] for r in arms["A"]} != {ARM_A_TREE}:
        problems.append("arm A tree")
    if {r["code_revision"] for r in arms["B"]} != {ARM_B_TIP}:
        problems.append("arm B revision")
    if {r["code_tree"] for r in arms["B"]} != {ARM_B_TREE}:
        problems.append("arm B tree")
    if {r["patch_tip"] for r in arms["B"]} != {ARM_B_TIP}:
        problems.append("arm B tip")
    if len({r["host_id_hash"] for r in rows}) != 1:
        problems.append("host id hash is not constant")
    if len({r["binary_sha256"] for r in arms["A"]}) != 1:
        problems.append("arm A binary id")
    if len({r["binary_sha256"] for r in arms["B"]}) != 1:
        problems.append("arm B binary id")
    if arms["A"][0]["binary_sha256"] == arms["B"][0]["binary_sha256"]:
        problems.append("arm binaries are identical")
    if len({r["seed"] for r in arms["A"]}) != N_SEEDS:
        problems.append("seed coverage")
    return problems


def _index(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out = {"A": {}, "B": {}}
    for row in rows:
        out[row["arm"]][int(row["seed"])] = row
    return out


def judge(rows: list[dict]) -> dict:
    problems = measurement_problems(rows)
    result = {
        "valid": not problems,
        "problems": problems,
        "n": N_SEEDS,
        "alpha": ALPHA,
        "outside_cap": OUTSIDE_CAP,
        "boot_seed": BOOT_SEED,
        "boot_draws": BOOT_DRAWS,
    }
    if problems:
        result.update(
            verdict="UNRESOLVED",
            hit_status="unavailable",
            calls_status="unavailable",
            outside_status="unavailable",
            hits_a=None,
            hits_b=None,
            fisher_p=None,
            fisher_less_p=None,
            mwu_p=None,
            mwu_u=None,
            calls_ratio=None,
            calls_lo=None,
            calls_hi=None,
            outside_ratio=None,
            outside_lo=None,
            outside_hi=None,
            both_hit=None,
            hit_seeds_differ="",
        )
        return result
    indexed = _index(rows)
    seeds = list(SEEDS)
    hits_a = sum(int(indexed["A"][s]["hit"]) for s in seeds)
    hits_b = sum(int(indexed["B"][s]["hit"]) for s in seeds)
    table = [[hits_b, N_SEEDS - hits_b], [hits_a, N_SEEDS - hits_a]]
    fisher_p = float(fisher_exact(table, alternative="two-sided")[1])
    fisher_less_p = float(fisher_exact(table, alternative="less")[1])
    calls_a = np.array([int(indexed["A"][s]["calls_to_hit"]) for s in seeds], dtype=float)
    calls_b = np.array([int(indexed["B"][s]["calls_to_hit"]) for s in seeds], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mwu = mannwhitneyu(calls_b, calls_a, alternative="two-sided", method="asymptotic")
    mwu_p = float(mwu.pvalue)
    mwu_u = float(mwu.statistic)
    b_slower = mwu_u > (N_SEEDS * N_SEEDS / 2.0)
    if hits_b < hits_a and fisher_p < ALPHA:
        hit_status = "clear_fail"
    elif hits_b < hits_a:
        hit_status = "unclear"
    else:
        hit_status = "pass"
    if mwu_p < ALPHA and b_slower:
        calls_status = "clear_fail"
    else:
        calls_status = "pass"
    both = [s for s in seeds if int(indexed["A"][s]["hit"]) and int(indexed["B"][s]["hit"])]
    call_ratios = np.array(
        [int(indexed["B"][s]["calls_to_hit"]) / int(indexed["A"][s]["calls_to_hit"]) for s in both],
        dtype=float,
    )
    outside_ratios = np.array(
        [
            _num(indexed["B"][s]["outside_cpu_s"]) / _num(indexed["A"][s]["outside_cpu_s"])
            for s in both
        ],
        dtype=float,
    )
    calls_ci = geomean_ci(call_ratios)
    outside_ci = geomean_ci(outside_ratios)
    if outside_ci is None:
        outside_status = "unavailable"
    else:
        point, lo, hi = outside_ci
        if hi <= OUTSIDE_CAP:
            outside_status = "pass"
        elif lo > 1.0 or point > OUTSIDE_CAP:
            outside_status = "clear_fail"
        else:
            outside_status = "unclear"
    differ = [
        str(s)
        for s in seeds
        if int(indexed["A"][s]["hit"]) != int(indexed["B"][s]["hit"])
    ]
    clear = [name for name, status in (
        ("hit", hit_status),
        ("calls", calls_status),
        ("outside", outside_status),
    ) if status == "clear_fail"]
    if clear:
        verdict = "FIX"
    elif hit_status == "pass" and calls_status == "pass" and outside_status == "pass":
        verdict = "PASS"
    else:
        verdict = "UNRESOLVED"
    result.update(
        verdict=verdict,
        hit_status=hit_status,
        calls_status=calls_status,
        outside_status=outside_status,
        hits_a=hits_a,
        hits_b=hits_b,
        fisher_p=fisher_p,
        fisher_less_p=fisher_less_p,
        mwu_p=mwu_p,
        mwu_u=mwu_u,
        b_slower=b_slower,
        calls_ratio=None if calls_ci is None else calls_ci[0],
        calls_lo=None if calls_ci is None else calls_ci[1],
        calls_hi=None if calls_ci is None else calls_ci[2],
        outside_ratio=None if outside_ci is None else outside_ci[0],
        outside_lo=None if outside_ci is None else outside_ci[1],
        outside_hi=None if outside_ci is None else outside_ci[2],
        both_hit=len(both),
        hit_seeds_differ=" ".join(differ),
        clear=" ".join(clear),
    )
    return result


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "na"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_report(rows: list[dict], result: dict) -> str:
    host = rows[0]["host_id_hash"] if rows else "na"
    core = rows[0]["core"] if rows else "na"
    lines = [
        "# LJ98 walklen4 gate, shard A",
        "",
        f"Arm A revision `{ARM_A_REV}` tree `{ARM_A_TREE}`.",
        f"Arm B revision `{ARM_B_TIP}` built tree `{ARM_B_TREE}`.",
        "B is `git merge-tree --write-tree` of A and that tip.",
        "The tip's parent is `f75836be832c0f953f99c26efb9cd5c4b87565b0`.",
        "The built tree adds `paper/related-work.md` from A; the search sources are the tip.",
        "",
        "Workload: LJ98, `h2h_seed lj 98 0 4000000 <seed> recommended -543.664361 <prefix>`.",
        "Budget 4000000. Seeds 2048-2083 (n=36). One core, one process, ABBA order.",
        "A hit is energy <= -543.665361 + 0.001. A miss scores 4000001 calls.",
        "Outside-potential CPU time is process CPU (`wait4` user+system) minus harness `pot_s`.",
        "Ratios use seeds where both arms hit. Bootstrap 10000 resamples, seed 20261123,",
        "each ratio with its own generator. Mann-Whitney is two-sided, asymptotic, tie-corrected.",
        "Fisher exact is two-sided. GMIN was not run.",
        "",
        f"Host id hash `{host}`. Core `{core}`.",
        "",
        "## Verdict",
        "",
        f"**{result['verdict']}**",
        "",
        "| check | value | bar | call |",
        "| --- | --- | --- | --- |",
        f"| hits A | {result['hits_a']} / {N_SEEDS} |  |  |",
        f"| hits B | {result['hits_b']} / {N_SEEDS} | B >= A | {result['hit_status']} |",
        f"| Fisher two-sided p | {_fmt(result['fisher_p'])} | p < 0.05 and B fewer is a clear loss | {result['hit_status']} |",
        f"| Fisher one-sided less p | {_fmt(result['fisher_less_p'])} | reported |  |",
        f"| MWU two-sided p | {_fmt(result['mwu_p'])} | p < 0.05 and B slower is a clear loss | {result['calls_status']} |",
        f"| MWU U (B) | {_fmt(result['mwu_u'])} | null center {N_SEEDS * N_SEEDS / 2} |  |",
        f"| calls geomean B/A | {_fmt(result['calls_ratio'])} | both-hit only, reported |  |",
        f"| calls 95% CI | {_fmt(result['calls_lo'])} .. {_fmt(result['calls_hi'])} | reported |  |",
        f"| outside geomean B/A | {_fmt(result['outside_ratio'])} | both-hit only | {result['outside_status']} |",
        f"| outside 95% CI | {_fmt(result['outside_lo'])} .. {_fmt(result['outside_hi'])} | upper <= 1.10 | {result['outside_status']} |",
        f"| both-hit pairs | {result['both_hit']} |  |  |",
        f"| valid | {result['valid']} | complete ABBA measurement |  |",
        f"| verdict | {result['verdict']} | PASS, else FIX if a bar fails clearly, else UNRESOLVED | {result['verdict']} |",
        "",
    ]
    if result["problems"]:
        lines.append("Measurement problems:")
        lines.append("")
        for item in result["problems"]:
            lines.append(f"- {item}")
        lines.append("")
    differ = result.get("hit_seeds_differ") or ""
    lines.append("Seeds whose hit flags differ: " + (differ if differ else "none") + ".")
    lines.append("")
    if result["valid"]:
        indexed = _index(rows)
        both = [
            s
            for s in SEEDS
            if int(indexed["A"][s]["hit"]) and int(indexed["B"][s]["hit"])
        ]
        lines.append("## Descriptive")
        lines.append("")
        lines.append("| arm | hits | median calls (miss = 4000001) | median outside CPU on both-hit s | median wall s |")
        lines.append("| --- | --- | --- | --- | --- |")
        for arm in ("A", "B"):
            calls = sorted(int(indexed[arm][s]["calls_to_hit"]) for s in SEEDS)
            walls = sorted(_num(indexed[arm][s]["wall_s"]) for s in SEEDS)
            outs = sorted(_num(indexed[arm][s]["outside_cpu_s"]) for s in both)
            lines.append(
                f"| {arm} | {sum(int(indexed[arm][s]['hit']) for s in SEEDS)} "
                f"| {_median(calls):.1f} | {_median(outs):.6f} | {_median(walls):.6f} |"
            )
        lines.append("")
        lines.append(f"Both-hit seeds: {len(both)}.")
        lines.append("")
    lines.append("## Rules")
    lines.append("")
    lines.append("PASS needs no hit loss, no significant calls-to-hit loss, and an outside-time")
    lines.append("upper confidence bound of at most 1.10. FIX if any of those fails clearly.")
    lines.append("UNRESOLVED otherwise. The verdict is the table value and is not relabelled.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_report(rows_path: Path, report_path: Path) -> dict:
    rows = load_rows(rows_path)
    result = judge(rows)
    report_path.write_text(render_report(rows, result))
    return result


def _base_row(job: dict, hit: int, calls: int, outside: float) -> dict:
    arm = job["arm"]
    revision, tree, tip = (
        (ARM_A_REV, ARM_A_TREE, "") if arm == "A" else (ARM_B_TIP, ARM_B_TREE, ARM_B_TIP)
    )
    best = "-543.6654000000" if hit else "-540.0000000000"
    return {
        "arm": arm,
        "seed": str(job["seed"]),
        "hit": str(hit),
        "calls_to_hit": str(calls),
        "outside_cpu_s": f"{outside:.6f}",
        "cpu_s": "20.000000",
        "wall_s": "21.000000",
        "peak_rss_kb": "20000",
        "code_revision": revision,
        "code_tree": tree,
        "host_id_hash": "ab" * 32,
        "core": "0",
        "core_observed": "0",
        "budget": str(BUDGET),
        "pot_s": "10.000000",
        "harness_outside_s": f"{outside:.6f}",
        "harness_wall_s": "20.000000",
        "best": best,
        "charged": str(calls if hit else BUDGET),
        "potential_calls": str(calls if hit else BUDGET - 100),
        "best_agrees": "1",
        "calls_agrees": "1",
        "exit_code": "0",
        "seq": str(job["seq"]),
        "patch_tip": tip,
        "binary_sha256": "a" * 64 if arm == "A" else "b" * 64,
        "governor": "",
    }


def self_check() -> None:
    jobs = schedule()
    assert [job["arm"] for job in jobs[:8]] == ["A", "B", "B", "A", "A", "B", "B", "A"]
    assert len(jobs) == 72

    def run(mutate) -> str:
        rows = []
        for job in jobs:
            hit, calls, outside = 1, 1000, 10.0
            if job["arm"] == "B":
                hit, calls, outside = mutate(job, hit, calls, outside)
            rows.append(_base_row(job, hit, calls, outside))
        verdict = judge(rows)["verdict"]
        return verdict

    assert run(lambda job, hit, calls, outside: (hit, calls, outside)) == "PASS"
    assert run(lambda job, hit, calls, outside: (hit, calls, 12.5)) == "FIX"
    assert run(lambda job, hit, calls, outside: (hit, calls, 10.5)) == "PASS"

    def drop_six(job, hit, calls, outside):
        if job["arm"] == "B" and job["seed"] < 2048 + 6:
            return 0, MISS_CALLS, outside
        return hit, calls, outside

    assert run(drop_six) == "FIX"

    def drop_one(job, hit, calls, outside):
        if job["arm"] == "B" and job["seed"] == 2048:
            return 0, MISS_CALLS, outside
        return hit, calls, outside

    assert run(drop_one) == "UNRESOLVED"
    identical = geomean_ci(np.ones(36))
    assert identical is not None and identical[0] == 1.0 and identical[2] <= OUTSIDE_CAP
    again = geomean_ci(np.ones(36))
    assert again == identical


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-check":
        self_check()
        print("self-check ok")
        return 0
    rows_path = HERE / "rows.csv"
    report_path = HERE / "REPORT.md"
    if not rows_path.is_file():
        print("rows.csv is missing", file=sys.stderr)
        return 1
    result = write_report(rows_path, report_path)
    print(result["verdict"])
    print(
        "hits",
        result["hits_a"],
        result["hits_b"],
        "fisher",
        _fmt(result["fisher_p"]),
        "mwu",
        _fmt(result["mwu_p"]),
        "outside",
        _fmt(result["outside_ratio"]),
        _fmt(result["outside_lo"]),
        _fmt(result["outside_hi"]),
        "both",
        result["both_hit"],
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
