#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Verdicts for the LJ75 and LJ38 shard B gate.

Fisher's exact test on hits is two-sided. The Mann-Whitney test on
calls to hit is two-sided, with the tie and continuity corrections,
and a miss is budget + 1. Geometric means are paired B/A on seeds
where both arms hit, with a 10000-draw bootstrap and seed 20261123.
A system PASSes when B loses no hits, calls-to-hit show no significant
loss, and the outside-time upper interval is at most 1.10. A clear
failure is FIX. Anything else is UNRESOLVED.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import fisher_exact, mannwhitneyu

BOOT_SEED = 20261123
BOOT_DRAWS = 10_000
ALPHA = 0.05
CAP = 1.10
HERE = Path(__file__).resolve().parent


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def geomean_ci(ratios: np.ndarray) -> tuple[float, float, float]:
    logs = np.log(ratios)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(logs), size=(BOOT_DRAWS, len(logs)))
    means = np.exp(logs[idx].mean(axis=1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.exp(logs.mean())), float(lo), float(hi)


def mwu(left: list[float], right: list[float]) -> tuple[float, float, bool]:
    """U of ``left``, two-sided p, and whether ``left`` tends to be larger."""
    if len(set(left + right)) == 1:
        return 0.5 * len(left) * len(right), 1.0, False
    result = mannwhitneyu(
        left, right, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    expected = len(left) * len(right) / 2.0
    return float(result.statistic), float(result.pvalue), float(result.statistic) > expected


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    table = [[hits_a, n_a - hits_a], [hits_b, n_b - hits_b]]
    return float(fisher_exact(table, alternative="two-sided")[1])


def paired(rows_a: dict[int, dict[str, str]], rows_b: dict[int, dict[str, str]], key: str) -> np.ndarray:
    values = []
    for seed in sorted(set(rows_a) & set(rows_b)):
        if rows_a[seed]["hit"] != "1" or rows_b[seed]["hit"] != "1":
            continue
        left = float(rows_b[seed][key])
        right = float(rows_a[seed][key])
        if not (left > 0.0 and right > 0.0 and math.isfinite(left) and math.isfinite(right)):
            raise ValueError(f"non-positive {key} at seed {seed}")
        values.append(left / right)
    return np.asarray(values, dtype=float)


def judge(system: str, rows: list[dict[str, str]]) -> dict:
    group = [row for row in rows if row["system"] == system]
    by_arm = {
        arm: {int(row["seed"]): row for row in group if row["arm"] == arm} for arm in ("A", "B")
    }
    seeds = set(range(2120, 2132))
    complete = set(by_arm["A"]) == seeds and set(by_arm["B"]) == seeds
    result: dict = {
        "system": system,
        "complete": complete,
        "n_a": len(by_arm["A"]),
        "n_b": len(by_arm["B"]),
    }
    if not complete:
        result["verdict"] = "UNRESOLVED"
        result["note"] = "missing seeds"
        return result
    ordered = sorted(seeds)
    hits_a = sum(by_arm["A"][seed]["hit"] == "1" for seed in ordered)
    hits_b = sum(by_arm["B"][seed]["hit"] == "1" for seed in ordered)
    calls_a = [float(by_arm["A"][seed]["calls_to_hit"]) for seed in ordered]
    calls_b = [float(by_arm["B"][seed]["calls_to_hit"]) for seed in ordered]
    p_fisher = fisher(hits_a, len(ordered), hits_b, len(ordered))
    u_stat, p_mwu, b_slower = mwu(calls_b, calls_a)
    result.update(
        hits_a=hits_a,
        hits_b=hits_b,
        fisher_p=p_fisher,
        mwu_u=u_stat,
        mwu_p=p_mwu,
        b_slower=b_slower,
    )
    clear: list[str] = []
    soft: list[str] = []
    if hits_b < hits_a:
        if p_fisher < ALPHA:
            clear.append("hit loss")
        else:
            soft.append("fewer hits, Fisher p at least 0.05")
    if b_slower and p_mwu < ALPHA:
        clear.append("calls-to-hit loss")
    call_ratios = None
    out_ratios = None
    try:
        call_ratios = paired(by_arm["A"], by_arm["B"], "calls_to_hit")
    except ValueError as exc:
        soft.append(str(exc))
    try:
        out_ratios = paired(by_arm["A"], by_arm["B"], "outside_cpu_to_hit_s")
    except ValueError as exc:
        soft.append(str(exc))
    if call_ratios is not None:
        result["n_call_pairs"] = int(call_ratios.size)
        if call_ratios.size:
            result["calls_ratio"], result["calls_lo"], result["calls_hi"] = geomean_ci(call_ratios)
    if out_ratios is not None:
        result["n_out_pairs"] = int(out_ratios.size)
        if out_ratios.size:
            result["out_ratio"], result["out_lo"], result["out_hi"] = geomean_ci(out_ratios)
            if result["out_hi"] > CAP:
                if result["out_lo"] > CAP:
                    clear.append("outside-time ratio")
                else:
                    soft.append("outside upper CI above 1.10")
        else:
            soft.append("no paired first hits for outside time")
    if clear:
        result["verdict"] = "FIX"
    elif soft:
        result["verdict"] = "UNRESOLVED"
    else:
        result["verdict"] = "PASS"
    result["clear"] = clear
    result["soft"] = soft
    return result


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return ""
    if value < 1.0e-4:
        return f"{value:.6e}"
    return f"{value:.6f}"


def report(rows: list[dict[str, str]], judged: list[dict]) -> str:
    lines = [
        "# Walklen4 gate, shard B",
        "",
        "Arms are A (develop `4feb0699d4e61c9531283e82196fcb60baf66932`) and B",
        "(merge-tree of that commit and `ddd53f2f60d53aebdf2e97f03d3ade07c2905738`).",
        "LJ75 budget 4000000, reference -397.492331. LJ38 budget 400000,",
        "reference -173.928427. Seeds 2120-2131. Stop at the reference plus",
        "1e-3. Core 0. Misses count as budget + 1. Ratios use first hits only.",
        "Bootstrap: 10000 resamples, numpy Generator seed 20261123, percentiles",
        "2.5 and 97.5. Significant means two-sided p < 0.05.",
        "",
        "PASS: B has at least as many hits as A, calls-to-hit are not a",
        "significant loss for B, and the outside-time upper CI is <= 1.10.",
        "FIX: a hit loss with Fisher p < 0.05, a significant calls-to-hit loss",
        "(B slower), or an outside-time CI entirely above 1.10.",
        "UNRESOLVED: anything else.",
        "",
        "| system | hits A | hits B | Fisher p | MWU p | calls B/A | outside B/A | verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in judged:
        calls = ""
        outside = ""
        if "calls_ratio" in item:
            calls = (
                f"{fmt(item['calls_ratio'])} [{fmt(item['calls_lo'])}, {fmt(item['calls_hi'])}]"
                f" n={item['n_call_pairs']}"
            )
        if "out_ratio" in item:
            outside = (
                f"{fmt(item['out_ratio'])} [{fmt(item['out_lo'])}, {fmt(item['out_hi'])}]"
                f" n={item['n_out_pairs']}"
            )
        lines.append(
            "| {system} | {ha} | {hb} | {fisher} | {mwu} | {calls} | {outside} | {verdict} |".format(
                system=item["system"],
                ha=item.get("hits_a", ""),
                hb=item.get("hits_b", ""),
                fisher=fmt_p(item.get("fisher_p")),
                mwu=fmt_p(item.get("mwu_p")),
                calls=calls,
                outside=outside,
                verdict=item["verdict"],
            )
        )
    lines.extend(["", "## Notes", ""])
    for item in judged:
        detail = ", ".join(item.get("clear", []) + item.get("soft", [])) or "none"
        note = item.get("note", "")
        if note:
            detail = f"{detail}. {note}"
        lines.append(f"- {item['system']}: {item['verdict']}. {detail}.")
    if rows:
        host = sorted({row["host_id"] for row in rows})
        cores = sorted({row["core"] for row in rows})
        lines.extend(["", f"Host id {', '.join(host)}. Core {', '.join(cores)}.", f"Rows {len(rows)}."])
        for arm in ("A", "B"):
            sample = next(row for row in rows if row["arm"] == arm)
            if sample["revision_kind"] == "commit":
                ident = f"commit {sample['code_revision']} tree {sample['tree']}"
            else:
                ident = f"tree {sample['tree']}"
            lines.append(f"Arm {arm} {ident} binary {sample['binary_sha256']}.")
        lines.append(f"numpy {np.__version__}. scipy {scipy.__version__}.")
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    point = geomean_ci(np.ones(8))
    if point != (1.0, 1.0, 1.0):
        raise SystemExit(f"unit ratio {point}")
    again = geomean_ci(np.asarray([1.0, 2.0, 4.0]))
    if geomean_ci(np.asarray([1.0, 2.0, 4.0])) != again:
        raise SystemExit("bootstrap seed is not stable")
    if abs(again[0] - 2.0) > 1e-12:
        raise SystemExit(f"geomean {again}")
    if fisher(12, 12, 12, 12) != 1.0:
        raise SystemExit("fisher of equal hits")
    _u, p, slower = mwu([2.0] * 12, [1.0] * 12)
    if not (p < ALPHA and slower):
        raise SystemExit(f"mwu loss {p} {slower}")
    _u, p, slower = mwu([1.0] * 12, [1.0] * 12)
    if p != 1.0 or slower:
        raise SystemExit("mwu ties")
    print("ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=HERE / "rows-shardB.csv")
    parser.add_argument("--report", type=Path, default=HERE / "REPORT.md")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    rows = read_rows(args.rows)
    judged = [judge(system, rows) for system in ("lj75", "lj38")]
    text = report(rows, judged)
    args.report.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
