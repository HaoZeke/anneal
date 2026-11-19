#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the no-size-constants-b2 gate CSVs and evaluate the gate verdict."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

SYSTEM_ORDER = [
    ("lj", "lj38"),
    ("lj", "lj75"),
    ("lj", "lj98"),
    ("lj", "lj103"),
    ("lj", "lj104"),
    ("binary-lj", "binary_lj"),
    ("morse", "m75r3"),
    ("morse", "m75r6"),
    ("morse", "m75r10"),
    ("morse", "m75r14"),
    ("morse", "m80r3"),
    ("morse", "m80r6"),
    ("morse", "m80r10"),
    ("morse", "m80r14"),
    ("tip4p", "tip4p8"),
    ("tip4p", "tip4p12"),
    ("tip4p", "tip4p15"),
    ("tip4p", "tip4p16"),
    ("tip4p", "tip4p20"),
    ("tip4p", "tip4p21"),
    ("tip4p", "tip4p25"),
    ("slab", "cuh2_fcc_slab"),
    ("slab", "cuh2_fcc_slab_h6"),
    ("conformer", "rq3"),
    ("conformer", "915"),
    ("conformer", "xld"),
    ("conformer", "fluvastatin"),
    ("conformer", "z34"),
    ("conformer", "amprenavir"),
    ("conformer", "lopinavir"),
    ("conformer", "lorlatinib"),
]

SUMMARY_FIELDS = [
    "panel",
    "system",
    "status",
    "n",
    "budget",
    "hits_a",
    "hits_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "mannwhitney_u",
    "mannwhitney_p",
    "b_more_calls_rank",
    "pot_median_s_a",
    "pot_median_s_b",
    "outside_median_s_a",
    "outside_median_s_b",
    "wall_median_s_a",
    "wall_median_s_b",
    "bit_identical_seeds",
    "outcome_identical_seeds",
    "hit_drop",
    "call_increase",
    "gate_fail",
    "a_revision",
    "b_revision",
    "host",
]

IDENTITY_FIELDS = [
    "panel",
    "system",
    "seed",
    "bit_identical",
    "outcome_identical",
    "hit_a",
    "hit_b",
    "evals_a",
    "evals_b",
    "charged_a",
    "charged_b",
    "best_a",
    "best_b",
    "trace_sha256_a",
    "trace_sha256_b",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return float("nan")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-7)))


def mann_whitney_two_sided(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided U of ``left`` vs ``right``, normal approximation with tie & continuity corrections."""
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        raise ValueError("mann-whitney needs non-empty samples")
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def norm_best(value: str) -> str:
    try:
        return f"{float(value):.8f}"
    except ValueError:
        return value.strip()


def summarise(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    by_sys_arm: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        by_sys_arm.setdefault((row["system"], row["arm"]), {})[int(row["seed"])] = row

    summary_rows: list[dict[str, str]] = []
    identity_rows: list[dict[str, str]] = []
    any_fail = False

    for panel, system in SYSTEM_ORDER:
        if panel == "binary-lj":
            summary_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "status": "unsupported",
                    "n": "0",
                    "budget": "",
                    "hits_a": "",
                    "hits_b": "",
                    "fisher_p": "",
                    "calls_median_a": "",
                    "calls_median_b": "",
                    "mannwhitney_u": "",
                    "mannwhitney_p": "",
                    "b_more_calls_rank": "",
                    "pot_median_s_a": "",
                    "pot_median_s_b": "",
                    "outside_median_s_a": "",
                    "outside_median_s_b": "",
                    "wall_median_s_a": "",
                    "wall_median_s_b": "",
                    "bit_identical_seeds": "",
                    "outcome_identical_seeds": "",
                    "hit_drop": "0",
                    "call_increase": "0",
                    "gate_fail": "0",
                    "a_revision": "",
                    "b_revision": "",
                    "host": "",
                }
            )
            continue

        arm_a = by_sys_arm.get((system, "A"), {})
        arm_b = by_sys_arm.get((system, "B"), {})
        seeds = sorted(set(arm_a) & set(arm_b))
        if not seeds:
            continue
        rows_a = [arm_a[s] for s in seeds]
        rows_b = [arm_b[s] for s in seeds]
        n = len(seeds)
        hits_a = sum(int(r["hit"]) for r in rows_a)
        hits_b = sum(int(r["hit"]) for r in rows_b)
        calls_a = [float(r["evals_to_hit"]) for r in rows_a]
        calls_b = [float(r["evals_to_hit"]) for r in rows_b]
        f_p = fisher_two_sided(hits_a, n, hits_b, n)
        u_b_vs_a, mw_p = mann_whitney_two_sided(calls_b, calls_a)
        med_a = type7(calls_a, 0.5)
        med_b = type7(calls_b, 0.5)
        b_more_rank = u_b_vs_a > (n * n) / 2.0
        hit_drop = hits_b < hits_a
        call_inc = (med_b > med_a) or b_more_rank
        fail = (hit_drop and f_p < 0.05) or (b_more_rank and mw_p < 0.05)
        if fail:
            any_fail = True

        bit_id_count = 0
        out_id_count = 0
        for s in seeds:
            ra, rb = arm_a[s], arm_b[s]
            out_same = (
                ra["hit"] == rb["hit"]
                and ra["evals_to_hit"] == rb["evals_to_hit"]
                and ra["charged"] == rb["charged"]
                and norm_best(ra["best"]) == norm_best(rb["best"])
            )
            bit_same = (
                out_same
                and ra["best"] == rb["best"]
                and bool(ra["trace_sha256"])
                and ra["trace_sha256"] == rb["trace_sha256"]
            )
            bit_id_count += int(bit_same)
            out_id_count += int(out_same)
            identity_rows.append(
                {
                    "panel": panel,
                    "system": system,
                    "seed": str(s),
                    "bit_identical": str(int(bit_same)),
                    "outcome_identical": str(int(out_same)),
                    "hit_a": ra["hit"],
                    "hit_b": rb["hit"],
                    "evals_a": ra["evals_to_hit"],
                    "evals_b": rb["evals_to_hit"],
                    "charged_a": ra["charged"],
                    "charged_b": rb["charged"],
                    "best_a": ra["best"],
                    "best_b": rb["best"],
                    "trace_sha256_a": ra["trace_sha256"],
                    "trace_sha256_b": rb["trace_sha256"],
                }
            )

        summary_rows.append(
            {
                "panel": panel,
                "system": system,
                "status": "measured",
                "n": str(n),
                "budget": rows_a[0]["budget"],
                "hits_a": f"{hits_a}/{n}",
                "hits_b": f"{hits_b}/{n}",
                "fisher_p": f"{f_p:.6g}",
                "calls_median_a": f"{med_a:.1f}",
                "calls_median_b": f"{med_b:.1f}",
                "mannwhitney_u": f"{u_b_vs_a:.1f}",
                "mannwhitney_p": f"{mw_p:.6g}",
                "b_more_calls_rank": str(int(b_more_rank)),
                "pot_median_s_a": f"{type7([float(r['pot_s']) for r in rows_a]):.6f}",
                "pot_median_s_b": f"{type7([float(r['pot_s']) for r in rows_b]):.6f}",
                "outside_median_s_a": f"{type7([float(r['outside_s']) for r in rows_a]):.6f}",
                "outside_median_s_b": f"{type7([float(r['outside_s']) for r in rows_b]):.6f}",
                "wall_median_s_a": f"{type7([float(r['wall_s']) for r in rows_a]):.6f}",
                "wall_median_s_b": f"{type7([float(r['wall_s']) for r in rows_b]):.6f}",
                "bit_identical_seeds": f"{bit_id_count}/{n}",
                "outcome_identical_seeds": f"{out_id_count}/{n}",
                "hit_drop": str(int(hit_drop)),
                "call_increase": str(int(call_inc)),
                "gate_fail": str(int(fail)),
                "a_revision": " ".join(sorted({r["code_revision"] for r in rows_a})),
                "b_revision": " ".join(sorted({r["code_revision"] for r in rows_b})),
                "host": " ".join(sorted({r["host"] for r in rows_a + rows_b})),
            }
        )

    verdict = "B LOSES" if any_fail else "PASS"
    return summary_rows, identity_rows, verdict


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    assert abs(type7([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.5) - 3.5) < 1e-12
    assert abs(fisher_two_sided(12, 12, 0, 12) - 2.0 / math.comb(24, 12)) < 1e-15
    assert fisher_two_sided(6, 12, 6, 12) == 1.0
    u, p = mann_whitney_two_sided([10.0] * 12, [10.0] * 12)
    assert u == 72.0 and p == 1.0
    try:
        from scipy.stats import fisher_exact, mannwhitneyu
    except ImportError:
        pass
    else:
        left = [9.0, 4e6, 4e6, 120.0, 55.0, 4e6, 7.0, 3.0]
        right = [4e6, 4e6, 4e6, 900.0, 4e6, 4e6, 4e6, 4e6]
        want = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
        u, p = mann_whitney_two_sided(left, right)
        assert abs(u - want.statistic) < 1e-9 and abs(p - want.pvalue) < 1e-12
        assert abs(fisher_two_sided(9, 12, 2, 12) - fisher_exact([[9, 3], [2, 10]])[1]) < 1e-12
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--rows", type=Path, default=here / "rows.csv")
    parser.add_argument("--summary", type=Path, default=here / "summary.csv")
    parser.add_argument("--identity", type=Path, default=here / "identity.csv")
    args = parser.parse_args(argv[1:])

    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary_rows, identity_rows, verdict = summarise(rows)
    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.identity, IDENTITY_FIELDS, identity_rows)
    print(f"verdict={verdict}")
    for r in summary_rows:
        if r["hit_drop"] == "1" or r["call_increase"] == "1" or r["gate_fail"] == "1":
            print(
                f"flag {r['system']}: hits {r['hits_a']}->{r['hits_b']} (p={r['fisher_p']}) "
                f"calls {r['calls_median_a']}->{r['calls_median_b']} (p={r['mannwhitney_p']}) "
                f"fail={r['gate_fail']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
