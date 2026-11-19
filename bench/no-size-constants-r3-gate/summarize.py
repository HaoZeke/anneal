#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the no-size-constants-r3 gate CSVs and write the verdict report.

B loses when any measured system has fewer hits at two-sided Fisher exact
p < 0.05, more calls at tie-corrected two-sided Mann-Whitney p < 0.05, or a
median outside-potential time ratio B/A above 1.1. Otherwise the verdict is
PASS. The report lists every system with a hit drop, a call increase, or an
outside ratio above 1, including those that miss the loss thresholds.
"""

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
    "outside_ratio",
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


def outside_ratio(out_a: float, out_b: float) -> float:
    if out_a > 0.0:
        return out_b / out_a
    if out_b == 0.0:
        return 1.0
    return float("inf")


def is_loss(hit_drop: bool, fisher_p: float, more_calls: bool, mw_p: float, ratio: float) -> bool:
    return (hit_drop and fisher_p < 0.05) or (more_calls and mw_p < 0.05) or (ratio > 1.1)


def fmt_ratio(ratio: float) -> str:
    if math.isfinite(ratio):
        return f"{ratio:.6g}"
    return "inf"


def ratio_above_one(text: str) -> bool:
    if text == "inf":
        return True
    if text == "":
        return False
    return float(text) > 1.0


FOCUS = ("tip4p12", "tip4p20", "rq3", "xld", "lj103", "lj104", "lj98")


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
                    "outside_ratio": "",
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
        pot_a = type7([float(r["pot_s"]) for r in rows_a])
        pot_b = type7([float(r["pot_s"]) for r in rows_b])
        out_a = type7([float(r["outside_s"]) for r in rows_a])
        out_b = type7([float(r["outside_s"]) for r in rows_b])
        wall_a = type7([float(r["wall_s"]) for r in rows_a])
        wall_b = type7([float(r["wall_s"]) for r in rows_b])
        ratio = outside_ratio(out_a, out_b)
        b_more_rank = u_b_vs_a > (n * n) / 2.0
        hit_drop = hits_b < hits_a
        call_inc = (med_b > med_a) or b_more_rank
        fail = is_loss(hit_drop, f_p, b_more_rank, mw_p, ratio)
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
                "pot_median_s_a": f"{pot_a:.6f}",
                "pot_median_s_b": f"{pot_b:.6f}",
                "outside_median_s_a": f"{out_a:.6f}",
                "outside_median_s_b": f"{out_b:.6f}",
                "outside_ratio": fmt_ratio(ratio),
                "wall_median_s_a": f"{wall_a:.6f}",
                "wall_median_s_b": f"{wall_b:.6f}",
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

    expected = [system for panel, system in SYSTEM_ORDER if panel != "binary-lj"]
    by_name = {row["system"]: row for row in summary_rows}
    complete = all(by_name.get(system, {}).get("status") == "measured" and by_name[system]["n"] == "12" for system in expected)
    supported = by_name.get("binary_lj", {}).get("status") == "unsupported"
    if complete and supported and any_fail:
        verdict = "B LOSES"
    elif complete and supported:
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"
    return summary_rows, identity_rows, verdict


def _cell(text: str) -> str:
    return text if text else "-"


def write_report(path: Path, rows: list[dict[str, str]], summary_rows: list[dict[str, str]], verdict: str) -> None:
    def column(key: str) -> str:
        found = sorted({row[key] for row in rows if row.get(key)})
        return " ".join(found)

    seeds = sorted({int(row["seed"]) for row in rows}) if rows else []
    seed_text = f"{seeds[0]}-{seeds[-1]} ({len(seeds)} values)" if seeds else "none"
    lines = [
        "# No-size-constants-r3 regression gate",
        "",
        f"Paired head-to-head of arm A (`{column('a_revision')}`, tree `{column('a_tree')}`) and arm B (`{column('b_revision')}`, tree `{column('b_tree')}`).",
        "",
        "## Protocol",
        "",
        "- Panel, presets, budgets and seeds are the no-size-constants-b2 gate.",
        "- Every supported system is n=12. That gate's measured rows are n=12 on every supported system, and this run keeps that count, fixed before any result.",
        f"- Seeds present in `rows.csv`: {seed_text}.",
        "- LJ and Morse use `Config::recommended`. TIP4P uses `Config::recommended_tip4p`. The slab arm is `recommended`. Conformers use `cluster_search` with `recommended=True`.",
        "- Paper budgets are the `budget` value on each summary row.",
        "- Runs are serial, one fresh process per arm and seed, with no warm-up and no rescue rerun.",
        "- Even seeds run A then B. Odd seeds run B then A.",
        "- OpenMP, OpenBLAS, MKL and Rayon threads are 1. Each process is pinned to CPU core 0.",
        "- Both arms are release builds. Every row records both commit SHAs, both trees, and a hashed host id.",
        "- Binary LJ is unsupported, as in the b2 gate.",
        f"- Hashed host id: `{column('host')}`.",
        "",
        f"## Verdict: `{verdict}`",
        "",
        "B loses if any system has fewer hits at two-sided Fisher exact p < 0.05, more calls at tie-corrected two-sided Mann-Whitney p < 0.05, or a median outside-potential time ratio B/A above 1.1. Otherwise the verdict is PASS.",
        "",
    ]
    losers = [row for row in summary_rows if row.get("gate_fail") == "1"]
    if verdict == "PASS":
        lines.append("No measured system meets a loss condition.")
    elif verdict == "B LOSES":
        lines.append("Loss conditions:")
        lines.append("")
        for row in losers:
            why = []
            if row["hit_drop"] == "1" and float(row["fisher_p"]) < 0.05:
                why.append(f"fewer hits, Fisher p = {row['fisher_p']}")
            if row["b_more_calls_rank"] == "1" and float(row["mannwhitney_p"]) < 0.05:
                why.append(f"more calls, Mann-Whitney p = {row['mannwhitney_p']}")
            if ratio_above_one(row["outside_ratio"]) and (row["outside_ratio"] == "inf" or float(row["outside_ratio"]) > 1.1):
                why.append(f"outside ratio B/A = {row['outside_ratio']}")
            lines.append(f"- `{row['system']}`: " + "; ".join(why) + ".")
    else:
        lines.append("The measured panel is short of 29 supported systems at n=12, so this file is not a gate verdict.")
    lines.extend(["", "## Focus", ""])
    lines.append(
        "The b2 losses and the size-sensitive rows are tip4p12 (calls), tip4p20 (hits), rq3 and xld (hits and outside time), lj103, lj104 and lj98."
    )
    lines.append("")
    by_name = {row["system"]: row for row in summary_rows}
    for name in FOCUS:
        row = by_name.get(name)
        if row is None or row["status"] != "measured":
            lines.append(f"### {name}")
            lines.append("")
            lines.append("No measured rows.")
            lines.append("")
            continue
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"Hits {row['hits_a']} (A) and {row['hits_b']} (B), Fisher p = {row['fisher_p']}. "
            f"Median calls {row['calls_median_a']} (A) and {row['calls_median_b']} (B), "
            f"Mann-Whitney U = {row['mannwhitney_u']}, p = {row['mannwhitney_p']}. "
            f"Median time inside the potential {row['pot_median_s_a']} s (A) and {row['pot_median_s_b']} s (B). "
            f"Median time outside the potential {row['outside_median_s_a']} s (A) and {row['outside_median_s_b']} s (B), "
            f"ratio B/A = {row['outside_ratio']}. "
            f"Bit-identical seeds {row['bit_identical_seeds']}."
        )
        lines.append("")
    lines.extend(["## Any hit drop, call increase, or outside ratio above 1", ""])
    notable = [
        row
        for row in summary_rows
        if row["status"] == "measured" and (row["hit_drop"] == "1" or row["call_increase"] == "1" or ratio_above_one(row["outside_ratio"]))
    ]
    if not notable:
        lines.append("No measured system has a hit drop, a call increase, or an outside ratio above 1.")
    else:
        for row in notable:
            lines.append(
                f"- `{row['system']}`: hits {row['hits_a']} -> {row['hits_b']} (Fisher p = {row['fisher_p']}), "
                f"median calls {row['calls_median_a']} -> {row['calls_median_b']} "
                f"(U = {row['mannwhitney_u']}, p = {row['mannwhitney_p']}), "
                f"outside ratio B/A = {row['outside_ratio']}, gate_fail = {row['gate_fail']}."
            )
    lines.extend(["", "## Per-system summary", ""])
    lines.append(
        "| Panel | System | Status | n | Hits A | Hits B | Fisher p | Med calls A | Med calls B | MW p | In (s) A | In (s) B | Out (s) A | Out (s) B | Out B/A | Bit-ID |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        lines.append(
            "| {panel} | {system} | {status} | {n} | {hits_a} | {hits_b} | {fisher_p} | {calls_a} | {calls_b} | {mw} | {in_a} | {in_b} | {out_a} | {out_b} | {ratio} | {bit} |".format(
                panel=row["panel"],
                system=row["system"],
                status=row["status"],
                n=_cell(row["n"]),
                hits_a=_cell(row["hits_a"]),
                hits_b=_cell(row["hits_b"]),
                fisher_p=_cell(row["fisher_p"]),
                calls_a=_cell(row["calls_median_a"]),
                calls_b=_cell(row["calls_median_b"]),
                mw=_cell(row["mannwhitney_p"]),
                in_a=_cell(row["pot_median_s_a"]),
                in_b=_cell(row["pot_median_s_b"]),
                out_a=_cell(row["outside_median_s_a"]),
                out_b=_cell(row["outside_median_s_b"]),
                ratio=_cell(row["outside_ratio"]),
                bit=_cell(row["bit_identical_seeds"]),
            )
        )
    lines.extend(
        [
            "",
            "Per-seed bit identity (trace hash, charged count, best energy, hit and calls) is in `identity.csv`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


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
    assert is_loss(True, 0.04, False, 1.0, 1.0)
    assert not is_loss(True, 0.06, False, 1.0, 1.0)
    assert is_loss(False, 1.0, True, 0.04, 1.0)
    assert not is_loss(False, 1.0, True, 0.06, 1.0)
    assert not is_loss(False, 1.0, False, 0.01, 1.0)
    assert is_loss(False, 1.0, False, 1.0, 1.1000001)
    assert not is_loss(False, 1.0, False, 1.0, 1.1)
    assert abs(outside_ratio(10.0, 11.0) - 1.1) < 1e-12
    assert outside_ratio(0.0, 0.0) == 1.0
    assert ratio_above_one("1.00001")
    assert not ratio_above_one("1")
    assert not ratio_above_one("0.9")
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
    parser.add_argument("--report", type=Path, default=here / "README.md")
    args = parser.parse_args(argv[1:])

    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary_rows, identity_rows, verdict = summarise(rows)
    write_csv(args.summary, SUMMARY_FIELDS, summary_rows)
    write_csv(args.identity, IDENTITY_FIELDS, identity_rows)
    write_report(args.report, rows, summary_rows, verdict)
    print(f"verdict={verdict}")
    for r in summary_rows:
        if r["status"] == "measured" and (
            r["hit_drop"] == "1" or r["call_increase"] == "1" or r["gate_fail"] == "1" or ratio_above_one(r["outside_ratio"])
        ):
            print(
                f"flag {r['system']}: hits {r['hits_a']}->{r['hits_b']} (p={r['fisher_p']}) "
                f"calls {r['calls_median_a']}->{r['calls_median_b']} (p={r['mannwhitney_p']}) "
                f"outside={r['outside_ratio']} fail={r['gate_fail']}"
            )
    return 0 if verdict in {"PASS", "B LOSES"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
