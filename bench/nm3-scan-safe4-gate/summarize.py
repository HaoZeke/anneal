#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize Part 1 and Part 2 gate results and emit README.md."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PART1_CASES = (
    "detect_all:random33",
    "detect_all:random55",
    "detect_all:random38",
    "detect_all:ring55",
    "detect_all:packed80",
    "detect_orders:random33",
    "detect_orders:random55",
    "detect_orders:random38",
    "detect_orders:ring55",
    "detect_orders:packed80",
)

SEEDS = tuple(range(2000, 2012))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on 2x2 table [[a, b], [c, d]]."""
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def hypergeom(x: int) -> float:
        return math.exp(
            math.lgamma(row1 + 1)
            + math.lgamma(row2 + 1)
            + math.lgamma(col1 + 1)
            + math.lgamma(total - col1 + 1)
            - math.lgamma(total + 1)
            - math.lgamma(x + 1)
            - math.lgamma(row1 - x + 1)
            - math.lgamma(col1 - x + 1)
            - math.lgamma(row2 - (col1 - x) + 1)
        )

    p_obs = hypergeom(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p = sum(hypergeom(x) for x in range(lo, hi + 1) if hypergeom(x) <= p_obs + 1e-12)
    return min(1.0, p)


def mann_whitney_two_sided(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Two-sided tie-corrected, continuity-corrected normal-approx Mann-Whitney U.

    Returns (p_value, rank_sum_x, rank_sum_y).
    """
    n1 = len(x)
    n2 = len(y)
    combined = [(val, 0) for val in x] + [(val, 1) for val in y]
    combined.sort(key=lambda t: t[0])

    ranks_x = 0.0
    ranks_y = 0.0
    tie_term = 0.0
    i = 0
    n = n1 + n2
    while i < n:
        j = i + 1
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        count = j - i
        if count > 1:
            tie_term += count**3 - count
        for k in range(i, j):
            if combined[k][1] == 0:
                ranks_x += avg_rank
            else:
                ranks_y += avg_rank
        i = j

    u1 = ranks_x - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    mean_u = n1 * n2 / 2.0
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0.0:
        return 1.0, ranks_x, ranks_y
    diff = abs(u1 - mean_u)
    z = max(0.0, diff - 0.5) / math.sqrt(var_u)
    p = math.erfc(z / math.sqrt(2.0))
    return min(1.0, p), ranks_x, ranks_y


def summarize_part1(rows: list[dict[str, str]]) -> tuple[list[str], bool, list[str]]:
    slot_arm = Counter((int(r["slot"]), r["arm"]) for r in rows)
    counts_ok = len(set(slot_arm.values())) == 1 and len(slot_arm) == 16
    fails: list[str] = []
    if not counts_ok:
        fails.append(f"Part 1 (slot, arm) imbalance: {dict(sorted(slot_arm.items()))}")

    lines = [
        "## Part 1: Symmetry Detection Microbench",
        "",
        "Cases: `detect_all` and `detect_orders` on `random33`, `random55`, `random38` (`orders 2..=38`), `ring55`, and `packed80` (`orders 2..=12` for the others, `tolerance = 0.1`).",
        f"Rounds: `{len({int(r['round']) for r in rows})}` interleaved rounds across `BASE`, `A'`, `SAFE2`, `SAFE4` (`{len(rows)}` rows total; `(slot, arm)` count = `{sorted(set(slot_arm.values()))}`).",
        "",
        "| case | BASE (us) | A' (us) | SAFE2 (us) | SAFE4 (us) | SAFE4/BASE | SAFE2/BASE | SAFE4/SAFE2 | A'/BASE | bit-identical | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |",
    ]

    for case in PART1_CASES:
        sub = [r for r in rows if r["case"] == case]
        med = {}
        for arm in ("BASE", "A'", "SAFE2", "SAFE4"):
            vals = [float(r["per_call_s"]) for r in sub if r["arm"] == arm]
            med[arm] = statistics.median(vals)
        digests = {r["digest_hex"] for r in sub}
        counts = {r["candidate_count"] for r in sub}
        bit_id = all(r["bit_identical"] == "1" for r in sub) and len(digests) == 1 and len(counts) == 1

        s4_b = med["SAFE4"] / med["BASE"]
        s2_b = med["SAFE2"] / med["BASE"]
        s4_s2 = med["SAFE4"] / med["SAFE2"]
        ap_b = med["A'"] / med["BASE"]
        aa_upper = max(ap_b, 1.0 / ap_b)

        geom = case.split(":", 1)[1]
        case_ok = bit_id
        if geom in ("random33", "random55", "random38"):
            if s4_b > 1.05:
                case_ok = False
                fails.append(f"Part 1 {case}: SAFE4/BASE={s4_b:.4f} > 1.05")
        else:
            if s4_s2 > 1.00 and s4_s2 > aa_upper:
                case_ok = False
                fails.append(
                    f"Part 1 {case}: SAFE4/SAFE2={s4_s2:.4f} > 1.00 and outside A/A spread ({aa_upper:.4f})"
                )
        if not bit_id:
            fails.append(f"Part 1 {case}: outputs not bit-identical")

        verdict = "PASS" if case_ok else "FAIL"
        ap_med = med["A'"]
        lines.append(
            f"| `{case}` | {med['BASE']*1e6:.3f} | {ap_med*1e6:.3f} | {med['SAFE2']*1e6:.3f} | "
            f"{med['SAFE4']*1e6:.3f} | {s4_b:.4f} | {s2_b:.4f} | {s4_s2:.4f} | {ap_b:.4f} | "
            f"{'YES' if bit_id else 'NO'} | {verdict} |"
        )

    lines.append("")
    return lines, len(fails) == 0, fails


def summarize_part2(rows: list[dict[str, str]]) -> tuple[list[str], bool, list[str]]:
    slot_arm = Counter((int(r["slot"]), r["arm"]) for r in rows)
    counts_ok = len(set(slot_arm.values())) == 1 and len(slot_arm) == 9
    fails: list[str] = []
    if not counts_ok:
        fails.append(f"Part 2 (slot, arm) imbalance: {dict(sorted(slot_arm.items()))}")

    # Preserve cell order from rows
    systems: list[str] = []
    seen_sys: set[str] = set()
    by_sys_arm: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        sys_name = r["system"]
        if sys_name not in seen_sys:
            seen_sys.add(sys_name)
            systems.append(sys_name)
        by_sys_arm[(sys_name, r["arm"])][int(r["seed"])] = r

    lines = [
        "## Part 2: Default-On Panel (`PAPER_BUDGETS`, `H2H_*` unset)",
        "",
        f"Cells: `{len(systems)}` systems across LJ, Metal/Au, SC, Morse, BLJ, and TIP4P (`{len(rows)}` rows total; `(slot, arm)` count = `{sorted(set(slot_arm.values()))}`).",
        "Hit criterion: finite `best <= reference + 1e-3`. Misses ranked at `budget` for Mann-Whitney.",
        "`out/call` is computed per row as `outside_s / charged`; ratios compare per-cell medians (`SAFE4/BASE` vs `A'/BASE`).",
        "Note: `for_tip4p` disables `symmetrise_on_stall` (`symmetrise_on_stall = false`), so `tip4p8..21` serve as bit-identical panel controls alongside the active symmetrise cells.",
        "",
        "| cell | hits BASE / SAFE4 / A' | Fisher p | med calls BASE / SAFE4 / A' | MW p (calls) | out/call med BASE / SAFE4 / A' (us) | out/call SAFE4/BASE | out/call A'/BASE | bit-id (best_bits & charged) | verdict |",
        "| --- | :---: | ---: | :---: | ---: | :---: | ---: | ---: | :---: | :---: |",
    ]

    non_identical_notes: list[str] = []
    for sys_name in systems:
        b_rows = [by_sys_arm[(sys_name, "BASE")][s] for s in SEEDS]
        s4_rows = [by_sys_arm[(sys_name, "SAFE4")][s] for s in SEEDS]
        ap_rows = [by_sys_arm[(sys_name, "A'")][s] for s in SEEDS]

        hits_b = sum(int(r["hit"]) for r in b_rows)
        hits_s4 = sum(int(r["hit"]) for r in s4_rows)
        hits_ap = sum(int(r["hit"]) for r in ap_rows)
        fisher_p = fisher_exact_two_sided(hits_b, 12 - hits_b, hits_s4, 12 - hits_s4)

        calls_b = [float(int(r["budget"]) if int(r["hit"]) == 0 else int(r["calls"])) for r in b_rows]
        calls_s4 = [float(int(r["budget"]) if int(r["hit"]) == 0 else int(r["calls"])) for r in s4_rows]
        calls_ap = [float(int(r["budget"]) if int(r["hit"]) == 0 else int(r["calls"])) for r in ap_rows]
        mw_p, ranks_b, ranks_s4 = mann_whitney_two_sided(calls_b, calls_s4)
        med_cb = statistics.median(calls_b)
        med_cs4 = statistics.median(calls_s4)
        med_cap = statistics.median(calls_ap)

        opc_b = [float(r["outside_s"]) / int(r["charged"]) for r in b_rows]
        opc_s4 = [float(r["outside_s"]) / int(r["charged"]) for r in s4_rows]
        opc_ap = [float(r["outside_s"]) / int(r["charged"]) for r in ap_rows]
        med_ob = statistics.median(opc_b)
        med_os4 = statistics.median(opc_s4)
        med_oap = statistics.median(opc_ap)
        ratio_s4 = med_os4 / med_ob
        ratio_ap = med_oap / med_ob
        aa_upper = max(ratio_ap, 1.0 / ratio_ap)

        same_count = 0
        for s in SEEDS:
            rb = by_sys_arm[(sys_name, "BASE")][s]
            rs4 = by_sys_arm[(sys_name, "SAFE4")][s]
            same = rb["best_bits"] == rs4["best_bits"] and rb["charged"] == rs4["charged"]
            if same:
                same_count += 1
            else:
                non_identical_notes.append(
                    f"- `{sys_name}` seed `{s}`: BASE (`best={rb['best']}`, `best_bits={rb['best_bits']}`, `charged={rb['charged']}`) vs "
                    f"SAFE4 (`best={rs4['best']}`, `best_bits={rs4['best_bits']}`, `charged={rs4['charged']}`)"
                )

        cell_fails: list[str] = []
        if hits_s4 < hits_b:
            cell_fails.append(f"fewer hits ({hits_s4} < {hits_b})")
        if mw_p < 0.05 and ranks_s4 > ranks_b:
            cell_fails.append(f"calls loss (MW p={mw_p:.4g})")
        if ratio_s4 > 1.10 and ratio_s4 > aa_upper:
            cell_fails.append(f"out/call SAFE4/BASE={ratio_s4:.4f} > 1.10 and > A'/A spread ({aa_upper:.4f})")

        if cell_fails:
            fails.append(f"Part 2 {sys_name}: " + "; ".join(cell_fails))

        verdict = "PASS" if not cell_fails else "FAIL"
        lines.append(
            f"| `{sys_name}` | {hits_b}/{hits_s4}/{hits_ap} | {fisher_p:.4g} | "
            f"{med_cb:.0f}/{med_cs4:.0f}/{med_cap:.0f} | {mw_p:.4g} | "
            f"{med_ob*1e6:.3f}/{med_os4*1e6:.3f}/{med_oap*1e6:.3f} | "
            f"{ratio_s4:.4f} | {ratio_ap:.4f} | {same_count}/12 | {verdict} |"
        )

    lines.append("")
    lines.append("### Bit-Identity Audit (`SAFE4` vs `BASE`)")
    lines.append("")
    if non_identical_notes:
        lines.append(f"Found `{len(non_identical_notes)}` non-identical `(system, seed)` pairs:")
        lines.extend(non_identical_notes)
    else:
        lines.append("All `(system, seed)` pairs are bit-identical (`best_bits` and `charged` match 12/12 on every cell).")
    lines.append("")
    return lines, len(fails) == 0, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part1", type=Path, required=True)
    ap.add_argument("--part2", type=Path, required=True)
    ap.add_argument("--readme", type=Path, required=True)
    args = ap.parse_args()

    with args.part1.open(newline="", encoding="utf-8") as f:
        p1_rows = list(csv.DictReader(f))
    with args.part2.open(newline="", encoding="utf-8") as f:
        p2_rows = list(csv.DictReader(f))

    p1_lines, p1_ok, p1_fails = summarize_part1(p1_rows)
    p2_lines, p2_ok, p2_fails = summarize_part2(p2_rows)
    overall_ok = p1_ok and p2_ok
    verdict = "PASS" if overall_ok else "FAIL"

    # Collect provenance without hostnames or VM paths
    machines = sorted(
        {r["machine_id_sha256"] for r in p1_rows} | {r["hashed_machine_id"] for r in p2_rows}
    )
    trees_p1 = sorted({f"{r['arm']}:{r['library_tree']}" for r in p1_rows})
    trees_p2 = sorted({f"{r['arm']}:{r['library_tree']}" for r in p2_rows})
    bins_p1 = sorted({f"{r['arm']}:{r['binary_sha256']}" for r in p1_rows})
    bins_p2 = sorted({f"{r['arm']}:{r['driver']}:{r['binary_sha256']}" for r in p2_rows})

    header = [
        "# nm3-scan-safe4 Gate Report",
        "",
        f"**Overall Verdict: {verdict}**",
        "",
        f"- Part 1 (Symmetry Detection Microbench): **{'PASS' if p1_ok else 'FAIL'}**",
        f"- Part 2 (Default-On Panel, `PAPER_BUDGETS`): **{'PASS' if p2_ok else 'FAIL'}**",
        "",
        "## Provenance",
        "",
        f"- Hashed machine id(s) (SHA-256): {', '.join(f'`{m}`' for m in machines)}",
        f"- Part 1 arm trees: {', '.join(f'`{t}`' for t in trees_p1)}",
        f"- Part 2 arm trees: {', '.join(f'`{t}`' for t in trees_p2)}",
        "- Part 1 microbench binary SHA-256s:",
        *[f"  - `{b}`" for b in bins_p1],
        "- Part 2 binary SHA-256s:",
        *[f"  - `{b}`" for b in bins_p2],
        "",
    ]

    if p1_fails or p2_fails:
        header.extend(["## Failing Cases", ""])
        for item in p1_fails + p2_fails:
            header.append(f"- {item}")
        header.append("")

    doc = "\n".join(header + p1_lines + p2_lines) + "\n"
    args.readme.write_text(doc, encoding="utf-8", newline="\n")
    print(f"verdict={verdict} p1_ok={p1_ok} p2_ok={p2_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
