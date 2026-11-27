#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Analyze anneal #179 r8 panel rows and emit PASS / FIX / INCONCLUSIVE."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

EXPECTED_CELLS = (
    "lj38",
    "lj75",
    "lj98",
    "lj102",
    "lj104",
    "m80r14",
    "sc108-80",
    "blj130n38",
)
EXPECTED_N = 12
OUTSIDE_RATIO_BAR = 1.10
ALPHA = 0.05

SUMMARY_FIELDS = [
    "cell",
    "n_a",
    "n_b",
    "n_aa",
    "n_ap",
    "hits_a",
    "hits_b",
    "hits_aa",
    "hits_ap",
    "fisher_p_ba",
    "calls_med_a",
    "calls_med_b",
    "calls_med_aa",
    "charged_med_a",
    "charged_med_b",
    "charged_med_aa",
    "mwu_p_calls_ba",
    "outside_us_med_a",
    "outside_us_med_b",
    "outside_us_med_aa",
    "outside_us_med_ap",
    "outside_ratio_ba",
    "outside_ratio_aaa",
    "outside_ratio_apa",
    "mwu_p_outside_ba",
    "mwu_p_outside_aaa",
    "mwu_p_outside_apa",
    "outside_s_med_a",
    "outside_s_med_b",
    "outside_s_med_aa",
    "wall_s_med_a",
    "wall_s_med_b",
    "wall_s_med_aa",
    "rss_kb_med_a",
    "rss_kb_med_b",
    "rss_kb_med_aa",
    "rss_kb_med_ap",
    "rss_ratio_ba",
    "bit_id_aa",
    "bit_id_ab",
    "cell_status",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        return float("nan")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher_exact(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    if n_a <= 0 or n_b <= 0:
        return 1.0
    total_hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, total_hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, total_hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, total_hits - n_b), min(n_a, total_hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1.0 + 1e-12)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n1, n2 = len(left), len(right)
    if n1 < 1 or n2 < 1:
        return 0.0, 1.0
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

def bit_identical(r1: dict[str, str], r2: dict[str, str]) -> bool:
    return (
        r1["hit"] == r2["hit"]
        and r1["evals_to_hit"] == r2["evals_to_hit"]
        and r1["charged"] == r2["charged"]
        and r1["best"] == r2["best"]
    )


def fmt(val: float, digits: int = 4) -> str:
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_p(val: float) -> str:
    if math.isnan(val):
        return ""
    return f"{val:.6g}"


def analyze(rows_path: Path, summary_path: Path, json_path: Path) -> str:
    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_cell: dict[str, dict[str, dict[int, dict[str, str]]]] = {}
    for row in rows:
        cell = row["cell"]
        arm = row["arm"]
        seed = int(row["seed"])
        by_cell.setdefault(cell, {}).setdefault(arm, {})[seed] = row

    cells = list(EXPECTED_CELLS) + sorted(c for c in by_cell if c not in EXPECTED_CELLS)
    summary_rows: list[dict[str, str]] = []
    any_incomplete = False
    hit_failures: list[str] = []
    outside_failures: list[str] = []
    outside_inconclusive: list[str] = []

    for cell in cells:
        arms = by_cell.get(cell, {})
        a_map = arms.get("A", {})
        b_map = arms.get("B", {})
        aa_map = arms.get("AA", {})
        ap_map = arms.get("AP", {})

        a_rows = [a_map[s] for s in sorted(a_map)]
        b_rows = [b_map[s] for s in sorted(b_map)]
        aa_rows = [aa_map[s] for s in sorted(aa_map)]
        ap_rows = [ap_map[s] for s in sorted(ap_map)]

        n_a, n_b, n_aa, n_ap = len(a_rows), len(b_rows), len(aa_rows), len(ap_rows)
        complete = (
            n_a == EXPECTED_N
            and n_b == EXPECTED_N
            and n_aa == EXPECTED_N
            and sorted(a_map) == sorted(b_map) == sorted(aa_map)
        )
        if not complete:
            any_incomplete = True

        hits_a = sum(int(r["hit"]) for r in a_rows)
        hits_b = sum(int(r["hit"]) for r in b_rows)
        hits_aa = sum(int(r["hit"]) for r in aa_rows)
        hits_ap = sum(int(r["hit"]) for r in ap_rows)

        fisher_p_ba = fisher_exact(hits_a, n_a, hits_b, n_b) if (n_a and n_b) else 1.0

        calls_a = [float(r["evals_to_hit"]) for r in a_rows]
        calls_b = [float(r["evals_to_hit"]) for r in b_rows]
        calls_aa = [float(r["evals_to_hit"]) for r in aa_rows]
        calls_rank_a = [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in a_rows]
        calls_rank_b = [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in b_rows]
        _, mwu_p_calls_ba = mann_whitney(calls_rank_b, calls_rank_a) if (n_a and n_b) else (0.0, 1.0)

        charged_a = [float(r["charged"]) for r in a_rows]
        charged_b = [float(r["charged"]) for r in b_rows]
        charged_aa = [float(r["charged"]) for r in aa_rows]

        out_us_a = [float(r["outside_us"]) for r in a_rows]
        out_us_b = [float(r["outside_us"]) for r in b_rows]
        out_us_aa = [float(r["outside_us"]) for r in aa_rows]
        out_us_ap = [float(r["outside_us"]) for r in ap_rows]

        med_out_us_a = type7(out_us_a)
        med_out_us_b = type7(out_us_b)
        med_out_us_aa = type7(out_us_aa)
        med_out_us_ap = type7(out_us_ap)

        ratio_ba = med_out_us_b / med_out_us_a if med_out_us_a > 0 else float("nan")
        ratio_aaa = med_out_us_aa / med_out_us_a if (n_aa and med_out_us_a > 0) else float("nan")
        ratio_apa = med_out_us_ap / med_out_us_a if (n_ap and med_out_us_a > 0) else float("nan")

        _, mwu_p_out_ba = mann_whitney(out_us_b, out_us_a) if (n_a and n_b) else (0.0, 1.0)
        _, mwu_p_out_aaa = mann_whitney(out_us_aa, out_us_a) if (n_a and n_aa) else (0.0, 1.0)
        _, mwu_p_out_apa = mann_whitney(out_us_ap, out_us_a) if (n_a and n_ap) else (0.0, 1.0)

        out_s_a = [float(r["outside_s"]) for r in a_rows]
        out_s_b = [float(r["outside_s"]) for r in b_rows]
        out_s_aa = [float(r["outside_s"]) for r in aa_rows]

        wall_a = [float(r["wall_s"]) for r in a_rows]
        wall_b = [float(r["wall_s"]) for r in b_rows]
        wall_aa = [float(r["wall_s"]) for r in aa_rows]

        rss_a = [float(r["rss_anon_kb"]) for r in a_rows]
        rss_b = [float(r["rss_anon_kb"]) for r in b_rows]
        rss_aa = [float(r["rss_anon_kb"]) for r in aa_rows]
        rss_ap = [float(r["rss_anon_kb"]) for r in ap_rows]

        med_rss_a = type7(rss_a)
        med_rss_b = type7(rss_b)
        med_rss_aa = type7(rss_aa)
        med_rss_ap = type7(rss_ap)
        rss_ratio_ba = med_rss_b / med_rss_a if med_rss_a > 0 else float("nan")

        common_aa = sorted(set(a_map) & set(aa_map))
        common_ab = sorted(set(a_map) & set(b_map))
        bit_id_aa = sum(1 for s in common_aa if bit_identical(a_map[s], aa_map[s]))
        bit_id_ab = sum(1 for s in common_ab if bit_identical(a_map[s], b_map[s]))

        cell_status = "PASS"
        if not complete or math.isnan(ratio_ba):
            cell_status = "INCONCLUSIVE"
        elif hits_b < hits_a:
            cell_status = "FIX"
            hit_failures.append(f"{cell} (hits B={hits_b} < A={hits_a}, Fisher p={fisher_p_ba:.4g})")
        elif ratio_ba > OUTSIDE_RATIO_BAR:
            if (not math.isnan(ratio_aaa) and ratio_aaa > OUTSIDE_RATIO_BAR) and mwu_p_out_ba >= ALPHA:
                cell_status = "INCONCLUSIVE"
                outside_inconclusive.append(
                    f"{cell} (B/A={ratio_ba:.4f}, AA/A={ratio_aaa:.4f}, MWU p={mwu_p_out_ba:.4g})"
                )
            else:
                cell_status = "FIX"
                outside_failures.append(
                    f"{cell} (outside_us B/A={ratio_ba:.4f} > 1.10, AA/A={ratio_aaa:.4f}, MWU p={mwu_p_out_ba:.4g})"
                )

        summary_rows.append(
            {
                "cell": cell,
                "n_a": str(n_a),
                "n_b": str(n_b),
                "n_aa": str(n_aa),
                "n_ap": str(n_ap),
                "hits_a": str(hits_a),
                "hits_b": str(hits_b),
                "hits_aa": str(hits_aa),
                "hits_ap": str(hits_ap) if n_ap else "",
                "fisher_p_ba": fmt_p(fisher_p_ba),
                "calls_med_a": fmt(type7(calls_a), 1),
                "calls_med_b": fmt(type7(calls_b), 1),
                "calls_med_aa": fmt(type7(calls_aa), 1),
                "charged_med_a": fmt(type7(charged_a), 1),
                "charged_med_b": fmt(type7(charged_b), 1),
                "charged_med_aa": fmt(type7(charged_aa), 1),
                "mwu_p_calls_ba": fmt_p(mwu_p_calls_ba),
                "outside_us_med_a": fmt(med_out_us_a, 4),
                "outside_us_med_b": fmt(med_out_us_b, 4),
                "outside_us_med_aa": fmt(med_out_us_aa, 4),
                "outside_us_med_ap": fmt(med_out_us_ap, 4) if n_ap else "",
                "outside_ratio_ba": fmt(ratio_ba, 4),
                "outside_ratio_aaa": fmt(ratio_aaa, 4),
                "outside_ratio_apa": fmt(ratio_apa, 4) if n_ap else "",
                "mwu_p_outside_ba": fmt_p(mwu_p_out_ba),
                "mwu_p_outside_aaa": fmt_p(mwu_p_out_aaa),
                "mwu_p_outside_apa": fmt_p(mwu_p_out_apa) if n_ap else "",
                "outside_s_med_a": fmt(type7(out_s_a), 4),
                "outside_s_med_b": fmt(type7(out_s_b), 4),
                "outside_s_med_aa": fmt(type7(out_s_aa), 4),
                "wall_s_med_a": fmt(type7(wall_a), 4),
                "wall_s_med_b": fmt(type7(wall_b), 4),
                "wall_s_med_aa": fmt(type7(wall_aa), 4),
                "rss_kb_med_a": fmt(med_rss_a, 1),
                "rss_kb_med_b": fmt(med_rss_b, 1),
                "rss_kb_med_aa": fmt(med_rss_aa, 1),
                "rss_kb_med_ap": fmt(med_rss_ap, 1) if n_ap else "",
                "rss_ratio_ba": fmt(rss_ratio_ba, 4),
                "bit_id_aa": f"{bit_id_aa}/{len(common_aa)}",
                "bit_id_ab": f"{bit_id_ab}/{len(common_ab)}",
                "cell_status": cell_status,
            }
        )

    if hit_failures or outside_failures:
        verdict = "FIX"
    elif any_incomplete or outside_inconclusive:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "verdict": verdict,
        "total_rows": len(rows),
        "hit_failures": hit_failures,
        "outside_failures": outside_failures,
        "outside_inconclusive": outside_inconclusive,
        "cells": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return verdict


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument("--rows", type=Path, default=base / "rows.csv")
    parser.add_argument("--summary", type=Path, default=base / "summary.csv")
    parser.add_argument("--json", type=Path, default=base / "analysis.json")
    args = parser.parse_args(argv)
    verdict = analyze(args.rows, args.summary, args.json)
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
