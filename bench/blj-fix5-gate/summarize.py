#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize rows.csv into summary.csv and README.md with Fisher and MW tests."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

SYSTEM_ORDER = (
    ("blj", "blj105n38", 400_000, -177.2090846748, 12),
    ("blj", "blj105n55", 1_000_000, -283.1545313810, 12),
    ("blj", "blj115n38", 400_000, -187.3676867453, 12),
    ("blj", "blj115n55", 1_000_000, -289.2931853058, 12),
    ("blj", "blj130n38", 400_000, -191.1840192602, 12),
    ("blj", "blj130n55", 1_000_000, -306.7337643721, 12),
    ("blj", "blj130n98", 4_000_000, -590.7874132760, 12),
    ("binary_water", "lj100-binary", 2_000_000, -604.7963065717, 12),
    ("water_gmin", "tip4p8", 2_000_000, -305.5183221676, 12),
    ("water_gmin", "tip4p12", 2_000_000, -492.9082357122, 12),
    ("water_gmin", "tip4p16", 2_000_000, -681.1925945898, 12),
    ("water_gmin", "tip4p20", 2_000_000, -872.9887544952, 12),
    ("water_gmin", "tip4p21", 2_000_000, -916.7074302940, 12),
)

SUMMARY_FIELDS = [
    "panel",
    "system",
    "n",
    "budget",
    "reference",
    "hits_a",
    "hits_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "mw_u_calls",
    "mw_p_calls",
    "pot_median_a_s",
    "pot_median_b_s",
    "outside_median_a_s",
    "outside_median_b_s",
    "wall_median_a_s",
    "wall_median_b_s",
    "best_min_a",
    "best_min_b",
    "hit_drop",
    "call_increase",
    "lose_hits",
    "lose_calls",
    "verdict",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    pos = 1.0 + (count - 1) * p
    lo = int(pos)
    frac = pos - lo
    if frac == 0.0:
        return ordered[lo - 1]
    return ordered[lo - 1] * (1.0 - frac) + ordered[lo] * frac


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    _, p = fisher_exact([[hits_b, n_b - hits_b], [hits_a, n_a - hits_a]], alternative="two-sided")
    return float(p)


def mw_two_sided(left_b: list[float], right_a: list[float]) -> tuple[float, float]:
    if len(set(left_b + right_a)) <= 1:
        return len(left_b) * len(right_a) / 2.0, 1.0
    res = mannwhitneyu(left_b, right_a, alternative="two-sided", method="asymptotic", use_continuity=True)
    return float(res.statistic), float(res.pvalue)


def fmt_p(p: float) -> str:
    if p >= 0.99995:
        return "1"
    if p >= 0.01:
        return f"{p:.4f}"
    return f"{p:.2e}"


def fmt_num(v: float) -> str:
    return f"{v:.1f}" if v != int(v) else f"{int(v)}"


def fmt_s(v: float) -> str:
    return f"{v:.3f}" if v < 100.0 else f"{v:.1f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["panel"], row["system"], row["arm"]), {})[int(row["seed"])] = row

    out = []
    for panel, system, budget, ref, expected_n in SYSTEM_ORDER:
        ga_map = grouped.get((panel, system, "A"), {})
        gb_map = grouped.get((panel, system, "B"), {})
        if not ga_map or not gb_map:
            continue
        seeds = sorted(set(ga_map) & set(gb_map))
        if len(seeds) != expected_n or len(ga_map) != expected_n or len(gb_map) != expected_n:
            raise SystemExit(
                f"{panel}/{system}: expected {expected_n} paired seeds, got A={len(ga_map)} B={len(gb_map)}"
            )
        ga = [ga_map[s] for s in seeds]
        gb = [gb_map[s] for s in seeds]

        def col(group: list[dict[str, str]], key: str) -> list[float]:
            return [float(r[key]) for r in group]

        ha = sum(int(r["hit"]) for r in ga)
        hb = sum(int(r["hit"]) for r in gb)
        fp = fisher_two_sided(ha, expected_n, hb, expected_n)

        ca = col(ga, "evals_to_hit")
        cb = col(gb, "evals_to_hit")
        med_ca = type7(ca, 0.5)
        med_cb = type7(cb, 0.5)
        u_calls, p_calls = mw_two_sided(cb, ca)

        pot_a = type7(col(ga, "pot_s"), 0.5)
        pot_b = type7(col(gb, "pot_s"), 0.5)
        out_a = type7(col(ga, "outside_s"), 0.5)
        out_b = type7(col(gb, "outside_s"), 0.5)
        wall_a = type7(col(ga, "wall_s"), 0.5)
        wall_b = type7(col(gb, "wall_s"), 0.5)
        best_a = min(col(ga, "best"))
        best_b = min(col(gb, "best"))

        hit_drop = hb < ha
        call_inc = (med_cb > med_ca) or (med_cb == med_ca and u_calls > (expected_n * expected_n / 2.0))
        lose_hits = hit_drop and fp < 0.05
        lose_calls = call_inc and p_calls < 0.05
        verdict = "LOSE" if (lose_hits or lose_calls) else "PASS"

        out.append({
            "panel": panel,
            "system": system,
            "n": str(expected_n),
            "budget": str(budget),
            "reference": f"{ref:.10f}",
            "hits_a": f"{ha}/{expected_n}",
            "hits_b": f"{hb}/{expected_n}",
            "fisher_p": f"{fp:.6g}",
            "calls_median_a": f"{med_ca:.1f}",
            "calls_median_b": f"{med_cb:.1f}",
            "mw_u_calls": f"{u_calls:.1f}",
            "mw_p_calls": f"{p_calls:.6g}",
            "pot_median_a_s": f"{pot_a:.6f}",
            "pot_median_b_s": f"{pot_b:.6f}",
            "outside_median_a_s": f"{out_a:.6f}",
            "outside_median_b_s": f"{out_b:.6f}",
            "wall_median_a_s": f"{wall_a:.6f}",
            "wall_median_b_s": f"{wall_b:.6f}",
            "best_min_a": f"{best_a:.10f}",
            "best_min_b": f"{best_b:.10f}",
            "hit_drop": "yes" if hit_drop else "no",
            "call_increase": "yes" if call_inc else "no",
            "lose_hits": "yes" if lose_hits else "no",
            "lose_calls": "yes" if lose_calls else "no",
            "verdict": verdict,
        })
    return out


def splice(path: Path, start: str, end: str, block: str) -> None:
    text = path.read_text()
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path} missing markers {start!r} / {end!r}")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}")


def render_readme_table(summary: list[dict[str, str]]) -> str:
    header = [
        "Panel",
        "System",
        "n",
        "Budget",
        "Hits A",
        "Hits B",
        "Fisher p",
        "Calls A",
        "Calls B",
        "MW p",
        "Inside A/B (s)",
        "Outside A/B (s)",
        "Verdict",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in summary:
        lines.append("| " + " | ".join([
            r["panel"],
            r["system"],
            r["n"],
            r["budget"],
            r["hits_a"],
            r["hits_b"],
            fmt_p(float(r["fisher_p"])),
            fmt_num(float(r["calls_median_a"])),
            fmt_num(float(r["calls_median_b"])),
            fmt_p(float(r["mw_p_calls"])),
            f"{fmt_s(float(r['pot_median_a_s']))} / {fmt_s(float(r['pot_median_b_s']))}",
            f"{fmt_s(float(r['outside_median_a_s']))} / {fmt_s(float(r['outside_median_b_s']))}",
            r["verdict"],
        ]) + " |")
    return "\n".join(lines)


def render_verdict_block(summary: list[dict[str, str]]) -> str:
    loses = [r for r in summary if r["verdict"] == "LOSE"]
    overall = "B LOSES (opt-in only)" if loses else "PASS"
    hit_drops = [r for r in summary if r["hit_drop"] == "yes"]
    call_incs = [r for r in summary if r["call_increase"] == "yes"]

    lines = [
        f"**Verdict: {overall}** (Arm B: `cb8ee65b`, tree `208b1b1d`)",
    ]
    if loses:
        reasons = []
        for r in loses:
            flags = []
            if r["lose_hits"] == "yes":
                flags.append(f"fewer hits ({r['hits_b']} vs {r['hits_a']}, Fisher p={fmt_p(float(r['fisher_p']))})")
            if r["lose_calls"] == "yes":
                flags.append(f"more calls ({fmt_num(float(r['calls_median_b']))} vs {fmt_num(float(r['calls_median_a']))}, MW p={fmt_p(float(r['mw_p_calls']))})")
            reasons.append(f"- `{r['panel']}/{r['system']}`: " + "; ".join(flags))
        lines.extend(["", "Significant losses:", *reasons])
    lines.append("")
    if hit_drops:
        lines.append("Hit drops (all, including non-significant):")
        for r in hit_drops:
            lines.append(f"- `{r['panel']}/{r['system']}`: {r['hits_a']} -> {r['hits_b']} (Fisher p={fmt_p(float(r['fisher_p']))})")
    else:
        lines.append("Hit drops: none.")
    lines.append("")
    if call_incs:
        lines.append("Call increases (median or stochastic rank increase, all including non-significant):")
        for r in call_incs:
            lines.append(
                f"- `{r['panel']}/{r['system']}`: {fmt_num(float(r['calls_median_a']))} -> "
                f"{fmt_num(float(r['calls_median_b']))} (U={r['mw_u_calls']}, MW p={fmt_p(float(r['mw_p_calls']))})"
            )
    else:
        lines.append("Call increases: none.")
    return "\n".join(lines)


def self_test() -> None:
    assert abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) < 1e-12
    assert fisher_two_sided(6, 12, 6, 12) == 1.0
    assert fisher_two_sided(0, 12, 12, 12) < 1e-6
    u, p = mw_two_sided([10.0] * 12, [10.0] * 12)
    assert u == 72.0 and p == 1.0
    # Cross-check exact combinatoric Fisher vs scipy.
    a, b, n = 9, 2, 12
    ways = math.comb(2 * n, a + b)
    p_obs = math.comb(n, a) * math.comb(n, b) / ways
    exact = sum(
        math.comb(n, k) * math.comb(n, a + b - k) / ways
        for k in range(max(0, a + b - n), min(n, a + b) + 1)
        if math.comb(n, k) * math.comb(n, a + b - k) / ways <= p_obs * (1 + 1e-9)
    )
    assert abs(fisher_two_sided(a, n, b, n) - min(1.0, exact)) < 1e-12
    print("ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.rows is None or args.summary is None:
        parser.error("--rows and --summary are required")
    rows = read_rows(args.rows)
    summary = summarise(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    if args.readme is not None and summary:
        splice(args.readme, "<!-- gate table: start -->", "<!-- gate table: end -->", render_readme_table(summary))
        splice(args.readme, "<!-- gate verdict: start -->", "<!-- gate verdict: end -->", render_verdict_block(summary))
    print(render_verdict_block(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
