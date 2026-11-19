#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize the LJ98 n=48 paired rows and the pooled n=60 rows.

The cell is the lj98 cell of bench/no-size-constants-r3-gate/rows.csv.
Budget, tolerance and reference are read from those rows and must match
every n=48 row. The gate rows are not rewritten.

Pre-set rule, fixed before the n=48 rows existed: PASS if the two-sided
Mann-Whitney p is at least 0.05 and B has no fewer hits than A, or B is
better. B is better when it has strictly more hits, or the same number of
hits and a strictly lower median evals_to_hit. Otherwise FAIL. The outside
ratio is reported and is not part of the rule. A hit drop fails.

Medians are the type-7 sample quantile. The Fisher exact test is two-sided.
The Mann-Whitney test is two-sided with the tie and continuity corrections
used by the gate summarizer. U is the statistic for arm B against arm A.
outside_s/charged is the per-row ratio; the reported value is its median,
and the ratio is median(B) / median(A).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE_ROWS = HERE.parent / "no-size-constants-r3-gate" / "rows.csv"
N48_ROWS = HERE / "rows-lj98-n48.csv"

A_REV = "dd054749d4703923f4cf4c34f50589c574ee1276"
A_TREE = "f1705d6f02f7f257f50e831c16057740283efea5"
B_REV = "33c799031109a32ff9f3e61e9cdd94774adf54c6"
B_TREE = "3fc07c20d323a7ae5b34304720da7d7656309ba4"

GATE_SEEDS = tuple(range(2000, 2012))
N48_SEEDS = tuple(range(2012, 2060))

SUMMARY_FIELDS = [
    "cohort",
    "n",
    "budget",
    "reference",
    "tol",
    "hits_a",
    "hits_b",
    "fisher_p",
    "median_evals_a",
    "median_evals_b",
    "mannwhitney_u",
    "mannwhitney_p",
    "median_outside_per_charged_a",
    "median_outside_per_charged_b",
    "outside_per_charged_ratio",
    "hit_drop",
    "b_better",
    "outcome",
]

PER_SEED_FIELDS = [
    "cohort",
    "seed",
    "hit_a",
    "hit_b",
    "evals_to_hit_a",
    "evals_to_hit_b",
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


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, log-space form of the gate sum."""
    hits = hits_a + hits_b
    lo, hi = max(0, hits - n_b), min(n_a, hits)

    def log_p(k: int) -> float:
        return _log_comb(n_a, k) + _log_comb(n_b, hits - k) - _log_comb(n_a + n_b, hits)

    observed = log_p(hits_a)
    cutoff = observed + math.log1p(1e-7)
    terms = [log_p(k) for k in range(lo, hi + 1) if log_p(k) <= cutoff]
    if not terms:
        return 0.0
    peak = max(terms)
    total = math.exp(peak) * sum(math.exp(term - peak) for term in terms)
    return min(1.0, total)


def mann_whitney_two_sided(left: list[float], right: list[float]) -> tuple[float, float]:
    """Two-sided U of ``left`` vs ``right``, normal approximation with tie and continuity corrections."""
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


def rule_outcome(hits_a: int, hits_b: int, median_a: float, median_b: float, mw_p: float) -> tuple[bool, bool, str]:
    hit_drop = hits_b < hits_a
    b_better = (hits_b > hits_a) or (hits_b == hits_a and median_b < median_a)
    if (mw_p >= 0.05 and not hit_drop) or b_better:
        return hit_drop, b_better, "PASS"
    return hit_drop, b_better, "FAIL"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _unique(rows: list[dict[str, str]], key: str) -> str:
    found = {row[key] for row in rows}
    if len(found) != 1:
        raise SystemExit(f"{key} is not unique: {sorted(found)}")
    return next(iter(found))


def _arm_map(rows: list[dict[str, str]], expect: tuple[int, ...], cohort: str) -> dict[str, dict[int, dict[str, str]]]:
    arms: dict[str, dict[int, dict[str, str]]] = {"A": {}, "B": {}}
    for row in rows:
        if row["panel"] != "lj" or row["system"] != "lj98" or row["driver"] != "h2h_seed":
            raise SystemExit(f"{cohort}: unexpected cell {row['panel']},{row['system']},{row['driver']}")
        arm = row["arm"]
        if arm not in arms:
            raise SystemExit(f"{cohort}: unexpected arm {arm}")
        seed = int(row["seed"])
        if seed in arms[arm]:
            raise SystemExit(f"{cohort}: duplicate {arm} seed {seed}")
        arms[arm][seed] = row
    for arm in ("A", "B"):
        got = tuple(sorted(arms[arm]))
        if got != expect:
            raise SystemExit(f"{cohort} arm {arm} seeds {got[:3]}..{got[-3:]} ({len(got)}) != expected {len(expect)}")
    return arms


def _check_identity(row: dict[str, str]) -> None:
    arm = row["arm"]
    if arm == "A":
        if row["code_revision"] != A_REV or row["tree"] != A_TREE:
            raise SystemExit(f"arm A identity mismatch on seed {row['seed']}")
    elif arm == "B":
        if row["code_revision"] != B_REV or row["tree"] != B_TREE:
            raise SystemExit(f"arm B identity mismatch on seed {row['seed']}")
    if row["a_revision"] != A_REV or row["b_revision"] != B_REV:
        raise SystemExit(f"revision pair mismatch on seed {row['seed']} arm {arm}")
    if row["a_tree"] != A_TREE or row["b_tree"] != B_TREE:
        raise SystemExit(f"tree pair mismatch on seed {row['seed']} arm {arm}")
    host = row["host"]
    if len(host) != 64 or any(c not in "0123456789abcdef" for c in host):
        raise SystemExit(f"host is not a hashed id on seed {row['seed']} arm {arm}")
    seed = int(row["seed"])
    expect_order = "0" if ((seed % 2 == 0 and arm == "A") or (seed % 2 == 1 and arm == "B")) else "1"
    if row["order"] != expect_order:
        raise SystemExit(f"order {row['order']} != {expect_order} for seed {seed} arm {arm}")


def _check_cell(row: dict[str, str], budget: str, reference: str, tol: str) -> None:
    if row["budget"] != budget or row["reference"] != reference or row["tol"] != tol:
        raise SystemExit(
            f"cell mismatch seed {row['seed']} arm {row['arm']}: "
            f"{row['budget']} {row['reference']} {row['tol']}"
        )
    hit = row["hit"]
    if hit not in ("0", "1"):
        raise SystemExit(f"bad hit {hit}")
    budget_n = int(budget)
    evals = int(row["evals_to_hit"])
    charged = int(row["charged"])
    if not (0 <= evals <= budget_n) or charged < 1:
        raise SystemExit(f"bad counts seed {row['seed']} arm {row['arm']}")
    if float(row["outside_s"]) < 0.0 or float(row["pot_s"]) < 0.0:
        raise SystemExit(f"negative time seed {row['seed']} arm {row['arm']}")


def load_gate_lj98(path: Path) -> tuple[dict[str, dict[int, dict[str, str]]], str, str, str]:
    selected = [row for row in read_rows(path) if row["system"] == "lj98"]
    if len(selected) != 24:
        raise SystemExit(f"gate lj98 row count {len(selected)} != 24")
    arms = _arm_map(selected, GATE_SEEDS, "gate")
    flat = [arms[arm][seed] for arm in ("A", "B") for seed in GATE_SEEDS]
    budget = _unique(flat, "budget")
    reference = _unique(flat, "reference")
    tol = _unique(flat, "tol")
    for row in flat:
        _check_identity(row)
        _check_cell(row, budget, reference, tol)
    return arms, budget, reference, tol


def load_n48(path: Path, budget: str, reference: str, tol: str) -> dict[str, dict[int, dict[str, str]]]:
    rows = read_rows(path)
    if any(row["system"] != "lj98" for row in rows):
        raise SystemExit("n=48 file contains a system other than lj98")
    arms = _arm_map(rows, N48_SEEDS, "n48")
    overlap = set(GATE_SEEDS) & set(N48_SEEDS)
    if overlap:
        raise SystemExit(f"seed ranges overlap: {sorted(overlap)}")
    for arm in ("A", "B"):
        for seed in N48_SEEDS:
            row = arms[arm][seed]
            _check_identity(row)
            _check_cell(row, budget, reference, tol)
    return arms


def _stats(arms: dict[str, dict[int, dict[str, str]]], seeds: tuple[int, ...]) -> dict[str, object]:
    rows_a = [arms["A"][seed] for seed in seeds]
    rows_b = [arms["B"][seed] for seed in seeds]
    n = len(seeds)
    hits_a = sum(int(row["hit"]) for row in rows_a)
    hits_b = sum(int(row["hit"]) for row in rows_b)
    evals_a = [float(row["evals_to_hit"]) for row in rows_a]
    evals_b = [float(row["evals_to_hit"]) for row in rows_b]
    ratio_a = [float(row["outside_s"]) / int(row["charged"]) for row in rows_a]
    ratio_b = [float(row["outside_s"]) / int(row["charged"]) for row in rows_b]
    fisher_p = fisher_two_sided(hits_a, n, hits_b, n)
    u_stat, mw_p = mann_whitney_two_sided(evals_b, evals_a)
    median_a = type7(evals_a)
    median_b = type7(evals_b)
    med_out_a = type7(ratio_a)
    med_out_b = type7(ratio_b)
    if med_out_a > 0.0:
        out_ratio = med_out_b / med_out_a
    elif med_out_b == 0.0:
        out_ratio = 1.0
    else:
        out_ratio = float("inf")
    hit_drop, b_better, outcome = rule_outcome(hits_a, hits_b, median_a, median_b, mw_p)
    return {
        "n": n,
        "hits_a": hits_a,
        "hits_b": hits_b,
        "fisher_p": fisher_p,
        "median_a": median_a,
        "median_b": median_b,
        "u": u_stat,
        "mw_p": mw_p,
        "out_a": med_out_a,
        "out_b": med_out_b,
        "out_ratio": out_ratio,
        "hit_drop": hit_drop,
        "b_better": b_better,
        "outcome": outcome,
    }


def _fmt_p(value: float) -> str:
    return f"{value:.6g}"


def _fmt_ratio(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.6g}"
    return "inf"


def summary_row(cohort: str, budget: str, reference: str, tol: str, stats: dict[str, object]) -> dict[str, str]:
    n = int(stats["n"])
    return {
        "cohort": cohort,
        "n": str(n),
        "budget": budget,
        "reference": reference,
        "tol": tol,
        "hits_a": f"{stats['hits_a']}/{n}",
        "hits_b": f"{stats['hits_b']}/{n}",
        "fisher_p": _fmt_p(float(stats["fisher_p"])),
        "median_evals_a": f"{float(stats['median_a']):.1f}",
        "median_evals_b": f"{float(stats['median_b']):.1f}",
        "mannwhitney_u": f"{float(stats['u']):.1f}",
        "mannwhitney_p": _fmt_p(float(stats["mw_p"])),
        "median_outside_per_charged_a": _fmt_ratio(float(stats["out_a"])),
        "median_outside_per_charged_b": _fmt_ratio(float(stats["out_b"])),
        "outside_per_charged_ratio": _fmt_ratio(float(stats["out_ratio"])),
        "hit_drop": str(int(bool(stats["hit_drop"]))),
        "b_better": str(int(bool(stats["b_better"]))),
        "outcome": str(stats["outcome"]),
    }


def per_seed_rows(cohort: str, arms: dict[str, dict[int, dict[str, str]]], seeds: tuple[int, ...]) -> list[dict[str, str]]:
    out = []
    for seed in seeds:
        out.append(
            {
                "cohort": cohort,
                "seed": str(seed),
                "hit_a": arms["A"][seed]["hit"],
                "hit_b": arms["B"][seed]["hit"],
                "evals_to_hit_a": arms["A"][seed]["evals_to_hit"],
                "evals_to_hit_b": arms["B"][seed]["evals_to_hit"],
            }
        )
    return out


def _paragraph(row: dict[str, str]) -> str:
    return (
        f"Hits {row['hits_a']} (A) and {row['hits_b']} (B), Fisher exact p = {row['fisher_p']}. "
        f"Median evals_to_hit {row['median_evals_a']} (A) and {row['median_evals_b']} (B), "
        f"Mann-Whitney U (B vs A) = {row['mannwhitney_u']}, two-sided p = {row['mannwhitney_p']}. "
        f"Median outside_s/charged {row['median_outside_per_charged_a']} (A) and "
        f"{row['median_outside_per_charged_b']} (B), ratio B/A = {row['outside_per_charged_ratio']}. "
        f"Hit drop = {row['hit_drop']}. B better = {row['b_better']}. Outcome = {row['outcome']}."
    )


def write_readme(
    path: Path,
    budget: str,
    reference: str,
    tol: str,
    host: str,
    n48: dict[str, str],
    n60: dict[str, str],
    seeds_table: list[dict[str, str]],
) -> None:
    lines = [
        "# LJ98 n=48 paired comparison",
        "",
        f"n=48 outcome: `{n48['outcome']}`. n=60 outcome: `{n60['outcome']}`.",
        "",
        "## Pre-set rule",
        "",
        "PASS if the two-sided Mann-Whitney p is at least 0.05 and B has no fewer hits than A, or B is better. B is better when it has strictly more hits, or the same number of hits and a strictly lower median evals_to_hit. Otherwise FAIL. A hit drop is not treated as B being better. The outside_s/charged ratio is reported and is not part of the rule.",
        "",
        "## Cell",
        "",
        f"System `lj98`. Budget `{budget}`. Reference `{reference}`. Tolerance `{tol}`. These three values are the lj98 cell of the gate rows, and every n=48 row carries the same three.",
        "",
        f"Arm A revision `{A_REV}`, tree `{A_TREE}`. Arm B revision `{B_REV}`, tree `{B_TREE}`. Hashed host id `{host}`.",
        "",
        "## Protocol",
        "",
        "- One serial session. No warm-up run and no second pass over a finished seed.",
        "- Seeds 2012-2059 (48 values), disjoint from the gate seeds 2000-2011.",
        "- Even seeds run A then B. Odd seeds run B then A. From seed 2012 that sequence is A, B, B, A, and it repeats.",
        "- Each arm and seed is a fresh process. OpenMP, OpenBLAS, MKL, BLIS, VECLIB, NUMBA and Rayon threads are 1. The process is pinned to CPU core 0.",
        "- Both arms are release builds. The preset is `recommended`. The stop level is the reference plus the tolerance.",
        "- Median is the type-7 sample quantile. Fisher exact is two-sided. Mann-Whitney is two-sided with tie and continuity corrections; U is arm B against arm A.",
        "- outside_s/charged is computed per row. The arm value is the median of those ratios.",
        "",
        "## n=48",
        "",
        _paragraph(n48),
        "",
        "## Pooled n=60",
        "",
        "The 48 rows above plus the 12 lj98 rows of the gate rows file, seeds 2000-2011, unchanged.",
        "",
        _paragraph(n60),
        "",
        "## Per-seed evals_to_hit",
        "",
        "Columns are cohort, seed, evals_to_hit A, evals_to_hit B. The same table is `per-seed.csv`.",
        "",
        "| cohort | seed | evals A | evals B |",
        "| --- | --- | --- | --- |",
    ]
    for row in seeds_table:
        lines.append(
            f"| {row['cohort']} | {row['seed']} | {row['evals_to_hit_a']} | {row['evals_to_hit_b']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pooled_arms(
    gate: dict[str, dict[int, dict[str, str]]],
    n48: dict[str, dict[int, dict[str, str]]],
) -> dict[str, dict[int, dict[str, str]]]:
    pooled: dict[str, dict[int, dict[str, str]]] = {"A": {}, "B": {}}
    for arm in ("A", "B"):
        pooled[arm].update(gate[arm])
        pooled[arm].update(n48[arm])
    return pooled


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-rows", type=Path, default=GATE_ROWS)
    parser.add_argument("--n48-rows", type=Path, default=N48_ROWS)
    parser.add_argument("--summary", type=Path, default=HERE / "summary.csv")
    parser.add_argument("--per-seed", type=Path, default=HERE / "per-seed.csv")
    parser.add_argument("--report", type=Path, default=HERE / "README.md")
    args = parser.parse_args(argv[1:])

    gate, budget, reference, tol = load_gate_lj98(args.gate_rows)
    n48 = load_n48(args.n48_rows, budget, reference, tol)
    hosts = {
        row["host"]
        for arm in ("A", "B")
        for seed in N48_SEEDS
        for row in (n48[arm][seed],)
    }
    if len(hosts) != 1:
        raise SystemExit(f"n=48 host ids differ: {sorted(hosts)}")
    host = next(iter(hosts))

    stats_48 = _stats(n48, N48_SEEDS)
    stats_60 = _stats(pooled_arms(gate, n48), GATE_SEEDS + N48_SEEDS)
    row_48 = summary_row("n48", budget, reference, tol, stats_48)
    row_60 = summary_row("n60", budget, reference, tol, stats_60)
    table = per_seed_rows("gate", gate, GATE_SEEDS) + per_seed_rows("n48", n48, N48_SEEDS)

    write_csv(args.summary, SUMMARY_FIELDS, [row_48, row_60])
    write_csv(args.per_seed, PER_SEED_FIELDS, table)
    write_readme(args.report, budget, reference, tol, host, row_48, row_60, table)
    print(f"n48 {row_48['outcome']} fisher={row_48['fisher_p']} mw={row_48['mannwhitney_p']}")
    print(f"n60 {row_60['outcome']} fisher={row_60['fisher_p']} mw={row_60['mannwhitney_p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
