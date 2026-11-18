#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the same-session baseline vs adaptive report from committed rows."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
SYSTEMS = (
    "m38r3",
    "m38r6",
    "m38r10",
    "m38r14",
    "m55r3",
    "m55r6",
    "m55r10",
    "m55r14",
    "m75r6",
    "m75r14",
    "m80r6",
    "m80r14",
    "lj38",
    "lj55",
    "lj75",
    "lj98",
    "lj104",
)
ARMS = ("recommended", "recommended_adaptive")
SPEC = {
    "m38r3": ("400000", -249.159174),
    "m38r6": ("400000", -157.477108),
    "m38r10": ("400000", -145.849817),
    "m38r14": ("400000", -144.321054),
    "m55r3": ("1000000", -417.918562),
    "m55r6": ("1000000", -250.286609),
    "m55r10": ("1000000", -225.814286),
    "m55r14": ("1000000", -220.646208),
    "m75r6": ("4000000", -351.472365),
    "m75r14": ("4000000", -318.407330),
    "m80r6": ("4000000", -378.333471),
    "m80r14": ("4000000", -340.811371),
    "lj38": ("400000", -173.928427),
    "lj55": ("1000000", -279.248470),
    "lj75": ("4000000", -397.492331),
    "lj98": ("4000000", -543.665361),
    "lj104": ("4000000", -582.086642),
}
ROW_FIELDS = (
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "wall_s",
    "inside_s",
    "outside_s",
    "code_revision",
    "host",
    "stop",
    "restarts",
    "orbits",
    "hops",
)
HIT_TOL = 0.001


def type7(ordered: list[float], probability: float) -> float:
    count = len(ordered)
    rank = 1.0 + (count - 1) * probability
    lower = math.floor(rank)
    weight = rank - lower
    left = ordered[int(lower) - 1]
    if weight == 0.0 or int(lower) >= count:
        return float(left)
    return float(left * (1.0 - weight) + ordered[int(lower)] * weight)


def median(values: list[float]) -> float:
    return type7(sorted(values), 0.5)


def calls(row: dict[str, str]) -> float:
    if row["evals_to_hit"]:
        return float(row["evals_to_hit"])
    return float(row["budget"])


def g6(value: float) -> str:
    return f"{value:.6g}"


def fisher(hits: int, base: int) -> float:
    row = 12
    col = hits + base
    total = 24

    def probability(value: int) -> float:
        return math.comb(row, value) * math.comb(row, col - value) / math.comb(total, col)

    observed = probability(hits)
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(max(0, col - row), min(row, col) + 1)
            if probability(value) <= observed * (1.0 + 1.0e-12)
        ),
    )


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    n_left, n_right = len(left), len(right)
    pooled = sorted([(value, 0) for value in left] + [(value, 1) for value in right])
    ranks = [0.0] * len(pooled)
    tie_term = 0
    index = 0
    while index < len(pooled):
        end = index + 1
        while end < len(pooled) and pooled[end][0] == pooled[index][0]:
            end += 1
        average = 0.5 * (index + 1 + end)
        ranks[index:end] = [average] * (end - index)
        width = end - index
        tie_term += width**3 - width
        index = end
    rank_left = sum(rank for rank, (_, arm) in zip(ranks, pooled) if arm == 0)
    statistic = rank_left - n_left * (n_left + 1) / 2.0
    total = n_left + n_right
    variance = n_left * n_right / 12.0 * (total + 1 - tie_term / (total * (total - 1)))
    if variance <= 0.0:
        return statistic, 1.0
    z = max(0.0, (abs(statistic - n_left * n_right / 2.0) - 0.5) / math.sqrt(variance))
    return statistic, min(1.0, math.erfc(z / math.sqrt(2.0)))


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    path = ROOT / "bench" / "morse-weak" / "rows.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise SystemExit(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    seen = set()
    for row in rows:
        system = row["system"]
        arm = row["arm"]
        if system not in SPEC or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        if row["budget"] != SPEC[system][0] or row["host"] != "bench":
            raise SystemExit(f"unexpected budget or host {row}")
        key = (system, arm, row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        budget = int(row["budget"])
        charged = int(row["charged"])
        if not 0 < charged <= budget:
            raise SystemExit(f"invalid charged count {row}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
        hit = row["hit"] == "1"
        if row["hit"] not in {"0", "1"} or hit != bool(row["evals_to_hit"]):
            raise SystemExit(f"hit/calls mismatch {row}")
        if hit:
            evals_to_hit = int(row["evals_to_hit"])
            if row["stop"] != "hit" or not 0 < evals_to_hit <= charged:
                raise SystemExit(f"invalid hit row {row}")
            if float(row["best"]) > SPEC[system][1] + HIT_TOL + 5.0e-9:
                raise SystemExit(f"hit above threshold {row}")
        elif row["stop"] == "budget":
            if charged != budget:
                raise SystemExit(f"invalid budget stop {row}")
        else:
            raise SystemExit(f"invalid stop {row}")
        grouped[system][arm].append(row)
    for system in SYSTEMS:
        for arm in ARMS:
            seeds = tuple(sorted(int(row["seed"]) for row in grouped[system][arm]))
            if seeds != SEEDS:
                raise SystemExit(f"{system} {arm} seeds {seeds}")
            grouped[system][arm].sort(key=lambda row: int(row["seed"]))
    return grouped


def main() -> None:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--check"):
        raise SystemExit("usage: summarize.py [--check]")
    grouped = load_rows()
    revisions = sorted(
        {row["code_revision"] for s in SYSTEMS for a in ARMS for row in grouped[s][a]}
    )
    summary_lines = [
        "system,base_hits,fix_hits,n,base_median_calls,fix_median_calls,"
        "base_median_wall_s,fix_median_wall_s,base_median_inside_s,base_median_outside_s,"
        "fix_median_inside_s,fix_median_outside_s,fisher_p,mann_whitney_U,mann_whitney_p,wall_ratio",
    ]
    readme = [
        "Copyright (c) anneal developers",
        "",
        "# Same-session baseline vs adaptive population (12 Morse + 5 LJ clusters)",
        "",
        "Baseline is `Config::recommended_with_scales` (`recommended`).",
        "Fix is `Config::recommended_adaptive_with_scales` (`recommended_adaptive`): "
        "a two-replica fingerprint-diverse non-reversible population (`ladder_top = 2.0`) with "
        "adaptive orbit-completion throttling, acceptance-coupled step/temperature scaling, and "
        "fingerprint-diverse plateau restarts.",
        "Seeds 2000-2011, host `bench`, serial one core (`taskset -c 0`), no wall cap.",
        f"code_revision {', '.join(revisions)}.",
        "Hits reach `reference + 0.001`. Misses are censored at the paper budget.",
        "",
        "| system | budget | base hits | fix hits | base med calls | fix med calls | "
        "base wall s | fix wall s | Fisher p | U | MW p | wall ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for system in SYSTEMS:
        budget, _ = SPEC[system]
        base = grouped[system]["recommended"]
        fix = grouped[system]["recommended_adaptive"]
        base_hits = sum(row["hit"] == "1" for row in base)
        fix_hits = sum(row["hit"] == "1" for row in fix)
        base_calls = median([calls(row) for row in base])
        fix_calls = median([calls(row) for row in fix])
        base_wall = median([float(row["wall_s"]) for row in base])
        fix_wall = median([float(row["wall_s"]) for row in fix])
        base_in = median([float(row["inside_s"]) for row in base])
        base_out = median([float(row["outside_s"]) for row in base])
        fix_in = median([float(row["inside_s"]) for row in fix])
        fix_out = median([float(row["outside_s"]) for row in fix])
        fp = fisher(fix_hits, base_hits)
        u_stat, mw_p = mann_whitney([calls(row) for row in fix], [calls(row) for row in base])
        ratio = fix_wall / base_wall
        summary_lines.append(
            f"{system},{base_hits},{fix_hits},12,{base_calls:.1f},{fix_calls:.1f},"
            f"{base_wall:.6f},{fix_wall:.6f},{base_in:.6f},{base_out:.6f},"
            f"{fix_in:.6f},{fix_out:.6f},{g6(fp)},{g6(u_stat)},{g6(mw_p)},{ratio:.6f}"
        )
        readme.append(
            f"| {system} | {int(budget):,} | {base_hits}/12 | {fix_hits}/12 | "
            f"{base_calls:.1f} | {fix_calls:.1f} | {base_wall:.3f} | {fix_wall:.3f} | "
            f"{g6(fp)} | {g6(u_stat)} | {g6(mw_p)} | {ratio:.3f} |"
        )

    readme.extend(
        [
            "",
            "Medians are Hyndman-Fan type 7. Wall ratio is fix median wall over base median wall.",
            "Fisher is two-sided exact on hits. Mann-Whitney is two-sided asymptotic on calls",
            "with tie and continuity corrections (fix against base; misses censored at budget).",
            "",
        ]
    )
    out = ROOT / "bench" / "morse-weak"
    outputs = {
        out / "summary.csv": "\n".join(summary_lines) + "\n",
        out / "README.md": "\n".join(readme),
    }
    if len(sys.argv) == 2:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if path.read_text() != text
        ]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.write_text(text)


if __name__ == "__main__":
    main()
