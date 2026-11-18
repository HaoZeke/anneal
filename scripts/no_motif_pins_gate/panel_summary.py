#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Panel view of the rows that `run_gate.py` records.

`run_gate.py summarize` reads each system on its own. This script groups the
systems into the panels of the gate. For each panel it gives the hits of both
arms with the exact Fisher test stratified by system, the van Elteren test on
calls (a Wilcoxon rank-sum test stratified by system, with the tie correction
and no continuity correction), the summed wall, time inside the potential and
time outside it, and the number of (system, seed) pairs on which the two arms
returned the same hit, calls and energy.

A panel of one system is that system: its tests are the system tests of
`run_gate.py`.

B loses a panel when it has fewer hits than A and the stratified Fisher p is
below 0.05, when its calls rank higher than A's and the van Elteren p is below
0.05, or when any system of the panel is lost on the system tests. The gate
is FIX when any panel is lost and PASS otherwise.

    panel_summary.py --rows rows.csv --panels panels.csv --out panel_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate  # noqa: E402

PANEL_SYSTEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("BLJ", run_gate.BLJ_SYSTEMS),
    ("Morse M75 rho14", ("m75r14",)),
    ("Morse M80 rho14", ("m80r14",)),
    ("Morse M80 rho6", ("m80r6",)),
    ("Sutton-Chen", run_gate.SC_SYSTEMS),
    ("water/TIP4P rigid", tuple(f"tip4p{n}" for n in run_gate.TIP4P_SIZES)),
)

FIELDS = (
    "panel",
    "systems",
    "runs_a",
    "runs_b",
    "hits_a",
    "hits_b",
    "p_hits",
    "p_calls",
    "calls_direction",
    "wall_a",
    "wall_b",
    "wall_ratio",
    "inside_a",
    "inside_b",
    "inside_ratio",
    "outside_a",
    "outside_b",
    "outside_ratio",
    "identical_runs",
    "failed_a",
    "failed_b",
    "lost_systems",
    "loss_hits",
    "loss_calls",
    "verdict",
)

# Two energies the two arms report as equal. The drivers print ten decimals.
SAME_ENERGY = 1e-8


def stratified_fisher(tables: list[tuple[int, int, int, int]]) -> float:
    """Two-sided exact p for the sum of A hits over strata.

    Each table is (hits A, misses A, hits B, misses B). Given the hits of each
    stratum, the A hits of that stratum are hypergeometric, and the strata are
    independent. The p value is the mass of the sums no more likely than the
    observed one, as in `run_gate.fisher_two_sided`, to which a single stratum
    reduces.
    """
    mass = {0: 1}
    denominator = 1
    observed = 0
    for hits_a, miss_a, hits_b, miss_b in tables:
        n_a, n_b = hits_a + miss_a, hits_b + miss_b
        total = hits_a + hits_b
        observed += hits_a
        stratum = {
            x: math.comb(n_a, x) * math.comb(n_b, total - x)
            for x in range(max(0, total - n_b), min(n_a, total) + 1)
        }
        merged: dict[int, int] = {}
        for x, left in mass.items():
            for y, right in stratum.items():
                merged[x + y] = merged.get(x + y, 0) + left * right
        mass = merged
        denominator *= math.comb(n_a + n_b, total)
    reference = mass[observed]
    tail = sum(m for m in mass.values() if m * 10**9 <= reference * (10**9 + 1))
    return min(1.0, tail / denominator)


def van_elteren(strata: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    """Two-sided van Elteren p for A against B, with the statistic z.

    Each stratum is (calls of A, calls of B). The statistic adds the A rank
    sums, less their means, with the weight 1 / (stratum size + 1). The
    variance carries the tie correction of each stratum. A positive z means
    A ranks higher, so B needs fewer calls. All-tied strata add nothing.
    """
    centred = 0.0
    variance = 0.0
    for x, y in strata:
        n1, n2 = len(x), len(y)
        n = n1 + n2
        if n1 == 0 or n2 == 0 or n < 2:
            continue
        ordered = sorted(x + y)
        ranks: dict[float, float] = {}
        tie = 0.0
        for i, j in run_gate.tie_groups(ordered):
            ranks[ordered[i]] = (i + 1 + j + 1) / 2.0
            size = j - i + 1
            tie += size**3 - size
        rank_sum = sum(ranks[v] for v in x)
        sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
        weight = 1.0 / (n + 1)
        centred += weight * (rank_sum - n1 * (n + 1) / 2.0)
        variance += weight * weight * sigma2
    if variance <= 0.0:
        return 1.0, 0.0
    z = centred / math.sqrt(variance)
    return min(1.0, math.erfc(abs(z) / math.sqrt(2.0))), z


def identical_runs(rows: list[dict[str, str]], systems: tuple[str, ...]) -> int:
    """(system, seed) pairs on which both arms returned the same result."""
    by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    for r in rows:
        if r["system"] in systems and r["status"] == "ok":
            by_key[(r["system"], int(r["seed"]), r["arm"])] = r
    same = 0
    for system in systems:
        for seed in run_gate.SEEDS:
            a = by_key.get((system, seed, "A"))
            b = by_key.get((system, seed, "B"))
            if a is None or b is None:
                continue
            if (
                a["hit"] == b["hit"]
                and a["evals_to_hit"] == b["evals_to_hit"]
                and abs(float(a["best"]) - float(b["best"])) <= SAME_ENERGY
            ):
                same += 1
    return same


def total(rows: list[dict[str, str]], systems: tuple[str, ...], arm: str, key: str) -> float:
    return sum(
        float(r[key])
        for r in rows
        if r["system"] in systems and r["arm"] == arm and r["status"] == "ok"
    )


def ratio(b: float, a: float) -> str:
    return f"{b / a:.4f}" if a > 0 else ""


def panel_summary(
    label: str,
    systems: tuple[str, ...],
    rows: list[dict[str, str]],
    panels: dict[str, dict[str, str]],
) -> dict[str, str]:
    present = tuple(s for s in systems if any(r["system"] == s for r in rows))
    per_system = {s: run_gate.system_summary(s, rows, panels[s]) for s in present}
    hits = {arm: 0 for arm in ("A", "B")}
    runs = {arm: 0 for arm in ("A", "B")}
    failed = {arm: 0 for arm in ("A", "B")}
    for r in rows:
        if r["system"] in present:
            runs[r["arm"]] += 1
            hits[r["arm"]] += int(r["hit"])
            failed[r["arm"]] += r["status"] != "ok"

    kinds = (("hits", "loss_hits"), ("calls", "loss_calls"), ("failure", "loss_failure"))
    lost = []
    for s, row in per_system.items():
        flags = [name for name, key in kinds if row[key] == "1"]
        if flags:
            lost.append(f"{s}:{'+'.join(flags)}")

    if len(present) == 1:
        only = per_system[present[0]]
        p_hits = float(only["fisher_p"])
        p_calls = float(only["mw_p_calls"])
        rank_a, rank_b = float(only["calls_rank_mean_a"]), float(only["calls_rank_mean_b"])
        direction = rank_b - rank_a
    else:
        tables = []
        strata = []
        for s in present:
            mine = {
                arm: [r for r in rows if r["system"] == s and r["arm"] == arm]
                for arm in ("A", "B")
            }
            hits_a = sum(int(r["hit"]) for r in mine["A"])
            hits_b = sum(int(r["hit"]) for r in mine["B"])
            tables.append((hits_a, len(mine["A"]) - hits_a, hits_b, len(mine["B"]) - hits_b))
            strata.append(
                (
                    [float(r["evals_to_hit"]) for r in mine["A"]],
                    [float(r["evals_to_hit"]) for r in mine["B"]],
                )
            )
        p_hits = stratified_fisher(tables)
        p_calls, z = van_elteren(strata)
        direction = -z

    loss_hits = hits["B"] < hits["A"] and p_hits < run_gate.ALPHA
    loss_calls = direction > 0 and p_calls < run_gate.ALPHA
    complete = (
        len(present) == len(systems)
        and runs["A"] == runs["B"] == len(systems) * len(run_gate.SEEDS)
    )
    if loss_hits or loss_calls or lost:
        verdict = "FIX"
    else:
        verdict = "PASS" if complete else "INCOMPLETE"

    wall = {arm: total(rows, present, arm, "wall_s") for arm in ("A", "B")}
    inside = {arm: total(rows, present, arm, "pot_s") for arm in ("A", "B")}
    outside = {arm: total(rows, present, arm, "outside_s") for arm in ("A", "B")}
    return {
        "panel": label,
        "systems": str(len(present)),
        "runs_a": str(runs["A"]),
        "runs_b": str(runs["B"]),
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "p_hits": f"{p_hits:.6g}",
        "p_calls": f"{p_calls:.6g}",
        "calls_direction": (
            "B more calls" if direction > 0 else "B fewer calls" if direction < 0 else "equal"
        ),
        "wall_a": f"{wall['A']:.3f}",
        "wall_b": f"{wall['B']:.3f}",
        "wall_ratio": ratio(wall["B"], wall["A"]),
        "inside_a": f"{inside['A']:.3f}",
        "inside_b": f"{inside['B']:.3f}",
        "inside_ratio": ratio(inside["B"], inside["A"]),
        "outside_a": f"{outside['A']:.3f}",
        "outside_b": f"{outside['B']:.3f}",
        "outside_ratio": ratio(outside["B"], outside["A"]),
        "identical_runs": str(identical_runs(rows, present)),
        "failed_a": str(failed["A"]),
        "failed_b": str(failed["B"]),
        "lost_systems": ";".join(lost),
        "loss_hits": str(int(loss_hits)),
        "loss_calls": str(int(loss_calls)),
        "verdict": verdict,
    }


def overall(summary: list[dict[str, str]]) -> str:
    lost = [r["panel"] for r in summary if r["verdict"] == "FIX"]
    if lost:
        return "FIX: " + ", ".join(lost)
    if len(summary) != len(PANEL_SYSTEMS) or any(r["verdict"] != "PASS" for r in summary):
        return "INCOMPLETE"
    return "PASS"


def markdown(summary: list[dict[str, str]]) -> str:
    head = (
        "| panel | systems | hits A | hits B | stratified Fisher p | van Elteren p on calls "
        "| calls | wall A (s) | wall B (s) | inside A (s) | inside B (s) "
        "| outside A (s) | outside B (s) | same result | verdict |"
    )
    lines = [head, "|" + " --- |" * 15]
    for r in summary:
        lines.append(
            f"| {r['panel']} | {r['systems']} | {r['hits_a']}/{r['runs_a']} "
            f"| {r['hits_b']}/{r['runs_b']} | {r['p_hits']} | {r['p_calls']} "
            f"| {r['calls_direction']} | {r['wall_a']} | {r['wall_b']} | {r['inside_a']} "
            f"| {r['inside_b']} | {r['outside_a']} | {r['outside_b']} "
            f"| {r['identical_runs']}/{r['runs_a']} | {r['verdict']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True)
    parser.add_argument("--panels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = run_gate.read_rows(Path(args.rows))
    if not rows:
        print("no rows", file=sys.stderr)
        return 2
    panels = {r["system"]: r for r in run_gate.read_rows(Path(args.panels))}
    summary = [
        panel_summary(label, systems, rows, panels)
        for label, systems in PANEL_SYSTEMS
        if any(r["system"] in systems for r in rows)
    ]
    with Path(args.out).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(overall(summary))
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
