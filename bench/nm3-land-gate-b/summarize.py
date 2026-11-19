#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize part B gate rows, verify ABBA balance and provenance, and generate README.md.

Computes per-system and per-panel statistics:
- Hits A vs B: two-sided exact Fisher test per system, stratified exact Fisher
  test across strata for multi-system panels.
- Calls to hit: Hyndman-Fan type 7 medians (misses entered at budget),
  two-sided tie- and continuity-corrected Mann-Whitney U test per system, and
  tie-corrected van Elteren test for multi-system panels.
- Timing: wall, time inside vs outside the potential, and outside-potential
  time per call (seconds and microseconds).
- TIP4P bit-identity: per-system and per-seed verification of byte-identical
  trajectories (.trace.csv, .xyz, .gmin), best energies, and call counts.
- Verdict: FIX naming any losing cells/panels if any loss at p < 0.05 on hits
  or calls (or B-only failures); otherwise PASS.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_gate  # noqa: E402

ALPHA = 0.05

PANEL_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Sutton-Chen", run_gate.SC_SYSTEMS),
    ("Morse M75 rho14", run_gate.MORSE_SYSTEMS),
    ("BLJ", run_gate.BLJ_SYSTEMS),
    (
        "water/TIP4P rigid",
        tuple(f"tip4p{n}" for n in range(run_gate.TIP4P_MIN_N, run_gate.TIP4P_MAX_N + 1)),
    ),
)

ALL_SYSTEMS = tuple(s for _label, systems in PANEL_ORDER for s in systems)

SUMMARY_FIELDS = [
    "panel",
    "system",
    "driver",
    "budget",
    "reference",
    "n_a",
    "n_b",
    "hits_a",
    "hits_b",
    "fisher_p",
    "calls_median_a",
    "calls_median_b",
    "calls_iqr_a",
    "calls_iqr_b",
    "mw_u_calls",
    "mw_p_calls",
    "calls_rank_mean_a",
    "calls_rank_mean_b",
    "calls_direction",
    "wall_median_a",
    "wall_median_b",
    "wall_ratio",
    "pot_median_a",
    "pot_median_b",
    "pot_ratio",
    "outside_median_a",
    "outside_median_b",
    "outside_ratio",
    "outside_per_call_median_a",
    "outside_per_call_median_b",
    "outside_per_call_us_a",
    "outside_per_call_us_b",
    "outside_per_call_ratio",
    "mw_p_outside_per_call",
    "identical_pairs",
    "failed_a",
    "failed_b",
    "loss_hits",
    "loss_calls",
    "loss_failure",
    "gain_hits",
    "gain_calls",
    "verdict",
]

PANEL_SUMMARY_FIELDS = [
    "panel",
    "systems",
    "runs_a",
    "runs_b",
    "hits_a",
    "hits_b",
    "p_hits",
    "p_calls",
    "van_elteren_z",
    "calls_direction",
    "wall_sum_a",
    "wall_sum_b",
    "wall_ratio",
    "inside_sum_a",
    "inside_sum_b",
    "inside_ratio",
    "outside_sum_a",
    "outside_sum_b",
    "outside_ratio",
    "outside_per_call_us_a",
    "outside_per_call_us_b",
    "outside_per_call_ratio",
    "identical_runs",
    "failed_a",
    "failed_b",
    "lost_systems",
    "loss_hits",
    "loss_calls",
    "verdict",
]

TIP4P_IDENTITY_FIELDS = [
    "system",
    "seed",
    "start_sha256_match",
    "trace_byte_identical",
    "geometry_byte_identical",
    "trajectory_byte_identical",
    "best_energy_identical",
    "calls_identical",
    "all_bit_identical",
    "hit_a",
    "hit_b",
    "evals_a",
    "evals_b",
    "charged_a",
    "charged_b",
    "best_a",
    "best_b",
    "best_bits_a",
    "best_bits_b",
    "trace_sha256_a",
    "trace_sha256_b",
    "xyz_sha256_a",
    "xyz_sha256_b",
    "gmin_sha256_a",
    "gmin_sha256_b",
]


def type7(values: list[float], p: float = 0.5) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float("nan")
    if n == 1:
        return ordered[0]
    h = 1.0 + (n - 1) * p
    lo = int(h)
    frac = h - lo
    idx = lo - 1
    if frac == 0.0:
        return ordered[idx]
    return ordered[idx] * (1.0 - frac) + ordered[idx + 1] * frac


def tie_groups(ordered: list[float]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        groups.append((i, j))
        i = j + 1
    return groups


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided exact Fisher test for [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1, row2 = a + b, a + c, c + d

    def mass(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = mass(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = mass(x)
        if px <= observed * (1.0 + 1e-9):
            total += px
    return min(1.0, total)


def mann_whitney(x: list[float], y: list[float]) -> tuple[float, float, float, float]:
    """Two-sided normal-approximation Mann-Whitney U with tie and continuity corrections.

    Returns (u_x, p_value, mean_rank_x, mean_rank_y).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ordered = sorted(x + y)
    n = len(ordered)
    ranks: dict[float, float] = {}
    tie = 0.0
    for i, j in tie_groups(ordered):
        ranks[ordered[i]] = (i + 1 + j + 1) / 2.0
        t = j - i + 1
        tie += t**3 - t
    r1 = sum(ranks[v] for v in x)
    r2 = sum(ranks[v] for v in y)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    if n < 2:
        return u1, 1.0, r1 / n1, r2 / n2
    sigma2 = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0.0:
        return u1, 1.0, r1 / n1, r2 / n2
    z = max(0.0, abs(u1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(sigma2)
    return u1, min(1.0, math.erfc(z / math.sqrt(2.0))), r1 / n1, r2 / n2


def stratified_fisher(tables: list[tuple[int, int, int, int]]) -> float:
    """Two-sided exact Fisher test stratified across independent 2x2 tables."""
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
    """Two-sided tie-corrected van Elteren test for A against B.

    Returns (p_value, z). Positive z means A ranks higher than B (B has fewer calls);
    negative z means A ranks lower than B (B has more calls).
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
        for i, j in tie_groups(ordered):
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


def verify_abba_balance(rows: list[dict[str, str]]) -> dict[tuple[int, str], int]:
    """Verify balanced (slot, arm) counts per system and overall."""
    overall: Counter[tuple[int, str]] = Counter()
    by_system: dict[str, Counter[tuple[int, str]]] = {}
    for r in rows:
        key = (int(r["slot"]), r["arm"])
        overall[key] += 1
        by_system.setdefault(r["system"], Counter())[key] += 1
    expected_keys = {(1, "A"), (1, "B"), (2, "A"), (2, "B")}
    for system, counts in sorted(by_system.items()):
        if set(counts) != expected_keys or len(set(counts.values())) != 1:
            raise SystemExit(f"unbalanced (slot, arm) counts for {system}: {dict(counts)}")
    if set(overall) != expected_keys or len(set(overall.values())) != 1:
        raise SystemExit(f"unbalanced overall (slot, arm) counts: {dict(overall)}")
    return dict(overall)


def verify_provenance_git(repo: Path, rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    """Verify recorded library commit/tree and harness blobs resolve with git cat-file -e."""
    objects: set[str] = set()
    for r in rows:
        for field in ("library", "library_tree"):
            val = r.get(field, "").strip()
            if val:
                sha = val.split(":", 1)[1] if ":" in val else val
                objects.add(sha)
        blobs = r.get("harness_blobs", "").strip()
        for item in blobs.split(";"):
            if ":" in item:
                _, sha = item.split(":", 1)
                objects.add(sha.strip())
    missing: list[str] = []
    for sha in sorted(objects):
        proc = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", sha],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            missing.append(sha)
    return len(objects), missing


def ratio(b: float, a: float) -> str:
    return f"{b / a:.4f}" if a > 0 else ""


def fmt(value: float, spec: str = ".6g") -> str:
    if value != value:
        return "nan"
    return format(value, spec)


def out_per_call(row: dict[str, str]) -> float:
    if row.get("outside_per_call_s"):
        return float(row["outside_per_call_s"])
    return float(row["outside_s"]) / max(int(float(row["charged"])), 1)


def is_same_result(a: dict[str, str], b: dict[str, str]) -> bool:
    if a["status"] != "ok" or b["status"] != "ok":
        return False
    if a.get("trace_sha256") and b.get("trace_sha256"):
        return (
            a["hit"] == b["hit"]
            and a["evals_to_hit"] == b["evals_to_hit"]
            and a["charged"] == b["charged"]
            and a["best"] == b["best"]
            and a["best_bits_hex"] == b["best_bits_hex"]
            and a["trace_sha256"] == b["trace_sha256"]
            and a["xyz_sha256"] == b["xyz_sha256"]
            and a["gmin_sha256"] == b["gmin_sha256"]
        )
    return (
        a["hit"] == b["hit"]
        and a["evals_to_hit"] == b["evals_to_hit"]
        and a["charged"] == b["charged"]
        and abs(float(a["best"]) - float(b["best"])) <= 1e-8
    )


def build_tip4p_identity(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tip4p_systems = PANEL_ORDER[3][1]
    by_key = {(r["system"], r["arm"], int(r["seed"])): r for r in rows if r["system"] in tip4p_systems}
    out: list[dict[str, str]] = []
    for system in tip4p_systems:
        for seed in run_gate.SEEDS:
            a = by_key.get((system, "A", seed))
            b = by_key.get((system, "B", seed))
            if a is None or b is None:
                continue
            start_match = bool(a["start_sha256"]) and a["start_sha256"] == b["start_sha256"]
            trace_same = bool(a["trace_sha256"]) and a["trace_sha256"] == b["trace_sha256"]
            geom_same = (
                bool(a["xyz_sha256"])
                and a["xyz_sha256"] == b["xyz_sha256"]
                and bool(a["gmin_sha256"])
                and a["gmin_sha256"] == b["gmin_sha256"]
            )
            traj_same = trace_same and geom_same
            best_same = (
                a["best"] == b["best"]
                and a["best_bits_hex"] == b["best_bits_hex"]
                and a["hit"] == b["hit"]
            )
            calls_same = a["evals_to_hit"] == b["evals_to_hit"] and a["charged"] == b["charged"]
            all_same = start_match and traj_same and best_same and calls_same
            out.append(
                {
                    "system": system,
                    "seed": str(seed),
                    "start_sha256_match": str(int(start_match)),
                    "trace_byte_identical": str(int(trace_same)),
                    "geometry_byte_identical": str(int(geom_same)),
                    "trajectory_byte_identical": str(int(traj_same)),
                    "best_energy_identical": str(int(best_same)),
                    "calls_identical": str(int(calls_same)),
                    "all_bit_identical": str(int(all_same)),
                    "hit_a": a["hit"],
                    "hit_b": b["hit"],
                    "evals_a": a["evals_to_hit"],
                    "evals_b": b["evals_to_hit"],
                    "charged_a": a["charged"],
                    "charged_b": b["charged"],
                    "best_a": a["best"],
                    "best_b": b["best"],
                    "best_bits_a": a["best_bits_hex"],
                    "best_bits_b": b["best_bits_hex"],
                    "trace_sha256_a": a["trace_sha256"],
                    "trace_sha256_b": b["trace_sha256"],
                    "xyz_sha256_a": a["xyz_sha256"],
                    "xyz_sha256_b": b["xyz_sha256"],
                    "gmin_sha256_a": a["gmin_sha256"],
                    "gmin_sha256_b": b["gmin_sha256"],
                }
            )
    return out


def system_summary(
    panel: str,
    system: str,
    rows: list[dict[str, str]],
    panel_meta: dict[str, str],
) -> dict[str, str]:
    by_arm = {
        arm: sorted(
            (r for r in rows if r["system"] == system and r["arm"] == arm),
            key=lambda r: int(r["seed"]),
        )
        for arm in ("A", "B")
    }
    good = {arm: [r for r in rs if r["status"] == "ok"] for arm, rs in by_arm.items()}
    failed = {arm: {r["seed"] for r in rs if r["status"] != "ok"} for arm, rs in by_arm.items()}
    hits = {arm: sum(int(r["hit"]) for r in rs) for arm, rs in by_arm.items()}
    runs = {arm: len(rs) for arm, rs in by_arm.items()}
    calls = {arm: [float(r["evals_to_hit"]) for r in rs] for arm, rs in by_arm.items()}

    p_hits = fisher_two_sided(hits["A"], runs["A"] - hits["A"], hits["B"], runs["B"] - hits["B"])
    u_calls, p_calls, rank_a, rank_b = mann_whitney(calls["A"], calls["B"])

    def col(arm: str, key: str) -> list[float]:
        return [float(r[key]) for r in good[arm]]

    wall_a = type7(col("A", "wall_s"))
    wall_b = type7(col("B", "wall_s"))
    pot_a = type7(col("A", "pot_s"))
    pot_b = type7(col("B", "pot_s"))
    out_a = type7(col("A", "outside_s"))
    out_b = type7(col("B", "outside_s"))
    opc_list_a = [out_per_call(r) for r in good["A"]]
    opc_list_b = [out_per_call(r) for r in good["B"]]
    opc_a = type7(opc_list_a)
    opc_b = type7(opc_list_b)
    p_opc = mann_whitney(opc_list_a, opc_list_b)[1]

    by_seed = {
        arm: {int(r["seed"]): r for r in rs}
        for arm, rs in by_arm.items()
    }
    common_seeds = sorted(set(by_seed["A"]) & set(by_seed["B"]))
    same_count = sum(is_same_result(by_seed["A"][s], by_seed["B"][s]) for s in common_seeds)

    loss_hits = hits["B"] < hits["A"] and p_hits < ALPHA
    loss_calls = rank_b > rank_a and p_calls < ALPHA
    loss_failure = bool(failed["B"] - failed["A"])
    gain_hits = hits["B"] > hits["A"] and p_hits < ALPHA
    gain_calls = rank_b < rank_a and p_calls < ALPHA
    complete = (
        set(common_seeds) == set(run_gate.SEEDS)
        and runs["A"] == len(run_gate.SEEDS)
        and runs["B"] == len(run_gate.SEEDS)
    )
    if loss_hits or loss_calls or loss_failure:
        verdict = "FIX"
    elif complete:
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"

    direction = (
        "B more calls" if rank_b > rank_a else "B fewer calls" if rank_b < rank_a else "equal"
    )
    return {
        "panel": panel,
        "system": system,
        "driver": panel_meta["driver"],
        "budget": panel_meta["budget"],
        "reference": panel_meta["reference"],
        "n_a": str(runs["A"]),
        "n_b": str(runs["B"]),
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "fisher_p": fmt(p_hits),
        "calls_median_a": fmt(type7(calls["A"])),
        "calls_median_b": fmt(type7(calls["B"])),
        "calls_iqr_a": fmt(type7(calls["A"], 0.75) - type7(calls["A"], 0.25)),
        "calls_iqr_b": fmt(type7(calls["B"], 0.75) - type7(calls["B"], 0.25)),
        "mw_u_calls": fmt(u_calls),
        "mw_p_calls": fmt(p_calls),
        "calls_rank_mean_a": fmt(rank_a),
        "calls_rank_mean_b": fmt(rank_b),
        "calls_direction": direction,
        "wall_median_a": f"{wall_a:.6f}" if good["A"] else "",
        "wall_median_b": f"{wall_b:.6f}" if good["B"] else "",
        "wall_ratio": ratio(wall_b, wall_a),
        "pot_median_a": f"{pot_a:.6f}" if good["A"] else "",
        "pot_median_b": f"{pot_b:.6f}" if good["B"] else "",
        "pot_ratio": ratio(pot_b, pot_a),
        "outside_median_a": f"{out_a:.6f}" if good["A"] else "",
        "outside_median_b": f"{out_b:.6f}" if good["B"] else "",
        "outside_ratio": ratio(out_b, out_a),
        "outside_per_call_median_a": f"{opc_a:.12f}" if opc_list_a else "",
        "outside_per_call_median_b": f"{opc_b:.12f}" if opc_list_b else "",
        "outside_per_call_us_a": f"{opc_a * 1e6:.4f}" if opc_list_a else "",
        "outside_per_call_us_b": f"{opc_b * 1e6:.4f}" if opc_list_b else "",
        "outside_per_call_ratio": ratio(opc_b, opc_a),
        "mw_p_outside_per_call": fmt(p_opc),
        "identical_pairs": str(same_count),
        "failed_a": str(len(failed["A"])),
        "failed_b": str(len(failed["B"])),
        "loss_hits": str(int(loss_hits)),
        "loss_calls": str(int(loss_calls)),
        "loss_failure": str(int(loss_failure)),
        "gain_hits": str(int(gain_hits)),
        "gain_calls": str(int(gain_calls)),
        "verdict": verdict,
    }


def panel_summary(
    label: str,
    systems: tuple[str, ...],
    rows: list[dict[str, str]],
    sys_summaries: dict[str, dict[str, str]],
) -> dict[str, str]:
    present = tuple(s for s in systems if s in sys_summaries)
    hits = {arm: 0 for arm in ("A", "B")}
    runs = {arm: 0 for arm in ("A", "B")}
    failed = {arm: 0 for arm in ("A", "B")}
    wall_sum = {arm: 0.0 for arm in ("A", "B")}
    pot_sum = {arm: 0.0 for arm in ("A", "B")}
    out_sum = {arm: 0.0 for arm in ("A", "B")}
    charged_sum = {arm: 0 for arm in ("A", "B")}
    for r in rows:
        if r["system"] not in present:
            continue
        arm = r["arm"]
        runs[arm] += 1
        hits[arm] += int(r["hit"])
        if r["status"] != "ok":
            failed[arm] += 1
        else:
            wall_sum[arm] += float(r["wall_s"])
            pot_sum[arm] += float(r["pot_s"])
            out_sum[arm] += float(r["outside_s"])
            charged_sum[arm] += int(float(r["charged"]))

    lost: list[str] = []
    for s in present:
        sr = sys_summaries[s]
        flags = [
            name
            for name, key in (("hits", "loss_hits"), ("calls", "loss_calls"), ("failure", "loss_failure"))
            if sr[key] == "1"
        ]
        if flags:
            lost.append(f"{s}:{'+'.join(flags)}")

    if len(present) == 1:
        only = sys_summaries[present[0]]
        p_hits = float(only["fisher_p"])
        p_calls = float(only["mw_p_calls"])
        rank_a, rank_b = float(only["calls_rank_mean_a"]), float(only["calls_rank_mean_b"])
        z_str = ""
        direction_val = rank_b - rank_a
    else:
        tables = []
        strata = []
        for s in present:
            mine = {
                arm: [r for r in rows if r["system"] == s and r["arm"] == arm]
                for arm in ("A", "B")
            }
            ha = sum(int(r["hit"]) for r in mine["A"])
            hb = sum(int(r["hit"]) for r in mine["B"])
            tables.append((ha, len(mine["A"]) - ha, hb, len(mine["B"]) - hb))
            strata.append(
                (
                    [float(r["evals_to_hit"]) for r in mine["A"]],
                    [float(r["evals_to_hit"]) for r in mine["B"]],
                )
            )
        p_hits = stratified_fisher(tables)
        p_calls, z = van_elteren(strata)
        z_str = f"{z:.6f}"
        direction_val = -z

    loss_hits = hits["B"] < hits["A"] and p_hits < ALPHA
    loss_calls = direction_val > 0 and p_calls < ALPHA
    complete = (
        len(present) == len(systems)
        and runs["A"] == runs["B"] == len(systems) * len(run_gate.SEEDS)
    )
    if loss_hits or loss_calls or lost:
        verdict = "FIX"
    elif complete:
        verdict = "PASS"
    else:
        verdict = "INCOMPLETE"

    opc_us_a = (out_sum["A"] / charged_sum["A"] * 1e6) if charged_sum["A"] > 0 else 0.0
    opc_us_b = (out_sum["B"] / charged_sum["B"] * 1e6) if charged_sum["B"] > 0 else 0.0
    identical_runs = sum(int(sys_summaries[s]["identical_pairs"]) for s in present)

    return {
        "panel": label,
        "systems": str(len(present)),
        "runs_a": str(runs["A"]),
        "runs_b": str(runs["B"]),
        "hits_a": str(hits["A"]),
        "hits_b": str(hits["B"]),
        "p_hits": fmt(p_hits),
        "p_calls": fmt(p_calls),
        "van_elteren_z": z_str,
        "calls_direction": (
            "B more calls" if direction_val > 0 else "B fewer calls" if direction_val < 0 else "equal"
        ),
        "wall_sum_a": f"{wall_sum['A']:.3f}",
        "wall_sum_b": f"{wall_sum['B']:.3f}",
        "wall_ratio": ratio(wall_sum["B"], wall_sum["A"]),
        "inside_sum_a": f"{pot_sum['A']:.3f}",
        "inside_sum_b": f"{pot_sum['B']:.3f}",
        "inside_ratio": ratio(pot_sum["B"], pot_sum["A"]),
        "outside_sum_a": f"{out_sum['A']:.3f}",
        "outside_sum_b": f"{out_sum['B']:.3f}",
        "outside_ratio": ratio(out_sum["B"], out_sum["A"]),
        "outside_per_call_us_a": f"{opc_us_a:.4f}",
        "outside_per_call_us_b": f"{opc_us_b:.4f}",
        "outside_per_call_ratio": ratio(opc_us_b, opc_us_a),
        "identical_runs": str(identical_runs),
        "failed_a": str(failed["A"]),
        "failed_b": str(failed["B"]),
        "lost_systems": ";".join(lost),
        "loss_hits": str(int(loss_hits)),
        "loss_calls": str(int(loss_calls)),
        "verdict": verdict,
    }


def overall_verdict(sys_rows: list[dict[str, str]], panel_rows: list[dict[str, str]]) -> str:
    lost_cells: list[str] = []
    for r in sys_rows:
        reasons = []
        if r["loss_hits"] == "1":
            reasons.append("hits")
        if r["loss_calls"] == "1":
            reasons.append("calls")
        if r["loss_failure"] == "1":
            reasons.append("failure")
        if reasons:
            lost_cells.append(f"{r['system']}({'+'.join(reasons)})")
    for p in panel_rows:
        reasons = []
        if p["loss_hits"] == "1":
            reasons.append("hits")
        if p["loss_calls"] == "1":
            reasons.append("calls")
        if reasons:
            lost_cells.append(f"panel:{p['panel']}({'+'.join(reasons)})")
    if lost_cells:
        return "FIX: " + ", ".join(lost_cells)
    if (
        len(sys_rows) != len(ALL_SYSTEMS)
        or len(panel_rows) != len(PANEL_ORDER)
        or any(r["verdict"] != "PASS" for r in sys_rows)
        or any(p["verdict"] != "PASS" for p in panel_rows)
    ):
        return "INCOMPLETE"
    return "PASS"


def generate_readme(
    rows: list[dict[str, str]],
    sys_rows: list[dict[str, str]],
    panel_rows: list[dict[str, str]],
    tip4p_id_rows: list[dict[str, str]],
    sessions: list[dict[str, str]],
    bin_rows: list[dict[str, str]],
    slot_arm_counts: dict[tuple[int, str], int],
    git_checked_count: int | None,
) -> str:
    verdict = overall_verdict(sys_rows, panel_rows)
    lines: list[str] = [
        "# nm3-land no-loss gate (part B)",
        "",
        f"**Verdict: {verdict}**",
        "",
    ]
    if sessions:
        s0 = sessions[0]
        lines.extend(
            [
                "## Provenance and Protocol",
                "",
                f"- Arm A: `{s0['arm_a_library']}` (tree `{s0['arm_a_tree']}`)",
                f"- Arm B: `{s0['arm_b_library']}` (tree `{s0['arm_b_tree']}`)",
                f"- Sessions: {len(sessions)} (first `{s0['session_id']}`)",
                f"- Machine ID (SHA-256): `{s0['machine_id_sha256']}`",
                f"- CPU: `{s0['cpu_model']}` (`{s0['arch']}`, kernel `{s0['kernel']}`, affinity `{s0['affinity']}`)",
                f"- Toolchains: `{s0['rustc_version']}`, `{s0['cargo_version']}`, Python `{s0['python_version']}`",
                f"- Seeds: `{s0['seeds']}` (n=12 per system, `PAPER_BUDGETS`, `H2H_BUDGET` unset)",
                f"- Order: `{s0['arm_order_protocol']}`; `(slot, arm)` counts across {len(rows)} rows: "
                + ", ".join(
                    f"slot {slot} arm {arm}={slot_arm_counts.get((slot, arm), 0)}"
                    for slot in (1, 2)
                    for arm in ("A", "B")
                ),
            ]
        )
        if git_checked_count is not None:
            lines.append(
                f"- Provenance verification (`git cat-file -e`): {git_checked_count}/{git_checked_count} recorded library and harness blob ids resolved."
            )
        lines.append("")

    if bin_rows:
        lines.extend(
            [
                "### Release binary digests",
                "",
                "| arm | binary | sha256 |",
                "| --- | --- | --- |",
            ]
        )
        for b in bin_rows:
            lines.append(f"| {b['arm']} | `{b['binary']}` | `{b['sha256']}` |")
        lines.append("")

    lines.extend(
        [
            "## Panels",
            "",
            "| panel | systems | hits A | hits B | Fisher / strat Fisher p | MW / van Elteren p (calls) | calls direction | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | outside/call A (us) | outside/call B (us) | outside/call B/A | same | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for p in panel_rows:
        lines.append(
            f"| {p['panel']} | {p['systems']} | {p['hits_a']}/{p['runs_a']} | {p['hits_b']}/{p['runs_b']} "
            f"| {p['p_hits']} | {p['p_calls']} | {p['calls_direction']} | {p['wall_sum_a']} | {p['wall_sum_b']} "
            f"| {p['inside_sum_a']} | {p['inside_sum_b']} | {p['outside_sum_a']} | {p['outside_sum_b']} "
            f"| {p['outside_per_call_us_a']} | {p['outside_per_call_us_b']} | {p['outside_per_call_ratio']} "
            f"| {p['identical_runs']}/{p['runs_a']} | {p['verdict']} |"
        )
    lines.extend(
        [
            "",
            "Multi-system panels report the two-sided stratified exact Fisher p on hits and the two-sided tie-corrected van Elteren test on calls to hit (misses entered at budget). Single-system panels report the system's exact Fisher p and tie-corrected Mann-Whitney p. Wall, inside, and outside are panel sums in seconds; outside/call is total outside-potential seconds divided by total charged calls in microseconds.",
            "",
            "## Systems",
            "",
            "| panel | system | budget | hits A | hits B | Fisher p | calls A | calls B | MW p calls | wall A (s) | wall B (s) | inside A (s) | inside B (s) | outside A (s) | outside B (s) | out/call A (us) | out/call B (us) | out/call B/A | verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in sys_rows:
        lines.append(
            f"| {r['panel']} | {r['system']} | {r['budget']} | {r['hits_a']}/{r['n_a']} | {r['hits_b']}/{r['n_b']} "
            f"| {r['fisher_p']} | {r['calls_median_a']} | {r['calls_median_b']} | {r['mw_p_calls']} "
            f"| {r['wall_median_a']} | {r['wall_median_b']} | {r['pot_median_a']} | {r['pot_median_b']} "
            f"| {r['outside_median_a']} | {r['outside_median_b']} | {r['outside_per_call_us_a']} "
            f"| {r['outside_per_call_us_b']} | {r['outside_per_call_ratio']} | {r['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## TIP4P bit-identity",
            "",
            "For every rigid TIP4P `(H2O)n` size (`n = 8..21`) and seed (`2000-2011`), both arms start from identical coordinates and compare `.trace.csv` bytes, `.xyz`/`.gmin` geometry bytes, best energy value/bits, and call counts (`evals_to_hit` and `charged`).",
            "",
            "| system | trajectory byte-identical | best energy identical | calls identical | all bit-identical |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    by_tip_sys: dict[str, list[dict[str, str]]] = {}
    for tr in tip4p_id_rows:
        by_tip_sys.setdefault(tr["system"], []).append(tr)
    for system in PANEL_ORDER[3][1]:
        items = by_tip_sys.get(system, [])
        n_items = len(items)
        traj_n = sum(int(i["trajectory_byte_identical"]) for i in items)
        best_n = sum(int(i["best_energy_identical"]) for i in items)
        calls_n = sum(int(i["calls_identical"]) for i in items)
        all_n = sum(int(i["all_bit_identical"]) for i in items)
        lines.append(
            f"| {system} | {traj_n}/{n_items} | {best_n}/{n_items} | {calls_n}/{n_items} | {all_n}/{n_items} |"
        )

    lines.extend(
        [
            "",
            "### TIP4P per-seed bit-identity matrix (`Y` = trajectory, best energy, and calls all byte-identical; `D` = differs)",
            "",
            "| system | " + " | ".join(str(s) for s in run_gate.SEEDS) + " |",
            "| --- | " + " | ".join("---" for _ in run_gate.SEEDS) + " |",
        ]
    )
    by_tip_pair = {(r["system"], int(r["seed"])): r for r in tip4p_id_rows}
    diffs: list[dict[str, str]] = []
    for system in PANEL_ORDER[3][1]:
        cells = []
        for seed in run_gate.SEEDS:
            item = by_tip_pair.get((system, seed))
            if item is None:
                cells.append("?")
            elif item["all_bit_identical"] == "1":
                cells.append("Y")
            else:
                cells.append("D")
                diffs.append(item)
        lines.append(f"| {system} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            f"Total TIP4P pairs all byte-identical: {len(tip4p_id_rows) - len(diffs)}/{len(tip4p_id_rows)}.",
        ]
    )
    if diffs:
        lines.extend(
            [
                "",
                "Non-identical TIP4P `(system, seed)` pairs:",
                "",
                "| system | seed | traj_id | best_id | calls_id | evals A | evals B | best A | best B |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for d in diffs:
            lines.append(
                f"| {d['system']} | {d['seed']} | {d['trajectory_byte_identical']} | {d['best_energy_identical']} "
                f"| {d['calls_identical']} | {d['evals_a']} | {d['evals_b']} | {d['best_a']} | {d['best_b']} |"
            )

    lines.extend(
        [
            "",
            "## Gate Rule",
            "",
            f"Any loss at `p < {ALPHA}` on hits (exact Fisher per system, stratified exact Fisher per multi-system panel) or calls to hit (tie-corrected Mann-Whitney per system, tie-corrected van Elteren per multi-system panel), or any B-only failure, returns `FIX` naming the failing cells; otherwise `PASS`.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--panel-summary", type=Path, required=True)
    parser.add_argument("--tip4p-identity", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--bin-sha256", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--verify-git", action="store_true")
    args = parser.parse_args(argv)

    rows = run_gate.read_csv(args.rows)
    if not rows:
        raise SystemExit("no rows to summarize")
    panels_meta = {r["system"]: r for r in run_gate.read_csv(args.panels)}
    slot_arm_counts = verify_abba_balance(rows)

    git_checked_count: int | None = None
    if args.verify_git:
        if not args.repo:
            raise SystemExit("--verify-git requires --repo")
        git_checked_count, missing = verify_provenance_git(args.repo, rows)
        if missing:
            raise SystemExit(f"missing git objects for provenance: {missing}")

    sys_rows: list[dict[str, str]] = []
    sys_map: dict[str, dict[str, str]] = {}
    for panel_label, systems in PANEL_ORDER:
        for system in systems:
            if not any(r["system"] == system for r in rows):
                continue
            sr = system_summary(panel_label, system, rows, panels_meta[system])
            sys_rows.append(sr)
            sys_map[system] = sr

    panel_rows: list[dict[str, str]] = []
    for panel_label, systems in PANEL_ORDER:
        if any(s in sys_map for s in systems):
            panel_rows.append(panel_summary(panel_label, systems, rows, sys_map))

    tip4p_id_rows = build_tip4p_identity(rows)
    sessions = run_gate.read_csv(args.sessions) if args.sessions else []
    bin_rows = run_gate.read_csv(args.bin_sha256) if args.bin_sha256 else []

    run_gate.write_csv(args.summary, SUMMARY_FIELDS, sys_rows)
    run_gate.write_csv(args.panel_summary, PANEL_SUMMARY_FIELDS, panel_rows)
    run_gate.write_csv(args.tip4p_identity, TIP4P_IDENTITY_FIELDS, tip4p_id_rows)

    readme_text = generate_readme(
        rows,
        sys_rows,
        panel_rows,
        tip4p_id_rows,
        sessions,
        bin_rows,
        slot_arm_counts,
        git_checked_count,
    )
    args.readme.parent.mkdir(parents=True, exist_ok=True)
    args.readme.write_text(readme_text, encoding="utf-8", newline="\n")
    print(overall_verdict(sys_rows, panel_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
