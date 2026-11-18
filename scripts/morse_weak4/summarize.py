#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Summarize and verify the morse-weak4 no-loss gate."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_PATH = Path(__file__).with_name("run.py")
RUN_SPEC = importlib.util.spec_from_file_location("morse_weak4_run", RUN_PATH)
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)
ARMS, FIELDS, SEEDS, SYSTEMS = RUN.ARMS, RUN.FIELDS, RUN.SEEDS, RUN.SYSTEMS
SHARED_PATH = ROOT / "scripts" / "morse_weak" / "summarize.py"
SHARED_SPEC = importlib.util.spec_from_file_location("morse_weak_shared", SHARED_PATH)
SHARED = importlib.util.module_from_spec(SHARED_SPEC)
SHARED_SPEC.loader.exec_module(SHARED)


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    with (ROOT / "bench" / "morse-weak4" / "rows.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise SystemExit("unexpected rows.csv columns")
        rows = list(reader)
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    seen: set[tuple[str, str, int]] = set()
    revisions: set[str] = set()
    for row in rows:
        system, arm, seed = row["system"], row["arm"], int(row["seed"])
        if system not in SYSTEMS or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        key = (system, arm, seed)
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        if row["host"] != "bench" or int(row["budget"]) != SYSTEMS[system]:
            raise SystemExit(f"host or budget mismatch {row}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
        revisions.add(row["code_revision"])
        grouped[system][arm].append(row)
    if len(revisions) != 1:
        raise SystemExit(f"both arms must share one revision, got {sorted(revisions)}")
    for system in SYSTEMS:
        for arm in ARMS:
            grouped[system][arm].sort(key=lambda row: int(row["seed"]))
            if tuple(int(row["seed"]) for row in grouped[system][arm]) != SEEDS:
                raise SystemExit(f"incomplete rows for {system} {arm}")
    return grouped


def main() -> None:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--check"):
        raise SystemExit("usage: summarize.py [--check]")
    grouped = load_rows()
    revisions = sorted(
        {
            row["code_revision"]
            for system in SYSTEMS
            for arm in ARMS
            for row in grouped[system][arm]
        }
    )
    summary = [
        "system,base_hits,fix_hits,n,base_median_calls,fix_median_calls,"
        "base_median_wall_s,fix_median_wall_s,fisher_p,calls_U,calls_p,"
        "wall_U,wall_p,wall_ratio"
    ]
    readme = [
        "Copyright (c) anneal developers",
        "",
        "# Held stream, then a forked adaptive escape",
        "",
        "Motif-free `recommended` against opt-in `recommended_adaptive` on the same",
        "binary. The adaptive arm keeps the incumbent on the single-chain proposal",
        "stream, then gives a fork of the unused stream the calls that remain.",
        "Seeds 2000-2011, host `bench`, serial `taskset -c 0`, paper call budgets,",
        "no wall cap.",
        f"code_revision {', '.join(revisions)}.",
        "Misses are censored at their call budget.",
        "",
        "| system | budget | base hits | fix hits | base med calls | fix med calls | "
        "base wall s | fix wall s | Fisher p | calls MW p | wall MW p | wall ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rejected = False
    for system, budget in SYSTEMS.items():
        base = grouped[system][ARMS[0]]
        fix = grouped[system][ARMS[1]]
        base_hits = sum(row["hit"] == "1" for row in base)
        fix_hits = sum(row["hit"] == "1" for row in fix)
        base_calls_values = [SHARED.calls(row) for row in base]
        fix_calls_values = [SHARED.calls(row) for row in fix]
        base_wall_values = [float(row["wall_s"]) for row in base]
        fix_wall_values = [float(row["wall_s"]) for row in fix]
        base_calls = SHARED.median(base_calls_values)
        fix_calls = SHARED.median(fix_calls_values)
        base_wall = SHARED.median(base_wall_values)
        fix_wall = SHARED.median(fix_wall_values)
        fisher = SHARED.fisher(fix_hits, base_hits)
        calls_u, calls_p = SHARED.mann_whitney(fix_calls_values, base_calls_values)
        wall_u, wall_p = SHARED.mann_whitney(fix_wall_values, base_wall_values)
        ratio = fix_wall / base_wall
        rejected |= fix_hits < base_hits
        rejected |= calls_p < 0.05 and fix_calls > base_calls
        rejected |= wall_p < 0.05 and fix_wall > base_wall
        summary.append(
            f"{system},{base_hits},{fix_hits},12,{base_calls:.1f},{fix_calls:.1f},"
            f"{base_wall:.6f},{fix_wall:.6f},{SHARED.g6(fisher)},"
            f"{SHARED.g6(calls_u)},{SHARED.g6(calls_p)},"
            f"{SHARED.g6(wall_u)},{SHARED.g6(wall_p)},{ratio:.6f}"
        )
        readme.append(
            f"| {system} | {budget:,} | {base_hits}/12 | {fix_hits}/12 | "
            f"{base_calls:.1f} | {fix_calls:.1f} | {base_wall:.3f} | {fix_wall:.3f} | "
            f"{SHARED.g6(fisher)} | {SHARED.g6(calls_p)} | "
            f"{SHARED.g6(wall_p)} | {ratio:.3f} |"
        )
    readme.extend(
        [
            "",
            f"Gate result: **{'rejected; adaptive remains opt-in' if rejected else 'passed'}**.",
            "A hit-count decrease rejects even when Fisher is not individually significant.",
            "Calls and wall use two-sided, tie- and continuity-corrected Mann-Whitney tests.",
            "",
        ]
    )
    out = ROOT / "bench" / "morse-weak4"
    outputs = {
        out / "summary.csv": "\n".join(summary) + "\n",
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
