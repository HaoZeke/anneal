#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Generate the morse-weak2 current-vs-new recommended gate."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED_PATH = ROOT / "scripts" / "morse_weak" / "summarize.py"
SHARED_SPEC = importlib.util.spec_from_file_location("morse_weak_shared", SHARED_PATH)
SHARED = importlib.util.module_from_spec(SHARED_SPEC)
SHARED_SPEC.loader.exec_module(SHARED)

SEEDS = tuple(range(2000, 2012))
SYSTEMS = (
    "m38r3", "m38r6", "m38r10", "m38r14",
    "m55r3", "m55r6", "m55r10", "m55r14",
    "m75r6", "m75r14", "m80r6", "m80r14",
    "lj38", "lj55", "lj75", "lj98", "lj102", "lj103", "lj104", "lj150", "au55",
)
ARMS = ("current_recommended", "new_recommended")
SPEC = {
    **SHARED.SPEC,
    "lj102": ("4000000", -569.363652),
    "lj103": ("4000000", -575.766131),
    "lj150": ("4000000", -893.310258),
    "au55": ("1000000", -191.03085280),
}
ROW_FIELDS = SHARED.ROW_FIELDS
HIT_TOL = SHARED.HIT_TOL


def load_rows() -> dict[str, dict[str, list[dict[str, str]]]]:
    path = ROOT / "bench" / "morse-weak2" / "rows.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise SystemExit(f"{path}: unexpected columns")
        rows = list(reader)
    grouped = {system: {arm: [] for arm in ARMS} for system in SYSTEMS}
    seen = set()
    for row in rows:
        system, arm = row["system"], row["arm"]
        if system not in SPEC or arm not in ARMS:
            raise SystemExit(f"unexpected row {row}")
        key = (system, arm, row["seed"])
        if key in seen:
            raise SystemExit(f"duplicate row {key}")
        seen.add(key)
        budget, reference = SPEC[system]
        if row["budget"] != budget or row["host"] != "bench":
            raise SystemExit(f"budget or host mismatch {row}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["code_revision"]):
            raise SystemExit(f"invalid revision {row}")
        charged = int(row["charged"])
        hit = row["hit"] == "1"
        if not 0 < charged <= int(budget) or hit != bool(row["evals_to_hit"]):
            raise SystemExit(f"invalid calls {row}")
        if hit:
            if row["stop"] != "hit" or float(row["best"]) > reference + HIT_TOL + 5.0e-9:
                raise SystemExit(f"invalid hit {row}")
        elif row["stop"] != "budget" or charged != int(budget):
            raise SystemExit(f"invalid miss {row}")
        grouped[system][arm].append(row)
    for system in SYSTEMS:
        for arm in ARMS:
            rows = grouped[system][arm]
            rows.sort(key=lambda row: int(row["seed"]))
            if tuple(int(row["seed"]) for row in rows) != SEEDS:
                raise SystemExit(f"{system} {arm} has incomplete seeds")
    return grouped


def main() -> None:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] != "--check"):
        raise SystemExit("usage: summarize.py [--check]")
    grouped = load_rows()
    revisions = sorted({row["code_revision"] for s in SYSTEMS for a in ARMS for row in grouped[s][a]})
    summary = [
        "system,base_hits,fix_hits,n,base_median_calls,fix_median_calls,"
        "base_median_wall_s,fix_median_wall_s,fisher_p,mann_whitney_U,mann_whitney_p,wall_ratio"
    ]
    readme = [
        "Copyright (c) anneal developers", "",
        "# Default adaptive-search gate", "",
        "Same-session current `Config::recommended` against the proposed adaptive default.",
        "Seeds 2000-2011, host `bench`, serial `taskset -c 0`, paper call budgets, no wall cap.",
        f"code_revision {', '.join(revisions)}.",
        "Misses are censored at their call budget.", "",
        "| system | budget | base hits | fix hits | base med calls | fix med calls | "
        "base wall s | fix wall s | Fisher p | U | MW p | wall ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system in SYSTEMS:
        base = grouped[system][ARMS[0]]
        fix = grouped[system][ARMS[1]]
        base_hits = sum(row["hit"] == "1" for row in base)
        fix_hits = sum(row["hit"] == "1" for row in fix)
        base_calls = SHARED.median([SHARED.calls(row) for row in base])
        fix_calls = SHARED.median([SHARED.calls(row) for row in fix])
        base_wall = SHARED.median([float(row["wall_s"]) for row in base])
        fix_wall = SHARED.median([float(row["wall_s"]) for row in fix])
        fisher = SHARED.fisher(fix_hits, base_hits)
        u_stat, mw_p = SHARED.mann_whitney(
            [SHARED.calls(row) for row in fix], [SHARED.calls(row) for row in base]
        )
        ratio = fix_wall / base_wall
        summary.append(
            f"{system},{base_hits},{fix_hits},12,{base_calls:.1f},{fix_calls:.1f},"
            f"{base_wall:.6f},{fix_wall:.6f},{SHARED.g6(fisher)},"
            f"{SHARED.g6(u_stat)},{SHARED.g6(mw_p)},{ratio:.6f}"
        )
        readme.append(
            f"| {system} | {int(SPEC[system][0]):,} | {base_hits}/12 | {fix_hits}/12 | "
            f"{base_calls:.1f} | {fix_calls:.1f} | {base_wall:.3f} | {fix_wall:.3f} | "
            f"{SHARED.g6(fisher)} | {SHARED.g6(u_stat)} | {SHARED.g6(mw_p)} | {ratio:.3f} |"
        )
    readme.extend([
        "", "Medians are Hyndman-Fan type 7. Fisher is two-sided exact on hits.",
        "Mann-Whitney is two-sided asymptotic on calls with tie and continuity corrections.", "",
    ])
    out = ROOT / "bench" / "morse-weak2"
    outputs = {
        out / "summary.csv": "\n".join(summary) + "\n",
        out / "README.md": "\n".join(readme),
    }
    if len(sys.argv) == 2:
        stale = [str(path.relative_to(ROOT)) for path, text in outputs.items() if path.read_text() != text]
        if stale:
            raise SystemExit("stale generated report: " + ", ".join(stale))
    else:
        for path, text in outputs.items():
            path.write_text(text)


if __name__ == "__main__":
    main()
