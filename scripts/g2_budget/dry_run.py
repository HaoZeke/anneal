#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Exercise one cell's row writer with a scaled cap.

G2_BUDGET_TEST_CAP_S is read only here. The Snakemake record rule does
not use it. Output stays under --out, outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import os
import subprocess
import sys
import time
from pathlib import Path

TEST_CAP_ENV = "G2_BUDGET_TEST_CAP_S"
# Enough arithmetic that a scaled cap stops before the paper budget.
WORK_STEPS = 40


def _load(name: str, path: Path):
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve() == path:
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[2]
protocol = _load("g2_budget_protocol", Path(__file__).resolve().with_name("protocol.py"))
rows = _load("g2_budget_rows", Path(__file__).resolve().with_name("rows.py"))


def _scratch(out: Path) -> Path:
    resolved = out.resolve()
    repo = ROOT.resolve()
    if resolved == repo or repo in resolved.parents or resolved in repo.parents:
        raise SystemExit("dry run refuses to write inside the repository")
    return resolved


def _test_cap() -> float:
    raw = os.environ.get(TEST_CAP_ENV)
    if raw is None:
        raise SystemExit(f"{TEST_CAP_ENV} is required")
    try:
        cap = float(raw)
    except ValueError as error:
        raise SystemExit(f"{TEST_CAP_ENV} must be a number of seconds") from error
    if not 0 < cap < protocol.CPU_CAP_S:
        raise SystemExit("test cap must be a positive scale-down of the pre-registered cap")
    return cap


def _work(calls: int) -> int:
    value = calls
    for _ in range(WORK_STEPS):
        value = (value * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
    return value


def _burn(budget: int, cap_s: float | None, calls: int) -> tuple[int, str, float, float, float]:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    value = 0
    stop = "budget"
    while calls < budget:
        calls += 1
        value = _work(calls)
        if (
            cap_s is not None
            and (calls & 127) == 0
            and time.process_time() - started_cpu >= cap_s
        ):
            stop = "cap"
            break
    best = -float(value % 100_000) / 1000.0
    return (
        calls,
        stop,
        best,
        time.process_time() - started_cpu,
        time.perf_counter() - started_wall,
    )


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def run_cell(system: str, arm: str, seed: int, out: Path) -> dict[str, str]:
    cell = protocol.find_cell(system, arm, seed)
    if cell.stop_rule == "cap":
        cap = _test_cap()
        half = cap / 2
        calls, stop, best, cpu_s, wall_s = _burn(cell.budget, half, 0)
        if stop != "cap":
            raise SystemExit("scaled cap did not stop the dry run before the budget")
        first = rows.Segment(calls, best, cpu_s, wall_s, "checkpoint")
        calls, stop, best, cpu_s, wall_s = _burn(cell.budget, half, calls)
        if stop != "cap":
            raise SystemExit("scaled cap did not stop the resumed dry run before the budget")
        segments = [first, rows.Segment(calls, best, cpu_s, wall_s, "cap")]
    else:
        calls, stop, best, cpu_s, wall_s = _burn(cell.budget, None, 0)
        if stop != "budget" or calls != cell.budget:
            raise SystemExit("budget dry run did not spend the paper budget")
        segments = [rows.Segment(calls, best, cpu_s, wall_s, "budget")]
    return rows.record_dry_run(_scratch(out), cell, segments, _revision())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run one G2 budget-stop cell.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    _scratch(Path(args.out))
    row = run_cell(args.system, args.arm, args.seed, Path(args.out))
    print(",".join(rows.ROW_FIELDS))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=rows.ROW_FIELDS, lineterminator="\n")
    writer.writerow(row)
    sys.stdout.write(buffer.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
