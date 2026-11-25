#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Resolve CPU-capped minima-hopping cells from their panel sources."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
CPU_CAP_S = 43_200.0
MORSE_REVISION = "61b5c9a01b2002223f9c1e7024863b112001ebf7"


class CellSpec(NamedTuple):
    name: str
    family: str
    size: int
    rho: float | None
    source: str
    budget: int
    reference: float


class RunPlan(NamedTuple):
    cell: CellSpec
    family: str
    runner: Path
    rows: Path
    shape_rows: Path
    work_env: str
    binary_env: str
    default_binary: Path
    revision_env: str
    revision: str
    arm: str
    seed: int
    cpu_cap_s: float

    def command(self, python: str) -> tuple[str, ...]:
        return (
            python,
            str(self.runner),
            self.cell.name,
            self.arm,
            str(self.seed),
        )


@lru_cache(maxsize=None)
def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def parse_cell(cell: str) -> CellSpec:
    morse_match = re.fullmatch(r"m(?P<size>[1-9][0-9]*)r(?P<rho>[1-9][0-9]*)", cell)
    if morse_match:
        source = _load(
            ROOT / "scripts" / "ase_nonlj" / "run_one.py",
            "ase_nonlj_cpu_source",
        )
        if cell not in source.SYSTEMS:
            raise ValueError(f"unknown cell {cell}")
        entry = source.SYSTEMS[cell]
        return CellSpec(
            cell,
            "morse",
            int(entry["n"]),
            float(entry["rho"]),
            "morse",
            int(entry["budget"]),
            float(entry["reference"]),
        )

    lj_match = re.fullmatch(r"lj(?P<size>[1-9][0-9]*)", cell)
    if lj_match:
        panel = _load(
            ROOT / "scripts" / "ase_panel" / "run_one.py",
            "ase_panel_cpu_source",
        )
        if cell in panel.SYSTEMS and panel.SYSTEMS[cell]["kind"] == "lj":
            entry = panel.SYSTEMS[cell]
            return CellSpec(
                cell,
                "lj",
                int(entry["n"]),
                None,
                "panel",
                int(entry["budget"]),
                float(entry["reference"]),
            )
        lj75 = _load(
            ROOT / "scripts" / "ase_lj75" / "run_one.py",
            "ase_lj75_cpu_source",
        )
        if int(lj_match.group("size")) == int(lj75.N):
            return CellSpec(
                cell,
                "lj",
                int(lj75.N),
                None,
                "lj75",
                int(lj75.BUDGET),
                float(lj75.REFERENCE),
            )
    raise ValueError(f"unknown cell {cell}")


def _latest_revision(rows: Path, cell: str) -> str:
    with rows.open(newline="") as handle:
        candidates = [
            row["code_revision"]
            for row in csv.DictReader(handle)
            if row["system"] == cell
            and row["arm"] == "ase_bh"
        ]
    if not candidates:
        raise ValueError(f"{cell} has no basin-hopping provenance")
    return candidates[-1]


def build_plan(root: Path, cell: str, seed: int) -> RunPlan:
    parsed = parse_cell(cell)
    if seed not in range(2000, 2012):
        raise ValueError(f"seed {seed} is outside 2000-2011")
    if parsed.source == "morse":
        runner = root / "scripts" / "ase_nonlj" / "run_one.py"
        rows = root / "bench" / "ase-nonlj" / "rows.csv"
        binary_env = "ASE_NONLJ"
        binary = root / "target" / "release" / "examples" / "ase_nonlj"
        revision_env = "ASE_NONLJ_REVISION"
    elif parsed.source == "panel":
        runner = root / "scripts" / "ase_panel" / "run_one.py"
        rows = root / "bench" / f"ase-{cell}" / "rows.csv"
        binary_env = "ASE_PANEL"
        binary = root / "target" / "release" / "examples" / "ase_panel"
        revision_env = "ASE_PANEL_REVISION"
    else:
        runner = root / "scripts" / "ase_lj75" / "run_one.py"
        rows = root / "bench" / "ase-lj75" / "rows.csv"
        binary_env = "ANNEAL_LJ75"
        binary = root / "target" / "release" / "examples" / "leave_gate"
        revision_env = "ASE_LJ75_REVISION"
    return RunPlan(
        parsed,
        parsed.family,
        runner,
        rows,
        rows.with_name("run-shape.csv"),
        "ASE_NONLJ_CPU_WORK",
        binary_env,
        binary,
        revision_env,
        MORSE_REVISION if parsed.source == "morse" else _latest_revision(rows, cell),
        "ase_mh",
        seed,
        CPU_CAP_S,
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: cpu_cap_spec.py CELL SEED", file=sys.stderr)
        return 2
    try:
        plan = build_plan(ROOT, sys.argv[1], int(sys.argv[2]))
    except (ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps({
        "family": plan.family,
        "source": plan.cell.source,
        "runner": str(plan.runner),
        "rows": str(plan.rows),
        "shape_rows": str(plan.shape_rows),
        "work_env": plan.work_env,
        "binary_env": plan.binary_env,
        "default_binary": str(plan.default_binary),
        "revision_env": plan.revision_env,
        "revision": plan.revision,
        "budget": plan.cell.budget,
        "reference": plan.cell.reference,
        "arm": plan.arm,
        "seed": plan.seed,
        "cpu_cap_s": plan.cpu_cap_s,
        "command": plan.command(sys.executable),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
