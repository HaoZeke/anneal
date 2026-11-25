# Copyright (c) anneal developers
"""The CPU-cap harness derives every cell from the panel sources."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "scripts" / "ase_nonlj" / "cpu_cap_spec.py"
SPEC = importlib.util.spec_from_file_location("cpu_cap_spec", SPEC_PATH)
CPU_CAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPU_CAP)
SCRIPT = ROOT / "scripts" / "ase_nonlj" / "run_cpu_cap_seed.sh"


@pytest.mark.parametrize(
    ("cell", "family", "size", "rho"),
    [
        ("m55r6", "morse", 55, 6.0),
        ("m38r14", "morse", 38, 14.0),
        ("lj75", "lj", 75, None),
        ("lj150", "lj", 150, None),
    ],
)
def test_parser_accepts_panel_cells(cell, family, size, rho):
    parsed = CPU_CAP.parse_cell(cell)
    assert (parsed.family, parsed.size, parsed.rho) == (family, size, rho)


def test_parser_rejects_junk():
    with pytest.raises(ValueError, match="unknown cell"):
        CPU_CAP.parse_cell("junk")


def test_m55r6_keeps_the_existing_command_and_config():
    plan = CPU_CAP.build_plan(ROOT, "m55r6", 2002)
    assert plan.family == "morse"
    assert plan.runner == ROOT / "scripts" / "ase_nonlj" / "run_one.py"
    assert plan.rows == ROOT / "bench" / "ase-nonlj" / "rows.csv"
    assert plan.work_env == "ASE_NONLJ_CPU_WORK"
    assert plan.binary_env == "ASE_NONLJ"
    assert plan.revision_env == "ASE_NONLJ_REVISION"
    assert plan.revision == "61b5c9a01b2002223f9c1e7024863b112001ebf7"
    assert plan.arm == "ase_mh"
    assert plan.cpu_cap_s == 43_200.0
    assert plan.command("python3") == (
        "python3", str(plan.runner), "m55r6", "ase_mh", "2002"
    )


def test_shell_harness_has_no_absolute_work_paths():
    assert "/work/" not in SCRIPT.read_text()


def test_lj75_smoke_appends_to_temporary_rows(tmp_path):
    rows = tmp_path / "rows.csv"
    rows.write_text(
        "system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,"
        "inside_s,outside_s,code_revision,host,stop,cap_s\n"
    )
    shape_rows = tmp_path / "run-shape.csv"
    shape_rows.write_text(
        "system,arm,seed,code_revision,nproc,model_name,memory_bytes,"
        "core_affinity,cpu_s,wall_s,cpu_wall_ratio,vm_stayed_live\n"
    )
    env = {
        **os.environ,
        "LJOS_SEAT": "benchbot",
        "PYTHON": sys.executable,
        "ASE_NONLJ_CPU_WORK": str(tmp_path / "work"),
        "ASE_CPU_CAP_ROWS": str(rows),
        "ASE_CPU_CAP_SHAPE_ROWS": str(shape_rows),
        "ASE_CPU_CAP_BUDGET": "100",
        "ASE_CPU_CAP_SECONDS": "30",
    }
    subprocess.run(
        ["bash", str(SCRIPT), "lj75", "2000"],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=120,
    )
    with rows.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    assert len(recorded) == 1
    assert recorded[0]["system"] == "lj75"
    assert recorded[0]["arm"] == "ase_mh"
    assert recorded[0]["seed"] == "2000"
    assert recorded[0]["budget"] == "100"
    assert recorded[0]["cap_s"] == "30.000000"
    with shape_rows.open(newline="") as handle:
        shape = list(csv.DictReader(handle))
    assert len(shape) == 1
    assert shape[0]["system"] == "lj75"
    assert shape[0]["core_affinity"] == "0"


def test_rerun_repairs_shape_after_interrupted_append(tmp_path):
    plan = CPU_CAP.build_plan(ROOT, "lj75", 2000)
    with plan.rows.open(newline="") as handle:
        source = next(
            row for row in csv.DictReader(handle)
            if row["arm"] == "ase_mh" and row["seed"] == "2000"
        )
    source["stop"] = "cap reached (12 h CPU)"
    source["cap_s"] = "43200.000000"
    source["code_revision"] = plan.revision

    rows = tmp_path / "rows.csv"
    with rows.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source)
        writer.writeheader()
        writer.writerow(source)
    shape_rows = tmp_path / "run-shape.csv"
    shape_rows.touch()

    work = tmp_path / "work"
    stdout = work / "runs" / "lj75-ase_mh-2000.stdout"
    stdout.parent.mkdir(parents=True)
    output = []
    with stdout.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source)
        writer.writerow(source)
    output.append(stdout.read_text().strip())
    output.append("shape_json=" + json.dumps({
        "system": "lj75",
        "arm": "ase_mh",
        "seed": 2000,
        "code_revision": plan.revision,
        "nproc": 1,
        "model_name": "generic",
        "memory_bytes": 1,
        "core_affinity": "0",
        "cpu_s": 1.0,
        "wall_s": 1.0,
        "cpu_wall_ratio": 1.0,
        "vm_stayed_live": True,
    }))
    stdout.write_text("\n".join(output) + "\n")

    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "ASE_NONLJ_CPU_WORK": str(work),
        "ASE_CPU_CAP_ROWS": str(rows),
        "ASE_CPU_CAP_SHAPE_ROWS": str(shape_rows),
    }
    subprocess.run(
        ["bash", str(SCRIPT), "lj75", "2000"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    with rows.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with shape_rows.open(newline="") as handle:
        shape = list(csv.DictReader(handle))
    assert len(shape) == 1
    assert shape[0]["cpu_s"] == "1.0"
