# Copyright (c) anneal developers
"""Amendment A1 is lj75 and lj150 on ase_mh, apart from the five systems."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "g2_budget" / "protocol.py"
ROWS_PATH = ROOT / "scripts" / "g2_budget" / "rows.py"
RECORD_PATH = ROOT / "scripts" / "g2_budget" / "record_rule.py"
DRY_PATH = ROOT / "scripts" / "g2_budget" / "dry_run.py"
RUNNER_PATH = ROOT / "scripts" / "g2_budget" / "run_cell.py"
SNAKEFILE = ROOT / "workflow" / "Snakefile"
README = ROOT / "bench" / "g2-budget" / "README.md"
TEST_CAP_ENV = "G2_BUDGET_TEST_CAP_S"
PREREG = "1308d8678a8912de95ded423291a25430bb073fc"
REVISION = "a" * 40


def load(name: str, path: Path):
    loaded = sys.modules.get(name)
    if loaded is not None and Path(loaded.__file__).resolve() == path.resolve():
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = load("g2_budget_protocol", PROTOCOL_PATH)
ROWS = load("g2_budget_rows", ROWS_PATH)


def _dry(out: Path, system: str, arm: str, seed: str):
    return subprocess.run(
        [
            sys.executable,
            str(DRY_PATH),
            "--system", system,
            "--arm", arm,
            "--seed", seed,
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, TEST_CAP_ENV: "0.25"},
        timeout=60,
    )


def test_preregistered_budgets_and_stop_rules_stay_fixed():
    assert PROTOCOL.PAPER_BUDGETS == {
        "lj38": 400_000,
        "lj55": 1_000_000,
        "lj98": 4_000_000,
        "lj104": 4_000_000,
        "au55": 1_000_000,
    }
    assert tuple(PROTOCOL.SYSTEMS) == ("lj38", "lj55", "lj98", "lj104", "au55")
    assert "lj75" not in PROTOCOL.SYSTEMS
    assert "lj150" not in PROTOCOL.SYSTEMS
    assert PROTOCOL.CPU_CAP_S == 43_200
    assert PROTOCOL.ARMS == ("ase_bh", "ase_mh", "anneal")
    cells = PROTOCOL.cells()
    assert len(cells) == 5 * 3 * 12
    assert all(cell.system not in {"lj75", "lj150"} for cell in cells)
    capped = {
        (cell.system, cell.arm)
        for cell in cells
        if cell.stop_rule == "cap"
    }
    assert capped == {
        ("lj98", "ase_mh"),
        ("lj98", "anneal"),
        ("lj104", "ase_mh"),
        ("lj104", "anneal"),
        ("au55", "ase_mh"),
        ("au55", "anneal"),
    }
    for system, budget in PROTOCOL.PAPER_BUDGETS.items():
        for arm in PROTOCOL.ARMS:
            rule = "cap" if (system, arm) in capped else "budget"
            assert PROTOCOL.stop_rule(system, arm) == rule
            cap = PROTOCOL.cpu_cap_s(system, arm)
            assert cap == (43_200 if rule == "cap" else None)
            for seed in (2000, 2011):
                cell = PROTOCOL.find_cell(system, arm, seed)
                assert cell.budget == budget
                assert cell.stop_rule == rule
                assert cell.cpu_cap_s == cap
    assert ROWS.ROW_FIELDS == (
        "system",
        "arm",
        "seed",
        "budget",
        "stop",
        "calls_reached",
        "fraction_of_budget",
        "best_e",
        "cpu_s",
        "wall_s",
        "code_revision",
    )
    listed = subprocess.run(
        [sys.executable, str(RECORD_PATH), "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in listed.stdout.splitlines() if line.strip()]
    assert records == [PROTOCOL.cell_record(cell) for cell in cells]
    assert all("lj75" not in line and "lj150" not in line for line in listed.stdout.splitlines())
    assert all("amendment" not in record for record in records)


def test_amendment_sources_are_the_existing_budgets_and_references():
    lj75 = (ROOT / "scripts" / "ase_lj75" / "run_one.py").read_text().splitlines()
    assert lj75[38] == "PAPER_BUDGETS = {75: 4_000_000}"
    assert lj75[39] == "BUDGET = PAPER_BUDGETS[75]"
    assert lj75[40] == "REFERENCE = -397.492331"
    panel = (ROOT / "scripts" / "ase_panel" / "run_one.py").read_text().splitlines()
    assert panel[45] == "PAPER_BUDGETS = {"
    assert panel[50] == "    150: 4_000_000,"
    assert panel[57] == (
        '    "lj150": {"n": 150, "budget": 4_000_000, "reference": -893.310258, "kind": "lj"},'
    )
    cap_spec = (ROOT / "scripts" / "ase_nonlj" / "cpu_cap_spec.py").read_text().splitlines()
    assert "int(lj75.BUDGET)," == cap_spec[118].strip()
    assert "float(lj75.REFERENCE)," == cap_spec[119].strip()
    gmin = (ROOT / "scripts" / "gmin_panel" / "run_panel.py").read_text().splitlines()
    assert "Cambridge Cluster Database" in gmin[78]
    assert gmin[83].strip().startswith('("lj75", "lj", 75, -397.492331,')
    assert gmin[88].strip().startswith('("lj150", "lj", 150, -893.310258,')
    assert getattr(PROTOCOL, "AMENDMENT_ID", None) == "A1"
    assert getattr(PROTOCOL, "AMENDMENT_BUDGETS", None) == {
        "lj75": 4_000_000,
        "lj150": 4_000_000,
    }
    assert getattr(PROTOCOL, "AMENDMENT_REFERENCES", None) == {
        "lj75": -397.492331,
        "lj150": -893.310258,
    }
    assert getattr(PROTOCOL, "AMENDMENT_ATOMS", None) == {"lj75": 75, "lj150": 150}


def test_runner_accepts_only_ase_mh_for_the_amendment():
    assert len(getattr(PROTOCOL, "amendment_cells", lambda: ())()) == 2 * 12
    for system in ("lj75", "lj150"):
        for seed in range(2000, 2012):
            cell = PROTOCOL.find_cell(system, "ase_mh", seed)
            assert cell.system == system
            assert cell.arm == "ase_mh"
            assert cell.seed == seed
            assert cell.budget == 4_000_000
            assert cell.stop_rule == "cap"
            assert cell.cpu_cap_s == 43_200
            assert cell.preset is None
            assert PROTOCOL.stop_rule(system, "ase_mh") == PROTOCOL.stop_rule("lj98", "ase_mh")
            assert PROTOCOL.cpu_cap_s(system, "ase_mh") == PROTOCOL.cpu_cap_s("lj98", "ase_mh")
            assert PROTOCOL.rows_path(ROOT, system) == (
                ROOT / "bench" / f"ase-{system}" / "rows-ljfix-budget.csv"
            )
            assert PROTOCOL.shapes_path(ROOT, system) == (
                ROOT / "bench" / f"ase-{system}" / "shapes-ljfix-budget.jsonl"
            )
        for seed in (1999, 2012):
            with pytest.raises(ValueError):
                PROTOCOL.find_cell(system, "ase_mh", seed)
        for arm in ("ase_bh", "anneal"):
            with pytest.raises(ValueError):
                PROTOCOL.find_cell(system, arm, 2000)
            with pytest.raises(ValueError):
                PROTOCOL.stop_rule(system, arm)
            with pytest.raises(ValueError):
                PROTOCOL.cpu_cap_s(system, arm)


def test_amendment_plan_is_separate_from_the_campaign_list(tmp_path):
    listed = subprocess.run(
        [sys.executable, str(RECORD_PATH), "--amendment"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    records = [json.loads(line) for line in listed.stdout.splitlines() if line.strip()]
    expect = [PROTOCOL.cell_record(cell) for cell in PROTOCOL.amendment_cells()]
    assert records == expect
    assert len(records) == 24
    assert {record["system"] for record in records} == {"lj75", "lj150"}
    assert {record["arm"] for record in records} == {"ase_mh"}
    assert all(record["amendment"] == "A1" for record in records)
    assert all(record["stop_rule"] == "cap" and record["cpu_cap_s"] == 43_200 for record in records)
    assert "lj38" not in listed.stdout
    stamp = tmp_path / "cell.json"
    accepted = subprocess.run(
        [
            sys.executable,
            str(RECORD_PATH),
            "--system", "lj150",
            "--arm", "ase_mh",
            "--seed", "2011",
            "--stamp", str(stamp),
        ],
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0, accepted.stderr
    plan = json.loads(stamp.read_text())
    assert plan["amendment"] == "A1"
    assert plan["budget"] == 4_000_000
    assert plan["stop_rule"] == "cap"
    assert plan["rows"] == "bench/ase-lj150/rows-ljfix-budget.csv"
    refused = subprocess.run(
        [
            sys.executable,
            str(RECORD_PATH),
            "--system", "lj75",
            "--arm", "ase_bh",
            "--seed", "2000",
            "--stamp", str(tmp_path / "no.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert not (tmp_path / "no.json").exists()


@pytest.mark.parametrize("system", ["lj75", "lj150"])
def test_amendment_dry_run_rows_are_marked(system, tmp_path):
    completed = _dry(tmp_path, system, "ase_mh", "2000")
    assert completed.returncode == 0, completed.stderr
    row = next(csv.DictReader(completed.stdout.splitlines()))
    assert row["system"] == system
    assert row["arm"] == "ase_mh"
    assert row["seed"] == "2000"
    assert row["budget"] == "4000000"
    assert row["stop"] == "cap"
    assert 0 <= int(row["calls_reached"]) < 4_000_000
    assert list(row) == list(ROWS.ROW_FIELDS)
    shape_path = tmp_path / "bench" / f"ase-{system}" / "shapes-ljfix-budget.jsonl"
    rows_path = tmp_path / "bench" / f"ase-{system}" / "rows-ljfix-budget.csv"
    assert rows_path.is_file()
    assert shape_path.is_file()
    shape = json.loads(shape_path.read_text())
    assert shape["amendment"] == "A1"
    assert shape["system"] == system
    assert shape["segments"] == 2
    assert shape["stop"] == "cap"
    assert list(ROOT.glob("bench/ase-*/rows-ljfix-budget.csv")) == []


@pytest.mark.parametrize("system,arm", [("lj75", "ase_bh"), ("lj75", "anneal"), ("lj150", "ase_bh"), ("lj150", "anneal")])
def test_amendment_dry_run_rejects_other_arms(system, arm, tmp_path):
    completed = _dry(tmp_path, system, arm, "2000")
    assert completed.returncode != 0
    assert not (tmp_path / "bench" / f"ase-{system}" / "rows-ljfix-budget.csv").exists()
    assert list(ROOT.glob("bench/ase-*/rows-ljfix-budget.csv")) == []


def test_preregistered_shape_has_no_amendment_marker(tmp_path):
    cell = PROTOCOL.find_cell("lj38", "ase_bh", 2000)
    row = ROWS.record_dry_run(
        tmp_path,
        cell,
        [ROWS.Segment(cell.budget, -1.0, 0.1, 0.2, "budget")],
        REVISION,
    )
    assert "amendment" not in row
    assert list(row) == list(ROWS.ROW_FIELDS)
    shape = json.loads(PROTOCOL.shapes_path(tmp_path, "lj38").read_text())
    assert "amendment" not in shape
    assert shape["system"] == "lj38"
    assert shape["stop"] == "budget"
    plain = PROTOCOL.cell_record(cell)
    assert "amendment" not in plain


def test_runner_spec_keeps_panel_objects_for_the_five_systems():
    runner = load("g2_budget_run_cell", RUNNER_PATH)
    assert runner.PREREG_REVISION == PREREG
    resolve = getattr(runner, "_system_spec", None)
    assert resolve is not None
    panel = SimpleNamespace(SYSTEMS={
        "lj38": {"n": 38, "budget": 400_000, "reference": -173.928427, "kind": "lj"},
        "lj55": {"n": 55, "budget": 1_000_000, "reference": -279.248470, "kind": "lj"},
        "lj98": {"n": 98, "budget": 4_000_000, "reference": -543.665361, "kind": "lj"},
        "lj104": {"n": 104, "budget": 4_000_000, "reference": -582.086642, "kind": "lj"},
        "lj150": {"n": 150, "budget": 4_000_000, "reference": -893.310258, "kind": "lj"},
        "au55": {
            "n": 55,
            "budget": 1_000_000,
            "reference": -191.03085280,
            "kind": "gupta",
            "symbol": "Au",
        },
    })
    for system in PROTOCOL.SYSTEMS:
        assert resolve(panel, SimpleNamespace(system=system)) is panel.SYSTEMS[system]
    assert resolve(panel, SimpleNamespace(system="lj150")) is panel.SYSTEMS["lj150"]
    assert resolve(panel, SimpleNamespace(system="lj75")) == {
        "n": 75,
        "budget": 4_000_000,
        "reference": -397.492331,
        "kind": "lj",
    }
    bad = SimpleNamespace(SYSTEMS={
        "lj150": {"n": 150, "budget": 1, "reference": -893.310258, "kind": "lj"},
    })
    with pytest.raises(RuntimeError):
        resolve(bad, SimpleNamespace(system="lj150"))


def test_readme_and_workflow_keep_amendment_a1_separate():
    text = README.read_text()
    marker = "## Amendment A1 (post-registration)"
    assert marker in text
    section = text.split(marker, 1)[1]
    for phrase in (
        "1308d867",
        "lj75",
        "lj150",
        "ase_mh",
        "ase_bh",
        "anneal",
        "43200",
        "rows-ljfix-budget.csv",
        "shapes-ljfix-budget.jsonl",
        "reported separately",
        "-397.492331",
        "-893.310258",
        "4_000_000",
        "scripts/ase_lj75/run_one.py",
        "scripts/ase_panel/run_one.py",
        "scripts/gmin_panel/run_panel.py",
        "scripts/ase_nonlj/cpu_cap_spec.py",
    ):
        assert phrase in section
    assert "pip install maturin" in text
    assert "maturin develop --release" in text
    assert "maturin develop --release --features python" in text
    assert "cargo build --release --example asg_eval" in text
    assert "ASG_EVAL" in text
    snake = SNAKEFILE.read_text()
    assert "rule g2_amend_a1:" in snake
    assert "rule g2_amend_a1_cell:" in snake
    assert "g2_protocol.amendment_cells()" in snake
    assert snake.count('system="lj38|lj55|lj98|lj104|au55"') == 1
    registry = json.loads((ROOT / "bench" / "h2h.json").read_text())
    assert registry["not_h2h"]["g2_amend_a1"]
    assert registry["not_h2h"]["g2_amend_a1_cell"]
    assert registry["not_h2h"]["g2_budget"]
    assert registry["not_h2h"]["g2_budget_cell"]
