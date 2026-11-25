# Copyright (c) anneal developers
"""The G2 budget-stop campaign is fixed before any seed is run."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "g2_budget" / "protocol.py"
ROWS_PATH = ROOT / "scripts" / "g2_budget" / "rows.py"
RECORD_PATH = ROOT / "scripts" / "g2_budget" / "record_rule.py"
DRY_PATH = ROOT / "scripts" / "g2_budget" / "dry_run.py"
SNAKEFILE = ROOT / "workflow" / "Snakefile"
README = ROOT / "bench" / "g2-budget" / "README.md"
PANEL_PATH = ROOT / "scripts" / "ase_panel" / "run_one.py"
REVISION = "a" * 40
TEST_CAP_ENV = "G2_BUDGET_TEST_CAP_S"
HEDGES = (
    "unfortunately",
    "might",
    "maybe",
    "approximately",
    "roughly",
    "best effort",
    "if possible",
    "hopefully",
    "perhaps",
    "likely",
)


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


def panel_module():
    """Read the panel budgets without importing the ASE runner."""
    namespace = {"__builtins__": {}}
    for node in ast.parse(PANEL_PATH.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if not all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
            continue
        if any(target.id in {"PAPER_BUDGETS", "SYSTEMS"} for target in node.targets):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(PANEL_PATH), "exec")
            exec(code, namespace)  # noqa: S102
    return SimpleNamespace(**namespace)


def g2_snakefile() -> str:
    text = SNAKEFILE.read_text()
    marker = "# Pre-registered G2 budget-stop campaign"
    assert marker in text
    return text.split(marker, 1)[1]


def reason_section() -> str:
    text = README.read_text()
    marker = "## Why the cap is 12 h"
    assert marker in text
    return text.split(marker, 1)[1]


def test_cell_list_matches_the_paper_budgets():
    panel = panel_module()
    assert PROTOCOL.PAPER_BUDGETS == {
        "lj38": 400_000,
        "lj55": 1_000_000,
        "lj98": 4_000_000,
        "lj104": 4_000_000,
        "au55": 1_000_000,
    }
    assert PROTOCOL.PAPER_BUDGETS["lj38"] == panel.PAPER_BUDGETS[38]
    assert PROTOCOL.PAPER_BUDGETS["lj55"] == panel.PAPER_BUDGETS[55]
    assert PROTOCOL.PAPER_BUDGETS["lj98"] == panel.PAPER_BUDGETS[98]
    assert PROTOCOL.PAPER_BUDGETS["lj104"] == panel.PAPER_BUDGETS[104]
    assert PROTOCOL.PAPER_BUDGETS["au55"] == panel.SYSTEMS["au55"]["budget"]
    assert tuple(PROTOCOL.SYSTEMS) == ("lj38", "lj55", "lj98", "lj104", "au55")
    assert "lj150" not in PROTOCOL.SYSTEMS
    assert "lj75" not in PROTOCOL.SYSTEMS
    assert PROTOCOL.SEEDS == tuple(range(2000, 2012))
    assert PROTOCOL.ARMS == ("ase_bh", "ase_mh", "anneal")
    assert PROTOCOL.ANNEAL_PRESET == "for_cluster"
    assert PROTOCOL.CORES == 1
    assert PROTOCOL.SERIAL is True
    assert PROTOCOL.ONE_PER_VM is True
    assert PROTOCOL.WALL_CAP_S is None

    cells = PROTOCOL.cells()
    assert len(cells) == 5 * 3 * 12
    assert [(cell.system, cell.arm, cell.seed) for cell in cells] == [
        (system, arm, seed)
        for system in PROTOCOL.SYSTEMS
        for arm in PROTOCOL.ARMS
        for seed in PROTOCOL.SEEDS
    ]
    for cell in cells:
        assert cell.budget == PROTOCOL.PAPER_BUDGETS[cell.system]
        assert cell.cores == 1
        assert cell.wall_cap_s is None
        assert cell.preset == ("for_cluster" if cell.arm == "anneal" else None)
        assert PROTOCOL.rows_path(ROOT, cell.system) == (
            ROOT / "bench" / f"ase-{cell.system}" / "rows-ljfix-budget.csv"
        )
        assert PROTOCOL.shapes_path(ROOT, cell.system) == (
            ROOT / "bench" / f"ase-{cell.system}" / "shapes-ljfix-budget.jsonl"
        )


def test_cap_applies_to_mh_and_anneal_on_the_three_large_systems():
    capped = {
        (cell.system, cell.arm)
        for cell in PROTOCOL.cells()
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
    assert PROTOCOL.CPU_CAP_S == 43_200
    for cell in PROTOCOL.cells():
        if (cell.system, cell.arm) in capped:
            assert cell.cpu_cap_s == 43_200
            assert cell.stop_rule == "cap"
        else:
            assert cell.cpu_cap_s is None
            assert cell.stop_rule == "budget"
    budget_cells = [cell for cell in PROTOCOL.cells() if cell.stop_rule == "budget"]
    cap_cells = [cell for cell in PROTOCOL.cells() if cell.stop_rule == "cap"]
    assert len(cap_cells) == 72
    assert len(budget_cells) == 108
    assert all(cell.arm == "ase_bh" or cell.system in {"lj38", "lj55"} for cell in budget_cells)


def test_no_pair_mixes_stop_rules():
    expected = {
        "lj38": (("ase_bh", "ase_mh"), ("ase_bh", "anneal"), ("ase_mh", "anneal")),
        "lj55": (("ase_bh", "ase_mh"), ("ase_bh", "anneal"), ("ase_mh", "anneal")),
        "lj98": (("ase_mh", "anneal"),),
        "lj104": (("ase_mh", "anneal"),),
        "au55": (("ase_mh", "anneal"),),
    }
    assert {system: PROTOCOL.pairs(system) for system in PROTOCOL.SYSTEMS} == expected
    for system, pairs in expected.items():
        for left, right in pairs:
            assert PROTOCOL.stop_rule(system, left) == PROTOCOL.stop_rule(system, right)
            PROTOCOL.assert_pair(system, left, right)
    for system in ("lj98", "lj104", "au55"):
        with pytest.raises(ValueError):
            PROTOCOL.assert_pair(system, "ase_bh", "ase_mh")
        with pytest.raises(ValueError):
            PROTOCOL.assert_pair(system, "ase_bh", "anneal")
        assert ("ase_bh", "ase_mh") not in PROTOCOL.pairs(system)
        assert ("ase_bh", "anneal") not in PROTOCOL.pairs(system)


def test_capped_stop_writes_calls_and_fraction(tmp_path):
    cell = PROTOCOL.find_cell("lj98", "ase_mh", 2000)
    calls = 1000
    row = ROWS.append_result(
        tmp_path,
        cell,
        [ROWS.Segment(calls, -532.5, 1.5, 1.75, "cap")],
        REVISION,
    )
    budget = PROTOCOL.PAPER_BUDGETS["lj98"]
    assert row["stop"] == "cap"
    assert row["calls_reached"] == str(calls)
    assert row["fraction_of_budget"] == "0.00025"
    assert Decimal(row["fraction_of_budget"]) == Decimal(calls) / Decimal(budget)
    assert row["best_e"] == "-532.50000000"
    assert row["cpu_s"] == "1.500000"
    assert row["wall_s"] == "1.750000"
    assert row["code_revision"] == REVISION
    assert "host" not in ROWS.ROW_FIELDS

    rows_file = PROTOCOL.rows_path(tmp_path, "lj98")
    shapes_file = PROTOCOL.shapes_path(tmp_path, "lj98")
    assert rows_file.name == "rows-ljfix-budget.csv"
    assert shapes_file.name == "shapes-ljfix-budget.jsonl"
    assert PROTOCOL.is_scope1_rows(rows_file.with_name("rows-ljfix.csv"))
    assert not PROTOCOL.is_scope1_rows(rows_file)
    with rows_file.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    assert recorded == [row]
    assert list(recorded[0]) == list(ROWS.ROW_FIELDS)
    shape = json.loads(shapes_file.read_text())
    assert shape["stop"] == "cap"
    assert shape["calls_reached"] == str(calls)
    assert shape["fraction_of_budget"] == row["fraction_of_budget"]
    assert shape["segments"] == 1
    assert not (tmp_path / "bench" / "ase-lj98" / "rows-ljfix.csv").exists()


def test_budget_and_hit_stops(tmp_path):
    budget_cell = PROTOCOL.find_cell("lj38", "ase_bh", 2000)
    budget = PROTOCOL.PAPER_BUDGETS["lj38"]
    budget_row = ROWS.append_result(
        tmp_path,
        budget_cell,
        [ROWS.Segment(budget, -173.928427, 2.0, 2.25, "budget")],
        REVISION,
    )
    assert budget_row["stop"] == "budget"
    assert budget_row["calls_reached"] == str(budget)
    assert budget_row["fraction_of_budget"] == "1"

    hit_root = tmp_path / "hit"
    hit_cell = PROTOCOL.find_cell("lj98", "anneal", 2011)
    hit_row = ROWS.append_result(
        hit_root,
        hit_cell,
        [ROWS.Segment(120, -543.665361, 0.5, 0.6, "hit")],
        REVISION,
    )
    assert hit_row["stop"] == "hit"
    assert hit_row["calls_reached"] == "120"
    assert Decimal(hit_row["fraction_of_budget"]) == Decimal(120) / Decimal(4_000_000)
    assert hit_row["arm"] == "anneal"

    with pytest.raises(ValueError):
        ROWS.build_row(
            budget_cell,
            [ROWS.Segment(10, -1.0, 1.0, 1.0, "cap")],
            REVISION,
        )


def test_resumed_capped_seed_sums_cpu(tmp_path):
    cell = PROTOCOL.find_cell("au55", "anneal", 2007)
    row = ROWS.append_result(
        tmp_path,
        cell,
        [
            ROWS.Segment(100, -10.0, 10.0, 11.0, "checkpoint"),
            ROWS.Segment(250, -12.5, 3.5, 4.0, "cap"),
        ],
        REVISION,
    )
    assert row["stop"] == "cap"
    assert row["calls_reached"] == "250"
    assert Decimal(row["fraction_of_budget"]) == Decimal(250) / Decimal(1_000_000)
    assert row["cpu_s"] == "13.500000"
    assert row["wall_s"] == "15.000000"
    assert row["best_e"] == "-12.50000000"
    shape = json.loads(PROTOCOL.shapes_path(tmp_path, "au55").read_text())
    assert shape["segments"] == 2
    assert shape["cpu_s"] == "13.500000"

    with PROTOCOL.rows_path(tmp_path, "au55").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with pytest.raises(ValueError):
        ROWS.append_result(
            tmp_path,
            cell,
            [ROWS.Segment(250, -12.5, 1.0, 1.0, "cap")],
            REVISION,
        )

    budget_cell = PROTOCOL.find_cell("lj55", "ase_mh", 2003)
    with pytest.raises(ValueError, match="capped"):
        ROWS.build_row(
            budget_cell,
            [
                ROWS.Segment(10, -1.0, 1.0, 1.0, "checkpoint"),
                ROWS.Segment(20, -2.0, 1.0, 1.0, "budget"),
            ],
            REVISION,
        )


def test_torn_tail_is_dropped_before_append(tmp_path):
    cell = PROTOCOL.find_cell("lj104", "ase_mh", 2004)
    path = PROTOCOL.rows_path(tmp_path, "lj104")
    path.parent.mkdir(parents=True)
    path.write_text(
        ",".join(ROWS.ROW_FIELDS) + "\n"
        "lj104,ase_mh,2004,4000000,cap,1,0.00000025,-1.00000000,1.000000,1.000000,"
        + REVISION
    )
    row = ROWS.append_result(
        tmp_path,
        cell,
        [ROWS.Segment(80, -2.0, 0.25, 0.3, "cap")],
        REVISION,
    )
    assert path.read_bytes().endswith(b"\n")
    with path.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    assert recorded == [row]
    assert recorded[0]["calls_reached"] == "80"


def test_cap_constant_is_not_a_cli_default():
    section = g2_snakefile()
    record = RECORD_PATH.read_text()
    dry = DRY_PATH.read_text()
    protocol = PROTOCOL_PATH.read_text()
    assert "reject_cap_override(config)" in section
    assert "g2_protocol.cells()" in section
    assert "rule g2_budget:" in section
    assert "rule g2_budget_cell:" in section
    assert "record_rule.py" in section
    assert "dry_run.py" not in section
    assert "config.get" not in section
    assert "config[" not in section
    assert "43200" not in section
    assert "43_200" not in section
    assert TEST_CAP_ENV not in section
    assert TEST_CAP_ENV not in record
    assert TEST_CAP_ENV not in protocol
    assert TEST_CAP_ENV not in ROWS_PATH.read_text()
    assert TEST_CAP_ENV in dry
    assert "43200" not in record
    assert "43_200" not in record
    assert "os.environ" not in record
    assert "getenv" not in record
    assert "os.environ" not in protocol
    assert PROTOCOL.CPU_CAP_S == 43_200
    PROTOCOL.reject_cap_override({})
    PROTOCOL.reject_cap_override({"sizes": "75"})
    for key in PROTOCOL.LOCKED_CONFIG_KEYS:
        with pytest.raises(ValueError):
            PROTOCOL.reject_cap_override({key: PROTOCOL.CPU_CAP_S})

    rules = re.findall(r"^rule (\w+):", SNAKEFILE.read_text(), re.MULTILINE)
    registry = json.loads((ROOT / "bench" / "h2h.json").read_text())
    for name in ("g2_budget", "g2_budget_cell"):
        assert name in rules
        assert registry["not_h2h"][name]
        assert name not in {entry["rule"] for entry in registry["h2h"]}


def test_record_rule_ignores_the_test_cap(tmp_path):
    env = {**os.environ, TEST_CAP_ENV: "0.25"}
    listed = subprocess.run(
        [sys.executable, str(RECORD_PATH), "--list"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    records = [json.loads(line) for line in listed.stdout.splitlines() if line.strip()]
    assert records == [PROTOCOL.cell_record(cell) for cell in PROTOCOL.cells()]
    assert records[0]["cpu_cap_s"] in {None, 43_200}
    capped = next(
        record for record in records
        if record["system"] == "lj98" and record["arm"] == "ase_mh" and record["seed"] == 2000
    )
    assert capped["cpu_cap_s"] == 43_200
    assert capped["stop_rule"] == "cap"
    assert capped["rows"] == "bench/ase-lj98/rows-ljfix-budget.csv"
    assert capped["shapes"] == "bench/ase-lj98/shapes-ljfix-budget.jsonl"
    assert capped["wall_cap_s"] is None
    assert capped["cores"] == 1

    stamp = tmp_path / "cell.json"
    subprocess.run(
        [
            sys.executable,
            str(RECORD_PATH),
            "--system", "lj38",
            "--arm", "ase_bh",
            "--seed", "2000",
            "--stamp", str(stamp),
        ],
        check=True,
        env=env,
    )
    budget_stamp = json.loads(stamp.read_text())
    assert budget_stamp["stop_rule"] == "budget"
    assert budget_stamp["cpu_cap_s"] is None
    assert budget_stamp["preset"] is None
    assert list(ROOT.glob("bench/ase-*/rows-ljfix-budget.csv")) == []
    assert list(ROOT.glob("bench/ase-*/shapes-ljfix-budget.jsonl")) == []


def test_readme_states_the_protocol_and_the_cap_reason():
    text = README.read_text()
    for phrase in (
        "400000",
        "1000000",
        "4000000",
        "2000-2011",
        "43200",
        "for_cluster",
        "stop=cap",
        "stop=budget",
        "stop=hit",
        "calls_reached",
        "fraction_of_budget",
        "rows-ljfix-budget.csv",
        "shapes-ljfix-budget.jsonl",
        "rows-ljfix*.csv",
        "one core",
        "no wall cap",
        "CPU_CAP_S",
        "one per VM",
        "censored miss",
        "ase_bh",
        "ase_mh",
    ):
        assert phrase in text
    reason = reason_section().lower()
    assert "lower bounds" in reason
    assert "ase mh duplicate checks grow about as budget squared" in reason
    assert "at least 22 h per seed" in reason
    assert "at least 25 h per seed" in reason
    assert "at least 1.9-4.3 h per seed" in reason
    assert "basin hopping fits in hours" in reason
    for hedge in HEDGES:
        assert hedge not in reason


def _dry(tmp_path: Path, system: str, arm: str, seed: str):
    env = {**os.environ, TEST_CAP_ENV: "0.25"}
    completed = subprocess.run(
        [
            sys.executable,
            str(DRY_PATH),
            "--system", system,
            "--arm", arm,
            "--seed", seed,
            "--out", str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return next(csv.DictReader(completed.stdout.splitlines()))


def test_dry_run_scales_only_the_capped_cell(tmp_path):
    capped = _dry(tmp_path / "capped", "lj98", "ase_mh", "2000")
    budget = PROTOCOL.PAPER_BUDGETS["lj98"]
    calls = int(capped["calls_reached"])
    assert capped["stop"] == "cap"
    assert 0 <= calls < budget
    assert Decimal(capped["fraction_of_budget"]) == Decimal(calls) / Decimal(budget)
    assert capped["best_e"]
    assert float(capped["cpu_s"]) > 0
    assert float(capped["wall_s"]) > 0
    assert re.fullmatch(r"[0-9a-f]{40}", capped["code_revision"])
    shape = json.loads(
        (tmp_path / "capped" / "bench" / "ase-lj98" / "shapes-ljfix-budget.jsonl").read_text()
    )
    assert shape["segments"] == 2
    assert shape["cpu_s"] == capped["cpu_s"]

    spent = _dry(tmp_path / "budget", "lj38", "ase_bh", "2000")
    assert spent["stop"] == "budget"
    assert spent["calls_reached"] == "400000"
    assert spent["fraction_of_budget"] == "1"
    assert list(ROOT.glob("bench/ase-*/rows-ljfix-budget.csv")) == []

    refused = subprocess.run(
        [
            sys.executable,
            str(DRY_PATH),
            "--system", "lj98",
            "--arm", "ase_mh",
            "--seed", "2001",
            "--out", str(ROOT),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, TEST_CAP_ENV: "0.25"},
        timeout=30,
    )
    assert refused.returncode != 0
    assert list(ROOT.glob("bench/ase-*/rows-ljfix-budget.csv")) == []
