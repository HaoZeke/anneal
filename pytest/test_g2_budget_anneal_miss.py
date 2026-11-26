# Copyright (c) anneal developers
"""Anneal budget misses are recorded rows."""

from __future__ import annotations

import csv
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "g2_budget" / "protocol.py"
RUNNER_PATH = ROOT / "scripts" / "g2_budget" / "run_cell.py"
ROWS_PATH = ROOT / "scripts" / "g2_budget" / "rows.py"
TINY_BUDGET = 40
NEAR_BUDGET = 1000
NEAR_SHORT = 29
DEAD_PID = 1 << 30


def load(name: str, path: Path):
    import importlib.util

    if not path.is_file():
        raise FileNotFoundError(path)
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


def runner():
    module = load("g2_budget_run_cell", RUNNER_PATH)
    assert module.protocol is PROTOCOL
    assert module.rows is ROWS
    return module


def records(root: Path, system: str):
    rows_path = PROTOCOL.rows_path(root, system)
    shapes_path = PROTOCOL.shapes_path(root, system)
    with rows_path.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    shapes = [json.loads(line) for line in shapes_path.read_text().splitlines() if line]
    return recorded, shapes


def _budget(monkeypatch, system: str, budget: int) -> None:
    budgets = dict(PROTOCOL.PAPER_BUDGETS)
    budgets[system] = budget
    monkeypatch.setattr(PROTOCOL, "PAPER_BUDGETS", budgets)


def _kill_status(work: Path) -> None:
    path = work / "status.json"
    if not path.is_file():
        return
    try:
        status = json.loads(path.read_text())
    except json.JSONDecodeError:
        return
    pid = status.get("pid")
    if isinstance(pid, int) and pid > 0:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _replaced_search(monkeypatch, root: Path, budget: int, calls: int, best_energy: float):
    """Run lj38 anneal in-process with a search that returns after ``calls``."""
    import anneal

    _budget(monkeypatch, "lj38", budget)
    mod = runner()
    seen: list[float] = []

    def fake(obj, grad, n, search_budget, seed=0, recommended=True, target=None, **kwargs):
        point = np.zeros(3 * n)
        for i in range(calls):
            point = np.zeros(3 * n)
            point[0] = float(i + 1)
            seen.append(float(obj(point)))
            grad(point)
        return {"best_energy": best_energy, "best": point}

    def spawn(work: Path) -> int:
        code = mod._worker_main(str(work))
        if code != 0:
            status = json.loads((work / "status.json").read_text())
            raise RuntimeError(status.get("message") or f"worker exited {code}")
        return DEAD_PID

    monkeypatch.setattr(anneal, "cluster_search", fake)
    monkeypatch.setattr(mod, "_spawn_worker", spawn)
    stamp = root / "stamp.json"
    code = mod.main(
        [
            "--system",
            "lj38",
            "--arm",
            "anneal",
            "--seed",
            "2002",
            "--stamp",
            str(stamp),
            "--root",
            str(root),
        ]
    )
    return code, seen


def test_tiny_budget_anneal_miss_exits_zero(tmp_path, monkeypatch):
    _budget(monkeypatch, "lj38", TINY_BUDGET)
    mod = runner()
    root = tmp_path / "tiny"
    code = mod.main(
        [
            "--system",
            "lj38",
            "--arm",
            "anneal",
            "--seed",
            "2002",
            "--stamp",
            str(root / "stamp.json"),
            "--root",
            str(root),
        ]
    )
    assert code == 0
    recorded, shapes = records(root, "lj38")
    assert len(recorded) == 1
    assert len(shapes) == 1
    row = recorded[0]
    assert list(row) == list(ROWS.ROW_FIELDS)
    assert row["stop"] == "budget"
    assert row["arm"] == "anneal"
    assert row["calls_reached"] == str(TINY_BUDGET)
    assert row["budget"] == str(TINY_BUDGET)
    assert float(row["best_e"]) > -173.0
    assert float(row["cpu_s"]) >= 0.0
    assert float(row["wall_s"]) >= float(row["cpu_s"])
    assert shapes[0]["stop"] == "budget"
    assert shapes[0]["calls_reached"] == row["calls_reached"]
    assert "amendment" not in shapes[0]


def test_near_budget_anneal_miss_keeps_real_calls(tmp_path, monkeypatch):
    calls = NEAR_BUDGET - NEAR_SHORT
    root = tmp_path / "near"
    code, seen = _replaced_search(monkeypatch, root, NEAR_BUDGET, calls, 1.0e6)
    assert code == 0
    assert seen
    recorded, shapes = records(root, "lj38")
    assert len(recorded) == 1
    assert len(shapes) == 1
    row = recorded[0]
    assert row["stop"] == "budget"
    assert row["calls_reached"] == str(calls)
    assert row["calls_reached"] != row["budget"]
    assert row["best_e"] == format(min(seen), ".8f")
    assert float(row["cpu_s"]) >= 0.0
    assert float(row["wall_s"]) >= 0.0
    assert shapes[0]["stop"] == "budget"
    assert shapes[0]["best_e"] == row["best_e"]
    assert shapes[0]["cpu_s"] == row["cpu_s"]
    assert shapes[0]["wall_s"] == row["wall_s"]


def test_anneal_hit_is_not_relabelled_as_a_miss(tmp_path, monkeypatch):
    root = tmp_path / "hit"
    code, _seen = _replaced_search(monkeypatch, root, NEAR_BUDGET, 1, -1000.0)
    assert code == 0
    recorded, shapes = records(root, "lj38")
    assert len(recorded) == 1
    assert len(shapes) == 1
    assert recorded[0]["stop"] == "hit"
    assert recorded[0]["calls_reached"] == "1"
    assert float(recorded[0]["best_e"]) <= -173.927427


def test_anneal_far_from_budget_still_raises(tmp_path, monkeypatch):
    root = tmp_path / "far"
    with pytest.raises(RuntimeError, match="anneal stopped after"):
        _replaced_search(monkeypatch, root, 500, 1, 1.0e6)
    assert not PROTOCOL.rows_path(root, "lj38").exists()


def test_anneal_cap_stop_writes_censored_row(tmp_path, monkeypatch):
    monkeypatch.setattr(PROTOCOL, "CPU_CAP_S", 0)
    mod = runner()
    root = tmp_path / "cap"
    work = mod.checkpoint_path(root, "lj98", "anneal", 2002).parent
    try:
        code = mod.main(
            [
                "--system",
                "lj98",
                "--arm",
                "anneal",
                "--seed",
                "2002",
                "--stamp",
                str(root / "stamp.json"),
                "--root",
                str(root),
            ]
        )
    finally:
        _kill_status(work)
    assert code == 0
    recorded, shapes = records(root, "lj98")
    assert len(recorded) == 1
    assert len(shapes) == 1
    row = recorded[0]
    assert row["stop"] == "cap"
    assert row["arm"] == "anneal"
    assert row["system"] == "lj98"
    assert 0 < int(row["calls_reached"]) < int(row["budget"])
    assert row["budget"] == "4000000"
    assert float(row["cpu_s"]) >= 0.0
    assert float(row["wall_s"]) >= float(row["cpu_s"])
    assert shapes[0]["stop"] == "cap"
    assert "amendment" not in shapes[0]
