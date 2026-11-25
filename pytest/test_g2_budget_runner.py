# Copyright (c) anneal developers
"""The production runner executes the pre-registered arms."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "scripts" / "g2_budget" / "protocol.py"
RUNNER_PATH = ROOT / "scripts" / "g2_budget" / "run_cell.py"
SNAKEFILE = ROOT / "workflow" / "Snakefile"
PREREG = "1308d8678a8912de95ded423291a25430bb073fc"
TEST_CAP_ENV = "G2_BUDGET_TEST_CAP_S"
TINY_BUDGET = 40


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


def runner():
    module = load("g2_budget_run_cell", RUNNER_PATH)
    assert module.protocol is PROTOCOL
    return module


def head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def machine_id_sha256() -> str:
    text = Path("/etc/machine-id").read_text().strip()
    return hashlib.sha256(text.encode()).hexdigest()


def model_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("cpu model is missing")


def rows_and_shapes(root: Path, system: str):
    rows_path = PROTOCOL.rows_path(root, system)
    shapes_path = PROTOCOL.shapes_path(root, system)
    import csv

    with rows_path.open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    shapes = [json.loads(line) for line in shapes_path.read_text().splitlines() if line]
    return recorded, shapes


def test_runner_sources_do_not_accept_a_cap_override():
    text = RUNNER_PATH.read_text()
    section = SNAKEFILE.read_text().split("# Pre-registered G2 budget-stop campaign", 1)[1]
    assert "run_cell.py" in section
    assert TEST_CAP_ENV not in text
    assert TEST_CAP_ENV not in section
    assert "--cpu-cap" not in text
    assert "--cap" not in text
    assert "hostname" not in text
    assert '== "bench"' not in text


def test_budget_arms_run_the_real_searches(tmp_path, monkeypatch):
    budgets = dict(PROTOCOL.PAPER_BUDGETS)
    budgets["lj38"] = TINY_BUDGET
    monkeypatch.setattr(PROTOCOL, "PAPER_BUDGETS", budgets)
    mod = runner()
    stamp_root = tmp_path / "stamps"
    for arm in PROTOCOL.ARMS:
        stamp = stamp_root / f"{arm}.json"
        row = mod.run_cell("lj38", arm, 2000, tmp_path, stamp)
        assert row["system"] == "lj38"
        assert row["arm"] == arm
        assert row["seed"] == "2000"
        assert row["budget"] == str(TINY_BUDGET)
        assert row["code_revision"] == PREREG
        assert row["stop"] in {"budget", "hit"}
        assert float(row["best_e"]) < 0
        assert float(row["cpu_s"]) > 0
        assert float(row["wall_s"]) > 0
        reached = int(row["calls_reached"])
        if row["stop"] == "budget":
            assert reached == TINY_BUDGET
            assert row["fraction_of_budget"] == "1"
        else:
            assert 0 < reached <= TINY_BUDGET
        assert stamp.is_file()
        plan = json.loads(stamp.read_text())
        assert plan["stop_rule"] == "budget"
        assert plan["cpu_cap_s"] is None

    recorded, shapes = rows_and_shapes(tmp_path, "lj38")
    assert len(recorded) == 3
    assert len(shapes) == 3
    assert [row["arm"] for row in recorded] == list(PROTOCOL.ARMS)
    revision = head()
    pinned = os.sched_getaffinity(0)
    assert len(pinned) == 1
    for shape in shapes:
        assert shape["code_revision"] == PREREG
        assert shape["runner_revision"] == revision
        assert shape["segments"] == 1
        assert shape["nproc"] == os.cpu_count()
        assert shape["model_name"] == model_name()
        assert shape["machine_id_sha256"] == machine_id_sha256()
        assert shape["core_affinity"] == next(iter(pinned))
        assert "host" not in shape
        assert "hostname" not in shape
    blob = (PROTOCOL.shapes_path(tmp_path, "lj38").read_text()
            + PROTOCOL.rows_path(tmp_path, "lj38").read_text())
    assert Path("/etc/machine-id").read_text().strip() not in blob
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["NUMBA_NUM_THREADS"] == "1"

    again = stamp_root / "again.json"
    mod.run_cell("lj38", "ase_bh", 2000, tmp_path, again)
    recorded, shapes = rows_and_shapes(tmp_path, "lj38")
    assert len(recorded) == 3
    assert len(shapes) == 3
    assert not list(tmp_path.glob("bench/ase-*/rows-ljfix.csv"))


def test_capped_resume_sums_time_and_continues_rng(tmp_path, monkeypatch):
    mod = runner()
    calls = {"n": 0}

    def segment_cap(cell):
        calls["n"] += 1
        if calls["n"] == 1:
            return 1.0e-4
        return None

    monkeypatch.setattr(mod, "segment_cap_s", segment_cap)
    stamp = tmp_path / "cell.json"
    assert mod.run_cell("lj98", "ase_mh", 2000, tmp_path, stamp) is None
    assert not stamp.exists()
    assert not PROTOCOL.rows_path(tmp_path, "lj98").exists()
    checkpoint_file = mod.checkpoint_path(tmp_path, "lj98", "ase_mh", 2000)
    checkpoint = json.loads(checkpoint_file.read_text())
    assert checkpoint["segments"] == 1
    assert checkpoint["calls"] > 0
    assert checkpoint["calls"] < PROTOCOL.PAPER_BUDGETS["lj98"]
    kind, keys, pos, has_gauss, cached = checkpoint["rng_state"]
    saved = (kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached)
    fresh = np.random.RandomState(2000).get_state()
    assert list(saved[1]) != list(fresh[1]) or saved[2] != fresh[2]

    class RecordingState(np.random.RandomState):
        draws: list = []

        def standard_normal(self, size=None):
            out = super().standard_normal(size=size)
            type(self).draws.append(np.array(out, copy=True))
            return out

    monkeypatch.setattr(np.random, "RandomState", RecordingState)
    budgets = dict(PROTOCOL.PAPER_BUDGETS)
    budgets["lj98"] = checkpoint["calls"] + 20
    monkeypatch.setattr(PROTOCOL, "PAPER_BUDGETS", budgets)
    RecordingState.draws = []
    row = mod.run_cell("lj98", "ase_mh", 2000, tmp_path, stamp)
    assert RecordingState.draws
    probe = np.random.RandomState(0)
    probe.set_state(saved)
    expected = probe.standard_normal(size=RecordingState.draws[0].shape)
    restarted = np.random.RandomState(2000).standard_normal(size=expected.shape)
    assert np.array_equal(RecordingState.draws[0], expected)
    assert not np.array_equal(RecordingState.draws[0], restarted)

    assert row["stop"] in {"budget", "hit"}
    assert int(row["calls_reached"]) > checkpoint["calls"]
    assert float(row["cpu_s"]) > checkpoint["cpu_s"]
    assert float(row["wall_s"]) > checkpoint["wall_s"]
    assert row["code_revision"] == PREREG
    assert float(row["best_e"]) <= checkpoint["best_e"] + 1.0e-8
    recorded, shapes = rows_and_shapes(tmp_path, "lj98")
    assert recorded == [row]
    assert len(shapes) == 1
    assert shapes[0]["segments"] == 2
    assert shapes[0]["runner_revision"] == head()
    assert shapes[0]["cpu_s"] == row["cpu_s"]
    assert shapes[0]["wall_s"] == row["wall_s"]
    assert stamp.is_file()

    mod.run_cell("lj98", "ase_mh", 2000, tmp_path, tmp_path / "rerun.json")
    recorded, shapes = rows_and_shapes(tmp_path, "lj98")
    assert recorded == [row]
    assert len(shapes) == 1
