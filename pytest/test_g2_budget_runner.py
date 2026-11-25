# Copyright (c) anneal developers
"""The production runner executes the pre-registered arms."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
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


def panel_binary() -> Path:
    env = os.environ.get("ASG_EVAL")
    if env and Path(env).is_file():
        return Path(env)
    built = ROOT / "target" / "release" / "examples" / "asg_eval"
    if built.is_file():
        return built
    raise AssertionError("panel evaluator is not built")


def panel_dimer_energy(server, distance: float) -> float:
    coords = np.zeros(55 * 3, dtype=np.float64)
    coords[3] = distance
    for index in range(2, 55):
        coords[3 * index] = 1.0e8 * index
    energy, _inside, _grad = server.eval(coords)
    return float(energy)


def panel_gupta_scales(server) -> tuple[float, float]:
    """The CLI dimer grid: r_min / 2^(1/6) and the well depth -E_min."""
    best_r = 0.0
    best_e = math.inf
    distance = 1.8
    while distance <= 4.5:
        energy = panel_dimer_energy(server, distance)
        if energy < best_e:
            best_e = energy
            best_r = distance
        distance += 0.002
    return best_r / 2.0 ** (1.0 / 6.0), -best_e


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_au55_anneal_evaluator_matches_the_panel_scales():
    binary = panel_binary()
    spec = subprocess.check_output([str(binary), "spec", "au55"], text=True).split()
    potential_dir = ROOT / "scripts" / "ase_gofee_h2h"
    if str(potential_dir) not in sys.path:
        sys.path.insert(0, str(potential_dir))
    import potential

    server = potential.AnnealServer(str(binary), "au55")
    try:
        length, well = panel_gupta_scales(server)
        assert abs(length - float(spec[5])) < 1.0e-12
        assert well > 0.0
        coords = np.random.default_rng(2025).normal(size=(55, 3))
        coords -= coords.mean(axis=0)
        coords *= 2.0
        physical = np.ascontiguousarray(coords.reshape(-1) * length)
        raw_energy, _inside, raw_grad = server.eval(physical)
        expected_energy = float(raw_energy) / well
        expected_grad = np.asarray(raw_grad, dtype=np.float64).reshape(-1) * (length / well)
    finally:
        server.close()

    got_energy, got_grad = runner().au55_scaled_energy_gradient(coords)
    got_grad = np.asarray(got_grad, dtype=np.float64).reshape(-1)
    assert got_grad.shape == expected_grad.shape
    assert abs(expected_energy) > 1.0
    assert abs(got_energy - expected_energy) / abs(expected_energy) < 1.0e-10
    assert np.linalg.norm(got_grad - expected_grad) / np.linalg.norm(expected_grad) < 1.0e-10


def test_anneal_cap_resume_matches_an_uncapped_run(tmp_path, monkeypatch):
    budgets = dict(PROTOCOL.PAPER_BUDGETS)
    budgets["lj38"] = TINY_BUDGET
    monkeypatch.setattr(PROTOCOL, "PAPER_BUDGETS", budgets)
    mod = runner()
    uncapped_root = tmp_path / "uncapped"
    uncapped = mod.run_cell("lj38", "anneal", 2000, uncapped_root, uncapped_root / "stamp.json")
    _rows, uncapped_shapes = rows_and_shapes(uncapped_root, "lj38")
    full_hash = uncapped_shapes[0]["trajectory_sha256"]

    calls = {"n": 0}

    def segment_cap(cell):
        calls["n"] += 1
        if calls["n"] == 1:
            return 1.0e-4
        return None

    monkeypatch.setattr(mod, "segment_cap_s", segment_cap)
    capped_root = tmp_path / "capped"
    stamp = capped_root / "stamp.json"
    paused = []
    try:
        assert mod.run_cell("lj38", "anneal", 2000, capped_root, stamp) is None
        assert not stamp.exists()
        assert not PROTOCOL.rows_path(capped_root, "lj38").exists()
        checkpoint = json.loads(mod.checkpoint_path(capped_root, "lj38", "anneal", 2000).read_text())
        pid = int(checkpoint["worker_pid"])
        paused.append(pid)
        assert process_alive(pid)
        assert 0 < checkpoint["calls"] < TINY_BUDGET
        assert checkpoint["trajectory"] != full_hash
        assert checkpoint["segments"] == 1

        resumed = mod.run_cell("lj38", "anneal", 2000, capped_root, stamp)
        assert resumed["calls_reached"] == uncapped["calls_reached"]
        assert resumed["best_e"] == uncapped["best_e"]
        assert resumed["stop"] == uncapped["stop"]
        assert float(resumed["cpu_s"]) > checkpoint["cpu_s"]
        assert float(resumed["wall_s"]) > checkpoint["wall_s"]
        recorded, shapes = rows_and_shapes(capped_root, "lj38")
        assert recorded == [resumed]
        assert shapes[0]["trajectory_sha256"] == full_hash
        assert shapes[0]["segment_count"] == 2
        assert shapes[0]["code_revision"] == PREREG
        assert not process_alive(pid)

        mod.run_cell("lj38", "anneal", 2000, capped_root, capped_root / "rerun.json")
        recorded, shapes = rows_and_shapes(capped_root, "lj38")
        assert recorded == [resumed]
        assert len(shapes) == 1

        calls["n"] = 0
        dead_root = tmp_path / "dead"
        assert mod.run_cell("lj38", "anneal", 2001, dead_root, dead_root / "stamp.json") is None
        dead = json.loads(mod.checkpoint_path(dead_root, "lj38", "anneal", 2001).read_text())
        dead_pid = int(dead["worker_pid"])
        paused.append(dead_pid)
        os.kill(dead_pid, signal.SIGKILL)
        with pytest.raises(RuntimeError, match="gone"):
            mod.run_cell("lj38", "anneal", 2001, dead_root, dead_root / "again.json")
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
