# Copyright (c) anneal developers
"""The Morse ASE panel reports only statistics supported by its rows."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ase_nonlj" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("ase_nonlj_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_rows_are_complete_and_generated_reports_are_current():
    grouped = SUMMARY.load_rows()
    assert set(grouped) == set(SUMMARY.SYSTEMS)
    for system in SUMMARY.SYSTEMS:
        assert set(grouped[system]) == set(SUMMARY.ARMS)
        assert all(
            tuple(int(row["seed"]) for row in rows) == SUMMARY.SEEDS
            for rows in grouped[system].values()
        )
        assert all(row["stop"] == "wall" for arm in SUMMARY.ASE_ARMS for row in grouped[system][arm])
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_summary_and_report_carry_no_rank_test():
    text = (ROOT / "bench" / "ase-nonlj" / "summary.csv").read_text()
    assert "mann_whitney" not in text.splitlines()[0]
    assert "Mann-Whitney" not in (ROOT / "bench" / "ase-nonlj" / "README.md").read_text()
    assert not hasattr(SUMMARY, "mann_whitney")


def test_fisher_matches_the_panel_definition():
    assert SUMMARY.fisher(0, 10) == pytest.approx(6.730380752335834e-05)


def test_wall_ratios_do_not_pair_independent_seed_labels():
    left = [{"wall_s": "1"}, {"wall_s": "100"}, {"wall_s": "101"}]
    right = [{"wall_s": "1"}, {"wall_s": "2"}, {"wall_s": "100"}]
    assert SUMMARY.wall_ratio(left, right) == (50.0, 0.02)


def _runner():
    path = ROOT / "scripts" / "ase_nonlj" / "run_one.py"
    spec = importlib.util.spec_from_file_location("ase_nonlj_run_one", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_pins_the_ase_version_that_the_summary_reports(monkeypatch):
    ase = pytest.importorskip("ase")
    pytest.importorskip("numba")
    runner = _runner()
    assert runner.ASE_VERSION == SUMMARY.ASE_VERSION
    monkeypatch.setattr(ase, "__version__", runner.ASE_VERSION)
    runner.require_ase_version()
    monkeypatch.setattr(ase, "__version__", "3.27.0")
    with pytest.raises(RuntimeError, match="ASE 3.29.0 required, found 3.27.0"):
        runner.require_ase_version()


def test_the_ase_extra_admits_the_version_the_panels_ran():
    text = (ROOT / "pyproject.toml").read_text()
    assert 'ase = ["ase>=3.23,<3.30"]' in text
    assert 'ase = ">=3.23,<3.30"' in (ROOT / "pixi.toml").read_text()
    assert "ase-3.29.0" in (ROOT / "pixi.lock").read_text()
