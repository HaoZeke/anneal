# Copyright (c) anneal developers
"""The ASE panel's rows and generated reports are a single checked record."""

from __future__ import annotations

import importlib.util
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ase_panel" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("ase_panel_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
RUNNER_PATH = ROOT / "scripts" / "ase_panel" / "run_one.py"
EVALUATOR_PATH = ROOT / "scripts" / "ase_panel" / "evaluator.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("ase_panel_evaluator", EVALUATOR_PATH)
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)


@pytest.mark.parametrize("system", SUMMARY.SYSTEMS)
def test_rows_are_complete_and_reports_are_generated_from_them(system):
    grouped = SUMMARY.load(system)
    assert set(grouped) == set(SUMMARY.ARMS)
    assert all(tuple(int(row["seed"]) for row in rows) == SUMMARY.SEEDS for rows in grouped.values())
    subprocess.run(["python3", str(SCRIPT), system, "--check"], cwd=ROOT, check=True)


def test_combined_recommended_summary_is_generated_from_the_same_rows():
    subprocess.run(["python3", str(SCRIPT), "--all", "--check"], cwd=ROOT, check=True)


def test_fisher_is_two_sided_exact():
    assert SUMMARY.fisher(0, 10) == pytest.approx(6.730380752335834e-05)
    assert SUMMARY.fisher(1, 12) == pytest.approx(9.614829913579703e-06)


def test_reports_compare_hits_by_fisher_alone():
    summary, report = SUMMARY.render("lj38")
    header = summary.splitlines()[0].split(",")
    assert not [column for column in header if "mann" in column]
    rows = {row.split(",", 1)[0]: dict(zip(header, row.split(","))) for row in summary.splitlines()[1:]}
    assert rows["recommended"]["fisher_p_bh"] == "7.39602e-07"
    assert "| recommended against basin hopping | 12/12 vs 0/12 | 7.39602e-07 |" in report
    assert "Mann-Whitney" not in report
    assert not hasattr(SUMMARY, "mann_whitney")


def test_runner_refuses_implicit_or_mismatched_provenance():
    shell = (ROOT / "scripts" / "ase_panel" / "enter_bench.sh").read_text()
    runner = (ROOT / "scripts" / "ase_panel" / "run_one.py").read_text()
    assert "set ASE_PANEL_REVISION to the exact revision" in shell
    assert "use a new campaign table" in shell
    assert "ASE_VERSION = \"3.29.0\"" in runner
    assert "5a7f629d79be88b9d3f0cb2c6665062bda4ac625" not in shell + runner


def test_au_length_comes_from_the_evaluator(monkeypatch):
    reply = "au55 55 165 1000000 -191.030852800000 2.057976038904\nAu\n"
    monkeypatch.setattr(
        EVALUATOR.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=reply),
    )
    assert EVALUATOR.evaluator_spec("asg_eval", "au55") == (
        55,
        1_000_000,
        -191.0308528,
        2.057976038904,
    )
    assert '"length"' not in RUNNER_PATH.read_text()
