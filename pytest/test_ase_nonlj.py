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
        assert all(SUMMARY.wall_censored(grouped[system][arm]) for arm in SUMMARY.ASE_ARMS)
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_wall_censored_ase_comparisons_have_no_rank_estimate():
    with (ROOT / "bench" / "ase-nonlj" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["mann_whitney_U_bh"] == ""
        assert row["mann_whitney_p_bh"] == ""
        assert row["mann_whitney_U_mh"] == ""
        assert row["mann_whitney_p_mh"] == ""


def test_exact_and_rank_statistics_match_the_panel_definitions():
    assert SUMMARY.fisher(0, 10) == pytest.approx(6.730380752335834e-05)
    statistic, probability = SUMMARY.mann_whitney(
        [400_000.0] * 12,
        [336_650, 141_347, 164_437, 10_413, 307_562, 203_015,
         400_000, 349_934, 164_167, 19_727, 400_000, 339_051],
    )
    assert statistic == 132
    assert probability == pytest.approx(0.0001253087278839309)


def test_wall_ratios_do_not_pair_independent_seed_labels():
    left = [{"wall_s": "1"}, {"wall_s": "100"}, {"wall_s": "101"}]
    right = [{"wall_s": "1"}, {"wall_s": "2"}, {"wall_s": "100"}]
    assert SUMMARY.wall_ratio(left, right) == (50.0, 0.02)
