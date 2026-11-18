# Copyright (c) anneal developers
"""The morse-weak same-session baseline vs adaptive report matches its rows."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "morse_weak" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("morse_weak_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_rows_are_complete_and_report_is_current():
    grouped = SUMMARY.load_rows()
    assert set(grouped) == set(SUMMARY.SYSTEMS)
    for system in SUMMARY.SYSTEMS:
        assert set(grouped[system]) == set(SUMMARY.ARMS)
        assert all(
            tuple(int(row["seed"]) for row in rows) == SUMMARY.SEEDS
            for rows in grouped[system].values()
        )
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_summary_matches_exact_fisher_and_tie_corrected_mann_whitney():
    with (ROOT / "bench" / "morse-weak" / "summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(SUMMARY.SYSTEMS)
    for row in rows:
        assert row["fisher_p"] != ""
        assert row["mann_whitney_U"] != ""
        assert row["mann_whitney_p"] != ""
