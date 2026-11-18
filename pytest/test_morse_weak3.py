# Copyright (c) anneal developers
"""The one-shot return-Twin gate is complete and current."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "morse_weak3" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("morse_weak3_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_rows_are_complete_and_report_is_current():
    grouped = SUMMARY.load_rows()
    assert set(grouped) == set(SUMMARY.SYSTEMS)
    for system in SUMMARY.SYSTEMS:
        assert all(len(grouped[system][arm]) == 12 for arm in SUMMARY.ARMS)
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_gate_records_the_losses_that_keep_adaptive_opt_in():
    with (ROOT / "bench" / "morse-weak3" / "summary.csv").open(newline="") as handle:
        rows = {row["system"]: row for row in csv.DictReader(handle)}
    assert (rows["m38r10"]["base_hits"], rows["m38r10"]["fix_hits"]) == ("10", "8")
    assert (rows["lj38"]["base_hits"], rows["lj38"]["fix_hits"]) == ("12", "8")
    assert float(rows["lj38"]["calls_p"]) < 0.05
    assert float(rows["lj38"]["wall_p"]) < 0.05
