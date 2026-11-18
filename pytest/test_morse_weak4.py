# Copyright (c) anneal developers
"""The held-stream gate is complete and loses nowhere."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "morse_weak4" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("morse_weak4_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_rows_are_complete_and_report_is_current():
    grouped = SUMMARY.load_rows()
    assert set(grouped) == set(SUMMARY.SYSTEMS)
    for system in SUMMARY.SYSTEMS:
        assert all(len(grouped[system][arm]) == 12 for arm in SUMMARY.ARMS)
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_gate_loses_nowhere_and_keeps_the_named_gains():
    with (ROOT / "bench" / "morse-weak4" / "summary.csv").open(newline="") as handle:
        rows = {row["system"]: row for row in csv.DictReader(handle)}
    for system, row in rows.items():
        assert int(row["fix_hits"]) >= int(row["base_hits"]), system
        if float(row["calls_p"]) < 0.05:
            assert float(row["fix_median_calls"]) <= float(row["base_median_calls"]), system
        if float(row["wall_p"]) < 0.05:
            assert float(row["fix_median_wall_s"]) <= float(row["base_median_wall_s"]), system
    assert (rows["lj38"]["base_hits"], rows["lj38"]["fix_hits"]) == ("12", "12")
    assert (rows["m38r10"]["base_hits"], rows["m38r10"]["fix_hits"]) == ("10", "10")
    assert (rows["m75r14"]["base_hits"], rows["m75r14"]["fix_hits"]) == ("7", "9")
    assert (rows["m80r14"]["base_hits"], rows["m80r14"]["fix_hits"]) == ("8", "11")
    assert (rows["lj98"]["base_hits"], rows["lj98"]["fix_hits"]) == ("9", "12")
