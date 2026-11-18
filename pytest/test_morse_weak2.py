# Copyright (c) anneal developers
"""The morse-weak2 rejected-default report is current."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "morse_weak2" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("morse_weak2_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def test_append_only_rows_and_rejected_gate_report_are_current():
    grouped = SUMMARY.load_rows()
    assert set(grouped) == set(SUMMARY.SYSTEMS)
    assert SUMMARY.complete_systems(grouped) == (
        "m38r3",
        "m38r6",
        "m38r10",
        "m38r14",
        "m55r3",
        "m55r6",
        "m55r10",
        "m55r14",
    )
    subprocess.run(["python3", str(SCRIPT), "--check"], cwd=ROOT, check=True)


def test_default_gate_records_the_hit_regressions_that_rejected_it():
    with (ROOT / "bench" / "morse-weak2" / "summary.csv").open(newline="") as handle:
        rows = {row["system"]: row for row in csv.DictReader(handle)}
    assert (rows["m38r10"]["base_hits"], rows["m38r10"]["fix_hits"]) == ("12", "8")
    assert (rows["m38r14"]["base_hits"], rows["m38r14"]["fix_hits"]) == ("12", "5")
