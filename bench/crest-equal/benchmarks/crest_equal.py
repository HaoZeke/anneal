# Copyright (c) anneal developers
"""Airspeed Velocity smoke for the (H2O)6 equal-call CREST panel.

``time_smoke`` rebuilds the summary beside the committed rows and checks
that a second pass does not change the committed file.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "crest_equal" / "run_panel.py"
ROWS = ROOT / "bench" / "crest-equal" / "rows.csv"
SUMMARY = ROOT / "bench" / "crest-equal" / "summary.csv"

PAPER_BUDGETS = {"h2o6": 4000}


class CrestEqual:
    PAPER_BUDGETS = PAPER_BUDGETS
    timeout = 120

    def time_smoke(self):
        before = SUMMARY.read_bytes()
        with tempfile.TemporaryDirectory() as scratch:
            side = Path(scratch) / "summary.csv"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--summarize", "--rows", str(ROWS), "--summary", str(side)],
                check=True,
            )
            subprocess.run(
                [sys.executable, str(SCRIPT), "--summarize", "--rows", str(ROWS), "--summary", str(SUMMARY)],
                check=True,
            )
        if SUMMARY.read_bytes() != before:
            raise AssertionError(f"{SCRIPT} changed {SUMMARY}")
