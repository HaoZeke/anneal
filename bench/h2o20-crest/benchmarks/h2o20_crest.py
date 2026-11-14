# Copyright (c) anneal developers
"""Airspeed Velocity check of the corrected (H2O)20 summary.

The clock is the rebuild of summary.csv from the committed rows. It does
not start a search. ``PAPER_BUDGETS`` is the charged-call cap all three arms
share.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "h2o20_crest" / "run_onehost.py"
SUMMARY = ROOT / "bench" / "h2o20-crest" / "summary.csv"
PAPER_BUDGETS = {"h2o20": 28000}


class H2o20Crest:
    PAPER_BUDGETS = PAPER_BUDGETS
    ARMS = ("anneal", "anneal-capped-lindh", "crest")


class Smoke:
    """The committed rows must still give the committed summary."""

    timeout = 120

    def time_smoke(self):
        before = SUMMARY.read_bytes()
        subprocess.run([sys.executable, str(SCRIPT), "--summarize-only"], check=True)
        if SUMMARY.read_bytes() != before:
            raise AssertionError(f"{SCRIPT} no longer reproduces {SUMMARY}")
