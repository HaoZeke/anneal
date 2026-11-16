# Copyright (c) anneal developers
"""Airspeed Velocity timing of the LJ75 AGOX summary.

The clock is scripts/agox_lj75/summarize.py on the committed rows.
It does not start a search. The Snakemake rule is the same script.
PAPER_BUDGETS for LJ75 is 4_000_000 force evaluations.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "agox_lj75" / "summarize.py"
ENTER = ROOT / "scripts" / "agox_lj75" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke

PAPER_BUDGETS = {75: 4_000_000}


class AgoxLj75:
    timeout = 120

    def setup(self):
        if 75 not in PAPER_BUDGETS:
            raise RuntimeError("no paper budget for LJ75")
        if not SCRIPT.is_file():
            raise RuntimeError(f"missing {SCRIPT}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("h2h_arm3", ["lj75", "default", "2000", "6400"], budget=None)
