# Copyright (c) anneal developers
"""Airspeed Velocity timing of the LJ98 AGOX summary.

The clock is scripts/agox_lj98/summarize.py on the committed rows.
It does not start a search. The Snakemake rule is the same script.
PAPER_BUDGETS for LJ98 is 4_000_000 force evaluations.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "agox_lj98" / "summarize.py"
ENTER = ROOT / "scripts" / "agox_lj98" / "enter_bench.sh"

PAPER_BUDGETS = {98: 4_000_000}


class AgoxLj98:
    timeout = 120

    def setup(self):
        if 98 not in PAPER_BUDGETS:
            raise RuntimeError("no paper budget for LJ98")
        if not SCRIPT.is_file():
            raise RuntimeError(f"missing {SCRIPT}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)
