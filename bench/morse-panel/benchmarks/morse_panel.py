# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Morse panel seed.

The comparison is ``bench/morse-panel/summary.csv``. This times the
anneal arm on M80 at rho = 6, seed 2000, at the paper budget.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_panel" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m70r3": 1_000_000,
    "m70r6": 1_000_000,
    "m70r10": 1_000_000,
    "m70r14": 1_000_000,
    "m80r3": 4_000_000,
    "m80r6": 4_000_000,
    "m80r10": 4_000_000,
    "m80r14": 4_000_000,
}


class MorsePanel:
    timeout = 86400

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                "m80r6",
                "--seed",
                "2000",
                "--arm",
                "anneal",
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_morse", system, budget)
        smoke("h2h_morse", ["run", "m55r6", "2000"])
