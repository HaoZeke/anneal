# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Morse head-to-head seed.

The comparison is ``bench/morse-gmin/summary.csv``. This benchmark
times the anneal arm of M30 at rho 6, seed 2000, which is the paper
budget of 400000 calls. A GMIN row already stored at that revision
is not launched again.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_gmin" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "m30r6": 400_000,
    "m30r14": 400_000,
    "m38r6": 400_000,
    "m38r14": 400_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r3": 4_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}


class MorseGmin:
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
                "m30r6",
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
        smoke("h2h_morse", ["run", "m30r6", "2000"])
