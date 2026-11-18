# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded Morse head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
force-call cap for each Cambridge Morse system.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "m38r14": 400_000,
    "m55r14": 1_000_000,
    "m80r14": 4_000_000,
    "m38r6": 400_000,
    "m55r6": 1_000_000,
    "m80r6": 4_000_000,
}


class MorseHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                system,
                "--seed",
                "2000",
                "--arm",
                arm,
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_morse", system, budget)
        smoke("h2h_morse", ["run", "m38r6", "2000"])
