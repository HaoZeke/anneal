# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Morse rho = 14 head-to-head seed.

The scientific comparison is ``bench/morse14-h2h/summary.csv``. This
benchmark times the anneal arm the Snakemake rule runs. A GMIN seed
that is already stored at the paper budget is not launched again.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse14_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "m55r14": 1_000_000,
    "m80r14": 4_000_000,
}


class Morse14HeadToHead:
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
        smoke("h2h_morse", ["run", "m55r14", "2000"])
