# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded GrowPAL head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
evaluation cap for each Lennard-Jones size, taken from ``paper_budget``
in the GMIN head-to-head binary.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "growpal_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    38: 400_000,
    75: 4_000_000,
    98: 4_000_000,
    104: 4_000_000,
}


class GrowpalHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "growpal"])
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
        if size not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {size}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, size, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--size",
                str(size),
                "--seed",
                "2000",
                "--arm",
                arm,
                "--budget",
                str(PAPER_BUDGETS[size]),
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for size, budget in PAPER_BUDGETS.items():
            check_budget("h2h_gmin_proof", f"lj{size}", budget)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
