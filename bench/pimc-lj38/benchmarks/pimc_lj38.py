# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded RPQA head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
LJ38 evaluation cap from the GMIN head-to-head binary.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "pimc_lj38" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    38: 400_000,
}


class PimcLj38:
    params = (list(PAPER_BUDGETS), ["anneal", "rpqa"])
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


class PimcLj38Paper:
    """One seed of the published-length schedule at 38, 55, or 75 atoms."""

    params = [38, 55, 75]
    param_names = ["size"]
    timeout = 86400

    def setup(self, size):
        if size not in (38, 55, 75):
            raise RuntimeError(f"no schedule for {size}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, size):
        subprocess.run(
            ["bash", str(ENTER), "paper", "--size", str(size), "--seed", "2000"],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for size, budget in PAPER_BUDGETS.items():
            check_budget("h2h_gmin_proof", f"lj{size}", budget)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
