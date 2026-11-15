# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded Lennard-Jones arm.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. The force-call budget is the
paper budget. This does not change a paper count.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "ase_gofee_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {38: 400_000, 55: 1_000_000, 75: 4_000_000}


class AseGofeeHeadToHead:
    params = ([38, 55, 75], ["paper", "ase_bh", "ase_mh", "agox_gofee"])
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
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
            ],
            check=True,
        )


class Asg2HeadToHead:
    params = (
        ["au55", "cu38", "cu12au26", "sw10", "tc10", "tip4p8"],
        ["ase_bh", "ase_mh", "agox_gofee"],
    )
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--panel",
                "asg2",
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
        for size, budget in PAPER_BUDGETS.items():
            check_budget("h2h_gmin_proof", f"lj{size}", budget)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
