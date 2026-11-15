# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded Pt–Cu Gupta run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
map in ``scripts/metal_h2h/run_ptcu.py``. N=38 is the LJ38 paper
budget. N=13 and N=23 are not in that map.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_ptcu.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "pt13": 100_000,
    "cu13": 100_000,
    "pt7cu6": 100_000,
    "pt12cu11": 200_000,
    "pt11cu12": 200_000,
    "pt6cu17": 200_000,
    "pt38": 400_000,
    "pt19cu19": 400_000,
    "cu38": 400_000,
}


class PlatinumCopperHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no Pt–Cu system {system}")
        if arm not in ("anneal", "gmin"):
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--only",
                system,
                "2000",
                arm,
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_metal", system, budget)
        smoke("h2h_metal", ["run", "pt13", "2000"])
