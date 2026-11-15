# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded Lennard-Jones run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. Sizes and the relaxation
budget are the campaign's. This does not change a paper count.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "lj_bonobo_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_bench_host, smoke

PAPER_BUDGETS = {38: 2000, 75: 2000, 76: 2000, 77: 2000}


class LjBonoboHeadToHead:
    params = ([38, 75, 76, 77], ["lean", "bonobo"])
    param_names = ["size", "arm"]
    timeout = 7200

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
                "0",
                "--arm",
                arm,
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        need_bench_host()

    def time_smoke(self):
        smoke("lj_lean_relax_budget", ["38", "20", "0", "-173.928427"], budget=None, bench_host=True)
