# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seed of the binary Lennard-Jones panel.

The clock is the serial pair the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
call cap for each system; the runner refuses a binary whose own budget
differs. A row already stored for that seed and revision is kept and its
run skipped.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "blj_gmin" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "blj105n38": 400_000,
    "blj105n55": 1_000_000,
    "blj115n38": 400_000,
    "blj115n55": 1_000_000,
    "blj130n38": 400_000,
    "blj130n55": 1_000_000,
    "blj130n98": 4_000_000,
}


class BljGmin:
    params = (list(PAPER_BUDGETS),)
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system):
        subprocess.run(["bash", str(ENTER), "--system", system, "--seed", "2000"], check=True)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_blj", system, budget)
        subprocess.run([sys.executable, str(ENTER.parent / "run_panel.py"), "--self-test"], check=True)
        smoke("h2h_blj", ["run", "blj105n38", "2000"])
