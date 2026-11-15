# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Gupta gold seed.

The clock is the serial arm in ``scripts/metal_h2h/enter_gold.sh``,
inside ``sudo unshare --uts`` with hostname bench, pinned to core 0.
The full twelve-seed table is ``bench/gold-gupta/seeds.csv``. This
benchmark times Au38 seed 2000, the smallest paper budget in that table.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_gold.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "au38": 400_000,
    "au55": 1_000_000,
    "au75": 4_000_000,
    "au98": 4_000_000,
}


class GoldGupta:
    params = (["au38"], ["anneal"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
        subprocess.run(
            [
                "sudo",
                "-n",
                "unshare",
                "--uts",
                "bash",
                "-c",
                f"hostname bench; exec bash {ENTER} --one {system} {arm} 2000",
            ],
            check=True,
            env=env,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_metal", system, budget)
        smoke("h2h_metal", ["run", "au38", "2000"])
