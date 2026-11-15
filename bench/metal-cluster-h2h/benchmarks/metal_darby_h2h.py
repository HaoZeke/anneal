# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded Darby Cu-Au head-to-head run.

The clock is the same serial arm the Snakemake rule ``metal_darby_h2h``
runs, inside ``sudo unshare --uts`` with hostname bench and ``taskset -c
0``. ``PAPER_BUDGETS`` is the call cap ``run_darby.py`` gives each system.
One job leaves ``bench/metal-cluster-h2h/darby.csv`` alone.
"""

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_darby.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke

PAPER_BUDGETS = {
    "au38": 400_000,
    "cu6au32": 400_000,
    "cu19au19": 400_000,
    "au55": 1_000_000,
    "cu27au28": 1_000_000,
}


class MetalDarbyHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin"])
    param_names: ClassVar[list[str]] = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(["bash", str(ENTER), "--only", system, "2000", arm], check=True)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        # run_darby.py sets H2H_BUDGET; the binary's own budgets are metal_h2h's.
        smoke("h2h_metal", ["run", "cu6au32", "2000"])
