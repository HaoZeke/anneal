# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded (H2O)n pair, anneal against GMIN.

The clock is the serial arm the Snakemake rule ``water_gmin`` runs, inside
``sudo unshare --uts`` with hostname bench and ``taskset -c 0``. One call
runs anneal's default preset and GMIN's rigid-body TIP4P basin-hopping from
the same start at a 2_000_000-evaluation cap, stopping at the Cambridge
header. Rows go to scratch files under /work/water-gmin, not to the tree.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tip4p_h2h" / "enter_water_gmin.sh"
SIZES = [8, 12, 16, 20, 21]

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke

PAPER_BUDGETS = {n: 2_000_000 for n in SIZES}


class WaterGminHeadToHead:
    params = SIZES
    param_names = ["size"]
    timeout = 86400

    def setup(self, size):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_pair(self, size):
        subprocess.run(
            ["bash", str(ENTER), "--size", str(size), "--seed", "2000"],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("h2h_seed", ["tip4p", "8", "0", "2000", "2000", "recommended", "-", "{work}/tip4p8"], budget=None)
