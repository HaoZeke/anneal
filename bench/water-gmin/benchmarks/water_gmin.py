# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded (H2O)n pair, anneal against GMIN.

The clock is the serial arm the Snakemake rule ``water_gmin`` runs, inside
``sudo unshare --uts`` with hostname bench and ``taskset -c 0``. One call
runs anneal's default preset and GMIN's rigid-body TIP4P basin-hopping from
the same start at a 2_000_000-evaluation cap, stopping at the Cambridge
header. Rows go to scratch files under /work/water-gmin, not to the tree.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tip4p_h2h" / "enter_water_gmin.sh"
SIZES = [8, 12, 16, 20, 21]


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
