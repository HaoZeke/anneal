"""Airspeed Velocity timing of one seeded TIP4P run.

The clock is the same serial paper preset the Snakemake rule runs,
inside ``sudo unshare --uts`` with hostname bench. The size and the
relaxation budget are the campaign's. This does not change a paper count.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tip4p_rates" / "enter_bench.sh"


class Tip4pWaterRates:
    timeout = 7200

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self):
        subprocess.run(
            ["bash", str(ENTER), "--size", "10", "--seed", "0"],
            check=True,
        )
