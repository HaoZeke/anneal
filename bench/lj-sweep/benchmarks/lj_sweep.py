"""Airspeed Velocity timing of one annealed Lennard-Jones seed.

The clock is the serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. GMIN is not rerun.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "lj_sweep" / "enter_bench.sh"


class LjSweep:
    params = ([38, 150],)
    param_names = ["size"]
    timeout = 86400

    def setup(self, size):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, size):
        subprocess.run(
            ["bash", str(ENTER), "--size", str(size), "--seed", "2000"],
            check=True,
        )
