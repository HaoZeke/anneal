"""Airspeed Velocity timing of one Gupta Au55 seed.

The scientific comparison is ``bench/au55-scale/summary.csv``. This
benchmark times the anneal arm the Snakemake rule runs.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "au55_scale" / "enter_bench.sh"


class Au55Scale:
    timeout = 86400

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self):
        subprocess.run(
            ["bash", str(ENTER), "--seed", "2000"],
            check=True,
        )
