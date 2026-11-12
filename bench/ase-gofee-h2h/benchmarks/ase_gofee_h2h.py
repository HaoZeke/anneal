"""Airspeed Velocity timing of one seeded Lennard-Jones arm.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. The force-call budget is the
paper budget. This does not change a paper count.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "ase_gofee_h2h" / "enter_bench.sh"


class AseGofeeHeadToHead:
    params = ([38, 55, 75], ["paper", "ase_bh", "ase_mh", "agox_gofee"])
    param_names = ["size", "arm"]
    timeout = 86400

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
                "2000",
                "--arm",
                arm,
            ],
            check=True,
        )
