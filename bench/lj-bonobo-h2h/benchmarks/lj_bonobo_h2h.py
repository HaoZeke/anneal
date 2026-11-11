"""Airspeed Velocity timing of one seeded Lennard-Jones run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. Sizes and the relaxation
budget are the campaign's. This does not change a paper count.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "lj_bonobo_h2h" / "enter_bench.sh"


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
