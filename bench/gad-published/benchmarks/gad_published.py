"""Airspeed Velocity timing of one deposited-GAD comparison run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. The GAD arm reads the
Zenodo deposit. It does not rerun the diffusion search. Anneal and
GMIN use the LJ38 force-call ceiling, 400000 calls.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gad_published" / "enter_bench.sh"

SIZES = (31, 38)
ARMS = ("gad", "anneal", "gmin")


class GadPublished:
    params = (list(SIZES), list(ARMS))
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
        if size not in SIZES or arm not in ARMS:
            raise RuntimeError(f"unexpected case {size} {arm}")
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
