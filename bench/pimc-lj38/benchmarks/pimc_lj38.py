"""Airspeed Velocity timing of one seeded RPQA head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
LJ38 evaluation cap from the GMIN head-to-head binary.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "pimc_lj38" / "enter_bench.sh"

PAPER_BUDGETS = {
    38: 400_000,
}


class PimcLj38:
    params = (list(PAPER_BUDGETS), ["anneal", "rpqa"])
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
        if size not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {size}")
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
                "--budget",
                str(PAPER_BUDGETS[size]),
            ],
            check=True,
        )


class PimcLj38Paper:
    """One seed of the published-length schedule at 38, 55, or 75 atoms."""

    params = [38, 55, 75]
    param_names = ["size"]
    timeout = 86400

    def setup(self, size):
        if size not in (38, 55, 75):
            raise RuntimeError(f"no schedule for {size}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, size):
        subprocess.run(
            ["bash", str(ENTER), "paper", "--size", str(size), "--seed", "2000"],
            check=True,
        )
