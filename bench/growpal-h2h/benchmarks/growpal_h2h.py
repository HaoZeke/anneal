"""Airspeed Velocity timing of one seeded GrowPAL head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
evaluation cap for each Lennard-Jones size, taken from ``paper_budget``
in the GMIN head-to-head binary.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "growpal_h2h" / "enter_bench.sh"

PAPER_BUDGETS = {
    38: 400_000,
    75: 4_000_000,
    98: 4_000_000,
    104: 4_000_000,
}


class GrowpalHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "growpal"])
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
