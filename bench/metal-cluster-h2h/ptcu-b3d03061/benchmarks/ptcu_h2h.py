"""Airspeed Velocity timing of one seeded Pt–Cu Gupta run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
map in ``scripts/metal_h2h/run_ptcu.py``. N=38 is the LJ38 paper
budget. N=13 and N=23 are not in that map.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_ptcu.sh"
PAPER_BUDGETS = {
    "pt13": 100_000,
    "cu13": 100_000,
    "pt7cu6": 100_000,
    "pt12cu11": 200_000,
    "pt11cu12": 200_000,
    "pt6cu17": 200_000,
    "pt38": 400_000,
    "pt19cu19": 400_000,
    "cu38": 400_000,
}


class PlatinumCopperHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no Pt–Cu system {system}")
        if arm not in ("anneal", "gmin"):
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--only",
                system,
                "2000",
                arm,
            ],
            check=True,
        )
