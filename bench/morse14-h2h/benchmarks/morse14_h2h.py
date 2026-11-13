"""Airspeed Velocity timing of one Morse rho = 14 head-to-head seed.

The scientific comparison is ``bench/morse14-h2h/summary.csv``. This
benchmark times the anneal arm the Snakemake rule runs. A GMIN seed
that is already stored at the paper budget is not launched again.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse14_h2h" / "enter_bench.sh"

PAPER_BUDGETS = {
    "m55r14": 1_000_000,
    "m80r14": 4_000_000,
}


class Morse14HeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                system,
                "--seed",
                "2000",
                "--arm",
                arm,
            ],
            check=True,
        )
