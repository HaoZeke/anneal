"""Airspeed Velocity timing of one Morse panel seed.

The comparison is ``bench/morse-panel/summary.csv``. This times the
anneal arm on M80 at rho = 6, seed 2000, at the paper budget.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_panel" / "enter_bench.sh"


class MorsePanel:
    timeout = 86400

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                "m80r6",
                "--seed",
                "2000",
                "--arm",
                "anneal",
            ],
            check=True,
        )
