"""Airspeed Velocity timing of the Morse hop outside the potential.

The comparison is ``bench/morse-hop/summary.csv``. This benchmark times
one M80 rho = 3 seed at 20_000 force calls, on the serial bench host.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_hop" / "enter_bench.sh"


class MorseHop:
    timeout = 86400

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self):
        subprocess.run(
            ["bash", str(ENTER), "--budget", "20000", "--seed", "2000"],
            check=True,
        )
