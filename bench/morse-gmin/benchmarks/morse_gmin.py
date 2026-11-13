# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Morse head-to-head seed.

The comparison is ``bench/morse-gmin/summary.csv``. This benchmark
times the anneal arm of M30 at rho 6, seed 2000, which is the paper
budget of 400000 calls. A GMIN row already stored at that revision
is not launched again.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_gmin" / "enter_bench.sh"


class MorseGmin:
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
                "m30r6",
                "--seed",
                "2000",
                "--arm",
                "anneal",
            ],
            check=True,
        )
