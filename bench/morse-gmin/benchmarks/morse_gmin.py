# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Morse head-to-head seed.

The comparison is ``bench/morse-gmin/summary.csv``. This benchmark
times the anneal arm of one stored seed. M30 is the 400000-call
budget. M100 and M147 are the larger sizes, at 50000 calls per atom.
A seed already stored is not launched again.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "morse_gmin" / "enter_bench.sh"


class MorseGmin:
    params = ["m30r6", "m100r6", "m147r6"]
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        if system not in ("m30r6", "m100r6", "m147r6"):
            raise RuntimeError(f"no benchmark seed for {system}")

    def time_seed(self, system):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                system,
                "--seed",
                "2000",
                "--arm",
                "anneal",
            ],
            check=True,
        )
