# Copyright (c) anneal developers
"""Airspeed Velocity timing for the morse-weak2 gate."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "morse_weak2" / "summarize.py"
PAPER_BUDGETS = {
    "m38r3": 400_000, "m38r6": 400_000, "m38r10": 400_000, "m38r14": 400_000,
    "m55r3": 1_000_000, "m55r6": 1_000_000, "m55r10": 1_000_000, "m55r14": 1_000_000,
    "m75r6": 4_000_000, "m75r14": 4_000_000, "m80r6": 4_000_000, "m80r14": 4_000_000,
    "lj38": 400_000, "lj55": 1_000_000, "lj75": 4_000_000, "lj98": 4_000_000,
    "lj102": 4_000_000, "lj103": 4_000_000, "lj104": 4_000_000, "lj150": 4_000_000,
    "au55": 1_000_000,
}


class MorseWeak2:
    timeout = 120

    def setup(self):
        if not SCRIPT.is_file() or PAPER_BUDGETS["au55"] != 1_000_000:
            raise RuntimeError("morse-weak2 inputs drifted")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class Smoke:
    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/morse_weak")
        if not binary.is_file():
            raise NotImplementedError("morse_weak is not on this machine")
        subprocess.run(
            [
                "sudo", "-n", "unshare", "--uts", "sh", "-c",
                f"hostname bench; exec taskset -c 0 {binary} run m80r14 2000 2000 recommended",
            ],
            check=True,
        )
