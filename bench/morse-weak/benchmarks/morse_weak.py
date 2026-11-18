# Copyright (c) anneal developers
"""Airspeed Velocity timing of the morse-weak same-session summary."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "morse_weak" / "summarize.py"

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "m38r3": 400_000,
    "m38r6": 400_000,
    "m38r10": 400_000,
    "m38r14": 400_000,
    "m55r3": 1_000_000,
    "m55r6": 1_000_000,
    "m55r10": 1_000_000,
    "m55r14": 1_000_000,
    "m75r6": 4_000_000,
    "m75r14": 4_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}


class MorseWeak:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS["m80r6"] != 4_000_000 or PAPER_BUDGETS["lj38"] != 400_000:
            raise RuntimeError("morse_weak budgets drifted")
        if not SCRIPT.is_file():
            raise RuntimeError("missing morse_weak summary")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class Smoke:
    """One recommended_adaptive seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/morse_weak")
        if not binary.is_file():
            raise NotImplementedError("morse_weak is not on this machine")
        subprocess.run(
            [
                "sudo",
                "-n",
                "unshare",
                "--uts",
                "sh",
                "-c",
                f"hostname bench; exec taskset -c 0 {binary} run m80r6 2000 2000 recommended_adaptive",
            ],
            check=True,
        )
