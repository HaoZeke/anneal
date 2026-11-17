# Copyright (c) anneal developers
"""Airspeed Velocity timing of the non-LJ (Morse) ASE panel summary.

The clock is scripts/ase_nonlj/summarize.py on the committed rows.
It does not start a search. Budgets follow the paper size map:
400,000 calls at 38 atoms, 1,000,000 calls at 55 atoms.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ase_nonlj" / "summarize.py"

PAPER_BUDGETS = {
    "m38r6": 400_000,
    "m38r14": 400_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
}


class AseNonLj:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS["m38r6"] != 400_000 or PAPER_BUDGETS["m55r14"] != 1_000_000:
            raise RuntimeError("non-LJ budgets drifted")
        if not SCRIPT.is_file():
            raise RuntimeError("missing the non-LJ summary")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class Smoke:
    """One recommended Morse seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/ase_nonlj")
        if not binary.is_file():
            raise NotImplementedError("ase_nonlj is not on this machine")
        subprocess.run(
            [
                "sudo",
                "-n",
                "unshare",
                "--uts",
                "sh",
                "-c",
                f"hostname bench; exec taskset -c 0 {binary} run m38r6 2000 2000 recommended",
            ],
            check=True,
        )
