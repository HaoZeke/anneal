# Copyright (c) anneal developers
"""Airspeed Velocity timing of the LJ150 ASE summary.

The clock is scripts/ase_panel/summarize.py on the committed rows.
It does not start a search. The LJ150 paper budget is 4_000_000
force evaluations.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ase_panel" / "summarize.py"

PAPER_BUDGETS = {150: 4_000_000}


class AseLj150:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS[150] != 4_000_000:
            raise RuntimeError("LJ150 budget drifted")
        if not SCRIPT.is_file():
            raise RuntimeError("missing the panel summary")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT), "lj150", "--check"], check=True)


class Smoke:
    """One anneal seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/ase_panel")
        if not binary.is_file():
            raise NotImplementedError("ase_panel is not on this machine")
        subprocess.run(
            ["sudo", "-n", "unshare", "--uts", "sh", "-c",
             f"hostname bench; exec taskset -c 0 {binary} lj150 2000 2000"],
            check=True,
        )
