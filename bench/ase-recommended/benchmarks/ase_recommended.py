# Copyright (c) anneal developers
"""Airspeed Velocity timing of the recommended ASE-panel summary.

The clock is scripts/ase_panel/summarize.py on the committed rows.
It does not start a search. Budgets match the paper: 400,000 calls
at LJ38, 1,000,000 at LJ55 and Au55, and 4,000,000 at LJ98, LJ104
and LJ150.
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ase_panel" / "summarize.py"

PAPER_BUDGETS = {
    38: 400_000,
    55: 1_000_000,
    98: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
    "au55": 1_000_000,
}


class AseRecommended:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS[38] != 400_000 or PAPER_BUDGETS["au55"] != 1_000_000:
            raise RuntimeError("recommended budgets drifted")
        if not SCRIPT.is_file():
            raise RuntimeError("missing the panel summary")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT), "--all"], check=True)


class Smoke:
    """One recommended seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/ase_panel")
        if not binary.is_file():
            raise NotImplementedError("ase_panel is not on this machine")
        subprocess.run(
            ["sudo", "-n", "unshare", "--uts", "sh", "-c",
             f"hostname bench; exec taskset -c 0 {binary} lj38 2000 2000 recommended"],
            check=True,
        )
