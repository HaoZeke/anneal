# Copyright (c) anneal developers
"""Airspeed Velocity timing of one Sutton–Chen c-MCTS seed.

The stored table is the Snakemake output and is not rewritten here.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "cmcts_au"))
sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke  # noqa: E402
from run_campaign import PAPER_BUDGETS  # noqa: E402

ENTER = ROOT / "scripts" / "cmcts_au" / "enter.sh"


class SuttonChenCmcts:
    params = (["au35", "au38", "au55"], ["recommended", "for_cluster", "cmcts"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no system {system}")
        if arm not in ("recommended", "for_cluster", "cmcts"):
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            ["bash", str(ENTER), "--only", system, "2000", arm],
            check=True,
        )


class Smoke:
    """One recommended seed at a small budget. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("cmcts_au", ["search", "recommended", "35", "2000"])
