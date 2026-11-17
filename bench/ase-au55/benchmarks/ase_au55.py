# Copyright (c) anneal developers
"""Airspeed Velocity timing of the Au55 ASE summary.

The clock is scripts/ase_panel/summarize.py on the committed rows.
It does not start a search. The Au55 paper budget is 1_000_000
force evaluations. The potential is Gupta::copper_gold.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ase_panel" / "summarize.py"

PAPER_BUDGETS = {"au55": 1_000_000}


class AseAu55:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS["au55"] != 1_000_000:
            raise RuntimeError("Au55 budget drifted")
        if not SCRIPT.is_file():
            raise RuntimeError("missing the panel summary")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT), "au55", "--check"], check=True)


class Smoke:
    """One anneal seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        binary = Path("/work/target-panel/release/examples/ase_panel")
        if not binary.is_file():
            raise NotImplementedError("ase_panel is not on this machine")
        subprocess.run(
            ["sudo", "-n", "unshare", "--uts", "sh", "-c",
             f"hostname bench; exec taskset -c 0 {binary} au55 2000 2000"],
            check=True,
        )


class AseSmoke:
    """Exercise ASE and the persistent Gupta evaluator transport."""

    timeout = 120

    def time_smoke(self):
        try:
            import ase
        except ImportError as missing:
            raise NotImplementedError("ASE is not installed") from missing
        if ase.__version__ != "3.29.0":
            raise NotImplementedError(f"ASE 3.29.0 required, found {ase.__version__}")
        binary = Path("/work/target-panel/release/examples/asg_eval")
        if not binary.is_file():
            raise NotImplementedError("asg_eval is not on this machine")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        script = ROOT / "scripts" / "ase_panel" / "run_one.py"
        for arm in ("ase_bh", "ase_mh"):
            command = (
                "hostname bench; "
                f"export ASE_PANEL_REVISION={revision} ASE_PANEL_BUDGET=8 "
                f"ASE_PANEL_WALL_CAP=30 ASE_PANEL_WORK=/tmp/ase-panel-asv ASG_EVAL={binary}; "
                f"exec taskset -c 0 {sys.executable} {script} au55 {arm} 2000"
            )
            subprocess.run(["sudo", "-n", "unshare", "--uts", "sh", "-c", command], check=True)
