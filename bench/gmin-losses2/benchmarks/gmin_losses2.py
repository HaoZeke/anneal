# Copyright (c) anneal developers
"""ASV entry point for one paired seed of the gmin-losses2 gate."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gmin_panel" / "enter_bench.sh"
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}


class GminLosses2:
    params = (tuple(PAPER_BUDGETS),)
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"unknown system {system}")

    def time_seed(self, system):
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        env = os.environ.copy()
        env.update(
            {
                "GMIN_PANEL_MODE": "fix2",
                "GMIN_PANEL_REV": revision,
                "GMIN_PANEL_WORK": "/work/gmin-losses2",
            }
        )
        subprocess.run(
            ["bash", str(ENTER), "--system", system, "--seed", "2000"],
            cwd=ROOT,
            env=env,
            check=True,
        )


class Smoke:
    timeout = 1800

    def time_smoke(self):
        subprocess.run(
            ["python3", str(ROOT / "scripts" / "gmin_panel" / "run_panel.py"), "--self-test"],
            cwd=ROOT,
            check=True,
        )
