# Copyright (c) anneal developers
"""Airspeed Velocity timing for one seed of the water-hexamer panel."""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "water_hexamer_192" / "enter_bench.sh"
RUN = ROOT / "scripts" / "water_hexamer_192" / "run_campaign.py"
ARMS = ["anneal", "polish_fair", "crest", "crest_fair"]

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke


class WaterHexamer192:
    PAPER_BUDGETS: ClassVar[dict[str, int]] = {"h2o6": 4000}
    params = ARMS
    param_names = ["arm"]
    timeout = 86400

    def setup(self, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_one_seed(self, arm):
        with tempfile.TemporaryDirectory(prefix=f"w192-{arm}-") as scratch:
            root = Path(scratch)
            subprocess.run(
                [
                    "bash",
                    str(ENTER),
                    "--seed",
                    "0",
                    "--arm",
                    arm,
                    "--rows",
                    str(root / "rows.csv"),
                    "--summary",
                    str(root / "summary.csv"),
                    "--compare",
                    str(root / "compare.csv"),
                    "--conclusion",
                    str(root / "conclusion.csv"),
                    "--progress",
                    str(root / "progress.md"),
                ],
                check=True,
            )


class Smoke:
    """One short run at one seed outside the bench namespace."""

    timeout = 1800

    def time_smoke(self):
        env = {
            "ASE_PYTHON": "python3",
            "WATER_SEED_FIRST": "0",
            "WATER_SEED_COUNT": "1",
            "WATER_FORCE_BUDGET": "12",
            "WATER_START_XYZ": str(ROOT / "bench" / "water-hexamer-192" / "inputs" / "water-v10.xyz"),
            "WATER_ISOMER_DIR": str(ROOT / "bench" / "water-hexamer-192" / "inputs" / "isomers"),
        }
        smoke("water_hexamer", ["gfn2", "recommended"], budget=None, env=env)
