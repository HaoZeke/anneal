# Copyright (c) anneal developers
"""Airspeed Velocity timing of one GFN2-xTB anneal seed per cluster.

The scientific comparison with CREST is ``bench/covalent-h2h/gfn2``. This
benchmark times the anneal arm the Snakemake rule runs, writing its rows to
a scratch file so the recorded ones stay put.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "covalent_h2h" / "gfn2_enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_env, smoke

PAPER_BUDGETS = {"si10": 100_000, "c10": 100_000, "si13": 100_000, "c20": 100_000, "si20": 100_000}


class Gfn2Covalent:
    timeout = 86400
    params: ClassVar[list[str]] = ["si10", "c10", "si13", "c20"]
    param_names: ClassVar[list[str]] = ["system"]

    def setup(self, system):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = tempfile.mkdtemp()

    def time_seed(self, system):
        env = dict(
            os.environ,
            GFN2_CODES="anneal-general",
            GFN2_ROWS=f"{self.scratch}/seeds.csv",
            GFN2_SUMMARY=f"{self.scratch}/summary.csv",
        )
        subprocess.run(
            ["bash", str(ENTER), "--systems", system, "--seeds", "2000"],
            check=True,
            env=env,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        self.env = need_env("RGPOT_XTB_ENGINE")

    def time_smoke(self):
        smoke("gfn2_covalent_h2h", ["si10", "2000", "2000"], budget=None, features=("rgpot-ex",), env=self.env)
