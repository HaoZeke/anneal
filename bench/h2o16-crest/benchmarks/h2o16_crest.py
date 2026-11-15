# Copyright (c) anneal developers
"""Airspeed Velocity timing of the (H2O)16 CREST comparison.

``H2o16Crest`` times scripts/h2o16_crest/summarize.py on the committed
seeds and does not start a search. ``H2o16CrestCapped`` times one seed of
each arm of the capped panel (scripts/h2o16_crest/enter_capped.sh, same
objective, same start, 280305 calls), writing its rows to a scratch
directory so the recorded ones stay put.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "h2o16_crest" / "summarize.py"
ENTER = ROOT / "scripts" / "h2o16_crest" / "enter_capped.sh"


class H2o16Crest:
    timeout = 120

    def setup(self):
        if not SCRIPT.is_file():
            raise RuntimeError(f"missing {SCRIPT}")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class H2o16CrestCapped:
    timeout = 43200
    params: ClassVar[list[str]] = ["anneal", "crest"]
    param_names: ClassVar[list[str]] = ["arm"]

    def setup(self, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = tempfile.mkdtemp()

    def teardown(self, arm):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def time_seed(self, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--seed",
                "2000",
                "--arm",
                arm,
                "--rows",
                f"{self.scratch}/rows.csv",
                "--summary",
                f"{self.scratch}/summary.csv",
                "--out",
                f"{self.scratch}/runs",
            ],
            check=True,
            env=dict(os.environ),
        )
