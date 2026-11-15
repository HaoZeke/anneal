# Copyright (c) anneal developers
"""Airspeed Velocity timing of one GFN2-xTB (H2O)20 seed, anneal against CREST.

The clock is the serial campaign the Snakemake rule ``water_h2o20_goat``
runs, each search inside ``sudo unshare --uts`` with hostname bench and
``taskset -c 0``. ``PAPER_BUDGETS`` is anneal's call cap; CREST and GOAT
run their own schedules, and GOAT only where ``orca`` is on PATH. The row
goes to a scratch CSV, not to ``bench/water-h2o20-goat/results.csv``.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "scripts" / "h2o20_goat_campaign.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_env, smoke

PAPER_BUDGETS = {20: 4000}


class WaterH2o20Goat:
    params: ClassVar[list[int]] = list(PAPER_BUDGETS)
    param_names: ClassVar[list[str]] = ["molecules"]
    timeout = 86400

    def setup(self, molecules):
        if not CAMPAIGN.is_file():
            raise RuntimeError(f"missing {CAMPAIGN}")

    def time_seed(self, molecules):
        with tempfile.TemporaryDirectory(prefix="h2o20-goat-") as scratch:
            env = dict(
                os.environ,
                H2O_MOLECULES=str(molecules),
                H2O_BUDGET=str(PAPER_BUDGETS[molecules]),
                H2O_SEEDS="15001",
                H2O_OUT_CSV=f"{scratch}/results.csv",
                H2O_WORK=f"/work/h2o20-goat-asv/{molecules}",
            )
            subprocess.run(["bash", str(CAMPAIGN)], check=True, env=env)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        self.env = need_env("RGPOT_XTB_ENGINE")

    def time_smoke(self):
        smoke("gfn2_water_h2h", ["20", "200", "15001", "{work}"], budget=None, features=("rgpot-ex",), env=self.env)
