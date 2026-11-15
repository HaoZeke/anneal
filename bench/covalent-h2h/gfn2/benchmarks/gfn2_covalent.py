# Copyright (c) anneal developers
"""Airspeed Velocity timing of one GFN2-xTB anneal seed and one CREST seed.

The scientific comparison with CREST is ``bench/covalent-h2h/gfn2``.
``Gfn2Covalent`` times the anneal arm the Snakemake rule runs and
``Gfn2Si10Crest`` one of the Si10 CREST seeds that bring that arm to
twelve, both writing their rows to a scratch file so the recorded ones
stay put.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "covalent_h2h" / "gfn2_enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_env, smoke

PAPER_BUDGETS = {"si10": 100_000, "c10": 100_000}


def run_enter(scratch, codes, arguments):
    env = dict(
        os.environ,
        GFN2_CODES=codes,
        GFN2_ROWS=f"{scratch}/rows.csv",
        GFN2_SUMMARY=f"{scratch}/summaries.csv",
    )
    subprocess.run(["bash", str(ENTER), *arguments], check=True, env=env)


class Gfn2Covalent:
    timeout = 86400
    params: ClassVar[list[str]] = ["si10", "c10"]
    param_names: ClassVar[list[str]] = ["system"]

    def setup(self, system):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = tempfile.mkdtemp()

    def teardown(self, system):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def time_seed(self, system):
        run_enter(
            self.scratch, "anneal-general", ["--systems", system, "--seeds", "2000"]
        )


class Gfn2Si10Crest:
    timeout = 14400

    def setup(self):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = tempfile.mkdtemp()

    def teardown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def time_seed(self):
        run_enter(self.scratch, "crest", ["--systems", "si10", "--seeds", "2004"])


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        self.env = need_env("RGPOT_XTB_ENGINE")

    def time_smoke(self):
        smoke("gfn2_covalent_h2h", ["si10", "2000", "2000"], budget=None, features=("rgpot-ex",), env=self.env)
