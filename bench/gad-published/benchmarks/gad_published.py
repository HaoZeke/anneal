# Copyright (c) anneal developers
"""Airspeed Velocity timing of one GAD comparison run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. The gad arm reruns the
diffusion search from the seeded start; gad_deposit reads the Zenodo
deposit. All reruns use the LJ38 force-call ceiling, 400000 calls.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gad_published" / "enter_bench.sh"
WORK = Path("/work/gad-published")

SIZES = (31, 38)
ARMS = ("gad", "anneal", "gmin", "gad_deposit")

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke

PAPER_BUDGETS = {31: 400_000, 38: 400_000}


class GadPublished:
    params = (list(SIZES), list(ARMS))
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
        if size not in SIZES or arm not in ARMS:
            raise RuntimeError(f"unexpected case {size} {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        WORK.mkdir(parents=True, exist_ok=True)
        # A saved row is reused by the runner, so each timing gets its own.
        self.runs = tempfile.mkdtemp(prefix="asv-", dir=WORK)

    def teardown(self, size, arm):
        shutil.rmtree(self.runs, ignore_errors=True)

    def time_seed(self, size, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--size",
                str(size),
                "--seed",
                "2000",
                "--arm",
                arm,
                "--runs",
                self.runs,
            ],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("gad_published", ["run", "lj38", "2000"])
