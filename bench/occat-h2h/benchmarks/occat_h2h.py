# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded OCCat head-to-head run.

The clock is the serial arm in ``sudo unshare --uts`` with hostname
``bench``. ``PAPER_BUDGETS`` is the evaluation cap of every system
scripts/occat_h2h/run_campaign.py runs.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "occat_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "cu38": 400_000,
    "cu55": 1_000_000,
    "cu75": 4_000_000,
    "cu98": 4_000_000,
    "cu104": 4_000_000,
    "pt38": 400_000,
    "pt55": 1_000_000,
    "pt75": 4_000_000,
    "pt98": 4_000_000,
    "pt104": 4_000_000,
    "au55": 1_000_000,
    "cu6au32": 400_000,
    "cu19au19": 400_000,
    "cu27au28": 1_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
}


class OccatHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "occat", "gmin"])
    param_names = ["system", "code"]
    timeout = 86400

    def setup(self, system, code):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, code):
        subprocess.run(["bash", str(ENTER)], check=True)


class Smoke:
    """One seed of each anneal binary at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
        smoke("h2h_metal", ["run", "cu38", "2000"])
