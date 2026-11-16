# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded OCCat head-to-head run.

The clock is the serial arm in ``sudo unshare --uts`` with hostname
``bench``. ``PAPER_BUDGETS`` is the evaluation cap.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "occat_h2h" / "enter_bench.sh"

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "au55": 1_000_000,
    "cu6au32": 400_000,
    "cu19au19": 400_000,
    "cu27au28": 1_000_000,
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
