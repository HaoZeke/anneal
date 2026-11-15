# Copyright (c) anneal developers
"""Airspeed Velocity timing of one system and code of the covalent panel.

The clock is the serial arm the Snakemake rule ``covalent_h2h`` runs,
inside ``sudo unshare --uts`` with hostname bench and ``taskset -c 0``:
Stillinger-Weber silicon and Tersoff carbon, anneal against GMIN and ASE
basin hopping, every seed of the chosen system and code. ``PAPER_BUDGETS``
is the call cap ``run_campaign.py`` demands of the binary. Rows go to
``bench/covalent-h2h/asv``, which git ignores, not to ``v3``.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "covalent_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, need_env, smoke

PAPER_BUDGETS = {"sw10": 100_000, "sw13": 100_000, "sw20": 400_000, "tc10": 100_000, "tc20": 400_000}


class CovalentHeadToHead:
    params = (list(PAPER_BUDGETS), ["anneal", "gmin", "ase"])
    param_names: ClassVar[list[str]] = ["system", "code"]
    timeout = 86400

    def setup(self, system, code):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, code):
        subprocess.run(
            ["bash", str(ENTER), "--system", system, "--code", code],
            check=True,
            env=dict(os.environ, BENCH_OUT="asv"),
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        self.env = need_env("RGPOT_COVALENT_LIB")

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_covalent", system, budget, features=("rgpot-ex",), env=self.env)
        smoke("h2h_covalent", ["run", "sw10", "2000"], features=("rgpot-ex",), env=self.env)
