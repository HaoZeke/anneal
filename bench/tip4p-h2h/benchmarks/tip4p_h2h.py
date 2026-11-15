# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded TIP4P head-to-head pair.

The clock is the same serial arm the Snakemake rule ``tip4p_h2h`` runs,
inside ``sudo unshare --uts`` with hostname bench and ``taskset -c 0``. One
call runs anneal and GMIN from the same start at a 2_000_000-evaluation cap,
the paper ceiling for (H2O)20. The Cambridge database lists the reference for
every size below except 25.
"""

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tip4p_h2h" / "enter_h2h.sh"
ENTER_BUDGET = ROOT / "scripts" / "tip4p_h2h" / "enter_budget.sh"
SIZES = [8, 15, 20, 21, 25]
BUDGETS = [2_000_000, 4_000_000, 8_000_000]

sys.path.insert(0, str(ROOT / "bench"))
from h2h import smoke


class Tip4pHeadToHead:
    PAPER_BUDGETS: ClassVar[dict[int, int]] = {n: 2_000_000 for n in SIZES}
    params = SIZES
    param_names = ["size"]
    timeout = 86400

    def setup(self, size):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_pair(self, size):
        subprocess.run(
            ["bash", str(ENTER), "--size", str(size), "--seed", "2000"],
            check=True,
        )


class Tip4p21BudgetHeadToHead:
    """One (H2O)21 pair on develop's default preset at each sweep budget."""

    PAPER_BUDGETS: ClassVar[dict[int, list[int]]] = {21: BUDGETS}
    params = BUDGETS
    param_names = ["budget"]
    timeout = 86400

    def setup(self, budget):
        if not ENTER_BUDGET.is_file():
            raise RuntimeError(f"missing {ENTER_BUDGET}")

    def time_pair(self, budget):
        subprocess.run(
            ["bash", str(ENTER_BUDGET), "--budget", str(budget), "--seed", "2000"],
            check=True,
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        smoke("h2h_seed", ["tip4p", "8", "0", "2000", "2000", "recommended", "-", "{work}/tip4p8"], budget=None)
