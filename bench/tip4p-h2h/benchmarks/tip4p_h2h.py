"""Airspeed Velocity timing of one seeded TIP4P head-to-head pair.

The clock is the same serial arm the Snakemake rule ``tip4p_h2h`` runs,
inside ``sudo unshare --uts`` with hostname bench and ``taskset -c 0``. One
call runs anneal and GMIN from the same start at a 2_000_000-evaluation cap,
the paper ceiling for (H2O)20. The Cambridge database lists the reference for
every size below except 25.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tip4p_h2h" / "enter_h2h.sh"
ENTER_BUDGET = ROOT / "scripts" / "tip4p_h2h" / "enter_budget.sh"
SIZES = [8, 15, 20, 21, 25]
BUDGETS = [2_000_000, 4_000_000, 8_000_000]


class Tip4pHeadToHead:
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
