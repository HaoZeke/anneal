# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seed of the A/B gate panel.

The clock is the serial pair the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the call cap
for each system; the runner refuses a binary whose own budget differs. A row
already stored for that seed and revision is kept and its run skipped.
``PANEL_B_REV`` names arm B.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "bench" / "blj-panel"
ENTER = HERE / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "blj105n38": 400_000,
    "blj105n55": 1_000_000,
    "blj115n38": 400_000,
    "blj115n55": 1_000_000,
    "blj130n38": 400_000,
    "blj130n55": 1_000_000,
    "blj130n98": 4_000_000,
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "m38r6": 400_000,
    "m38r14": 400_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
    "sc126-38": 400_000,
    "sc126-55": 1_000_000,
    "sc126-80": 1_000_000,
    "sc108-38": 400_000,
    "sc108-55": 1_000_000,
    "sc108-80": 1_000_000,
    "sc96-38": 400_000,
    "sc96-55": 1_000_000,
    "sc96-80": 1_000_000,
}


class BljPanel:
    params = (list(PAPER_BUDGETS),)
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        if not os.environ.get("PANEL_B_REV"):
            raise NotImplementedError("PANEL_B_REV does not name arm B")

    def time_seed(self, system):
        subprocess.run(["bash", str(ENTER), "--system", system, "--seed", "2000"], check=True)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row.

    The binary Lennard-Jones driver is not an example of this tree; the
    runner's preflight checks its budgets against both built arms.
    """

    timeout = 3600

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            if system.startswith("lj"):
                check_budget("h2h_gmin_proof", system, budget)
            elif system.startswith("m"):
                check_budget("h2h_morse", system, budget)
            elif system.startswith("sc"):
                check_budget("h2h_sc", system, budget, features=("rgpot-ex",))
        subprocess.run([sys.executable, "-B", str(HERE / "run_panel.py"), "--self-test"], check=True)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
