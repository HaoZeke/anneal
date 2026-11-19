# Copyright (c) anneal developers
"""One seed of the pair-scaled GMIN panel, and a smoke that does not search.

``PAPER_BUDGETS`` is the call cap in ``scripts/gmin_panel/run_panel.py``.
The coverage test rejects a drift between the two.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gmin_panel" / "enter_losses2.sh"
SCRIPT = ROOT / "scripts" / "gmin_panel" / "run_losses2.py"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget  # noqa: E402

EXAMPLES = {
    "au55": "h2h_metal",
    "m55r6": "h2h_morse",
    "m55r14": "h2h_morse",
    "m80r6": "h2h_morse",
    "m80r14": "h2h_morse",
}

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}


class GminLosses2:
    params = (tuple(PAPER_BUDGETS),)
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"unknown system {system}")

    def time_seed(self, system):
        env = os.environ.copy()
        env["OUTPUT"] = "/work/gmin-losses2"
        subprocess.run(
            ["bash", str(ENTER), "--system", system, "--seed", "2000"],
            cwd=ROOT,
            env=env,
            check=True,
        )


class Smoke:
    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget(EXAMPLES.get(system, "h2h_gmin_proof"), system, budget)
        subprocess.run([sys.executable, str(SCRIPT), "--self-test"], cwd=ROOT, check=True)
