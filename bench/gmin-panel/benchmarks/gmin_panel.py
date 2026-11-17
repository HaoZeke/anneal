# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seed of the GMIN panel, both arms.

The clock is the serial pair the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
call cap for each system; the runner refuses a binary whose own budget
differs. A row already stored for that seed and revision is kept and its
run skipped.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gmin_panel" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

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
# The develop h2h_gmin_proof has no LJ103 entry; the runner builds it
# with bench/paper-0120/lj103-table.patch.
EXAMPLES = {
    "au55": "h2h_metal",
    "m55r6": "h2h_morse",
    "m55r14": "h2h_morse",
    "m80r6": "h2h_morse",
    "m80r14": "h2h_morse",
}


class GminPanel:
    params = (list(PAPER_BUDGETS),)
    param_names = ["system"]
    timeout = 86400

    def setup(self, system):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system):
        subprocess.run(["bash", str(ENTER), "--system", system, "--seed", "2000"], check=True)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            if system != "lj103":
                check_budget(EXAMPLES.get(system, "h2h_gmin_proof"), system, budget)
        subprocess.run([sys.executable, str(ENTER.parent / "run_panel.py"), "--self-test"], check=True)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
        smoke("h2h_metal", ["run", "au55", "2000"])
        smoke("h2h_morse", ["run", "m55r14", "2000"])
