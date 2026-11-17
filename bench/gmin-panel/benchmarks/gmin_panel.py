# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seed of the GMIN panel, both arms.

The clock is the serial pair the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``PAPER_BUDGETS`` is the
call cap for each system; the runner refuses a binary whose own budget
differs. Rows from a timing run go to a scratch directory so the
recorded panel stays untouched.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "gmin_panel" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, need_bench_host, smoke

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
# c2de652d lacked the LJ103 entry in h2h_gmin_proof (added in 503470a0);
# enter_bench.sh applies bench/paper-0120/lj103-table.patch only when needed.
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
        need_bench_host()
        self.scratch = tempfile.mkdtemp(prefix="gmin-panel-asv-")
        self.rev = os.environ.get("GMIN_PANEL_REV") or subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def teardown(self, system):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def time_seed(self, system):
        env = dict(os.environ, GMIN_PANEL_OUT=self.scratch, GMIN_PANEL_REV=self.rev)
        subprocess.run(["bash", str(ENTER), "--system", system, "--seed", "2000"], check=True, env=env)


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget(EXAMPLES.get(system, "h2h_gmin_proof"), system, budget)
        subprocess.run([sys.executable, str(ENTER.parent / "run_panel.py"), "--self-test"], check=True)
        smoke("h2h_gmin_proof", ["run", "lj38", "2000"])
        smoke("h2h_metal", ["run", "au55", "2000"])
        smoke("h2h_morse", ["run", "m55r14", "2000"])
