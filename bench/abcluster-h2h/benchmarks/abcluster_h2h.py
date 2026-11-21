# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded run against ABCluster 3.4.

The clock is the serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench and core 0. ``PAPER_BUDGETS``
is the call cap for each system, the ``paper_budget`` of
``h2h_gmin_proof`` and ``h2h_metal``, counted in energy-and-gradient calls.
Rows go to a scratch directory, so a timing never adds to
``bench/abcluster-h2h``.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "abcluster_h2h" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import check_budget, smoke

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "au55": 1_000_000,
}
ARMS = ["abcluster", "anneal-recommended", "anneal-cluster"]


class AbclusterHeadToHead:
    params = (list(PAPER_BUDGETS), ARMS)
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = Path(tempfile.mkdtemp(prefix=f"abcluster-asv-{system}-{arm}-"))

    def time_seed(self, system, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--system",
                system,
                "--seed",
                "2000",
                "--arm",
                arm,
                "--budget",
                str(PAPER_BUDGETS[system]),
                "--out",
                str(self.scratch / "out"),
                "--runs",
                str(self.scratch / "runs"),
            ],
            check=True,
        )


class Smoke:
    """One anneal seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def time_smoke(self):
        for system, budget in PAPER_BUDGETS.items():
            check_budget("h2h_abcluster", system, budget)
        smoke("h2h_abcluster", ["run", "lj38", "2000", "recommended"])
