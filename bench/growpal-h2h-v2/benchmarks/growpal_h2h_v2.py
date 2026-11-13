"""Airspeed Velocity timing of one seeded GrowPAL head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench and core 0. ``PAPER_BUDGETS``
is the call cap for each system, the ``paper_budget`` of
``h2h_gmin_proof`` and ``h2h_morse``. Rows go to a scratch directory, so
a timing never adds to ``bench/growpal-h2h-v2``.
"""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "growpal_h2h_v2" / "enter_bench.sh"

PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "m38r6": 400_000,
}


class GrowpalHeadToHeadV2:
    params = (list(PAPER_BUDGETS), ["anneal", "growpal"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in PAPER_BUDGETS:
            raise RuntimeError(f"no paper budget for {system}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = Path(tempfile.mkdtemp(prefix=f"growpal-asv-{system}-{arm}-"))

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
