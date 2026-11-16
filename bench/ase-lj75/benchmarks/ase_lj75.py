# Copyright (c) anneal developers
"""Airspeed Velocity timing of the LJ75 ASE summary.

The clock is scripts/ase_lj75/summarize.py on the committed rows.
It does not start a search. The Snakemake rule is the same script.
The LJ75 paper budget is 4_000_000 force evaluations.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "ase_lj75" / "summarize.py"
RUN = ROOT / "scripts" / "ase_lj75" / "run_one.py"
ENTER = ROOT / "scripts" / "ase_lj75" / "enter_bench.sh"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_bench_host, need_modules

PAPER_BUDGETS = {75: 4_000_000}


class AseLj75:
    timeout = 120

    def setup(self):
        if PAPER_BUDGETS[75] != 4_000_000:
            raise RuntimeError("LJ75 budget drifted")
        if not SCRIPT.is_file() or not ENTER.is_file():
            raise RuntimeError("missing the LJ75 ASE scripts")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)


class Smoke:
    """One basin-hopping seed at 2,000 calls. Not a paper row."""

    timeout = 600

    def time_smoke(self):
        need_modules("ase", "numba", "numpy")
        need_bench_host()
        subprocess.run(
            [
                "sudo",
                "-n",
                "unshare",
                "--uts",
                "sh",
                "-c",
                f"hostname bench; exec taskset -c 0 {sys.executable} {RUN} --smoke",
            ],
            check=True,
        )
