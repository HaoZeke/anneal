"""Airspeed Velocity timing of one GFN2-xTB Si10 and one C10 anneal seed.

The scientific comparison with CREST is ``bench/covalent-h2h/gfn2``. This
benchmark times the anneal arm the Snakemake rule runs, writing its rows to
a scratch file so the recorded ones stay put.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "covalent_h2h" / "gfn2_enter_bench.sh"


class Gfn2Covalent:
    timeout = 86400
    params: ClassVar[list[str]] = ["si10", "c10"]
    param_names: ClassVar[list[str]] = ["system"]

    def setup(self, system):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")
        self.scratch = tempfile.mkdtemp()

    def time_seed(self, system):
        env = dict(
            os.environ,
            GFN2_CODES="anneal-general",
            GFN2_ROWS=f"{self.scratch}/seeds.csv",
            GFN2_SUMMARY=f"{self.scratch}/summary.csv",
        )
        subprocess.run(
            ["bash", str(ENTER), "--systems", system, "--seeds", "2000"],
            check=True,
            env=env,
        )
