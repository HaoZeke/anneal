# Copyright (c) anneal developers
"""Airspeed Velocity timing of the GFN2-xTB quench replay on one start set.

The clock is the serial replay the Snakemake rule ``tric_quench`` runs,
inside ``sudo unshare --uts`` with hostname bench and ``taskset -c 0``:
anneal's Cartesian L-BFGS and TRIC against geomeTRIC's TRIC and Cartesian
quenches from the same starts. ``PAPER_BUDGETS`` is the iteration cap of
each quench. Rows and summary go to scratch files, not to
``bench/tric-quench/results.csv``.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "tric_quench" / "enter_bench.sh"
SCRIPT = ROOT / "scripts" / "tric_quench" / "quench_compare.py"

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_library, need_modules, smoke_script

PAPER_BUDGETS = {"water": 3000, "nh3": 3000, "drug": 3000}


class TricQuench:
    params: ClassVar[list[str]] = list(PAPER_BUDGETS)
    param_names: ClassVar[list[str]] = ["system"]
    timeout = 86400

    def setup(self, system):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_replay(self, system):
        with tempfile.TemporaryDirectory(prefix="tric-quench-") as scratch:
            env = dict(
                os.environ,
                TRIC_QUENCH_ROWS=f"{scratch}/quenches.csv",
                TRIC_QUENCH_SUMMARY=f"{scratch}/results.csv",
            )
            subprocess.run(
                ["bash", str(ENTER), "--only", system, "--max-iter", str(PAPER_BUDGETS[system])],
                check=True,
                env=env,
            )


class Smoke:
    """Two water starts and a 50-iteration replay, outside the bench namespace."""

    timeout = 1800

    def setup(self):
        need_modules("anneal", "geometric", "rdkit")
        need_library("XTB_LIBRARY", Path(sys.base_prefix) / "lib" / "libxtb.so.6")

    def time_smoke(self):
        smoke_script(
            SCRIPT,
            ["starts", "--starts", "{work}/starts.npz", "--take", "1", "--drugs", "1"],
            ["run", "--starts", "{work}/starts.npz", "--out", "{work}/quenches.csv", "--only", "water",
             "--max-iter", "50"],
            modules=("anneal", "geometric", "rdkit"),
            libraries={"XTB_LIBRARY": Path(sys.base_prefix) / "lib" / "libxtb.so.6"},
        )
