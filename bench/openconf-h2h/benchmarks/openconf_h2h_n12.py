# Copyright (c) anneal developers
"""Airspeed Velocity timing of one paired anneal and openconf seed.

openconf runs first and anneal then gets the energy and gradient calls
openconf spent, as in ``snakemake openconf_h2h_n12``, inside
``sudo unshare --uts`` with hostname bench, on core 0. The three-arm pair
adds anneal at the base revision, as in ``snakemake openconf_h2h_conf_fix``.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "openconf_h2h" / "enter_bench.sh"
BASELINE = Path("/work/openconf-h2h/baseline")
with (ROOT / "bench" / "openconf-h2h" / "molecules.csv").open(newline="") as handle:
    MOLECULES = tuple(row["name"] for row in csv.DictReader(handle))

sys.path.insert(0, str(ROOT / "bench"))
from h2h import need_modules, smoke_script


class OpenconfPairedSeed:
    params = list(MOLECULES)
    param_names = ["molecule"]
    timeout = 3600

    def setup(self, molecule):
        if molecule not in MOLECULES:
            raise RuntimeError(f"no molecule {molecule}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_pair(self, molecule):
        subprocess.run(
            ["bash", str(ENTER), "paired", "--pair", molecule, "2000"],
            check=True,
            env={**os.environ, "OPENCONF_H2H_SKIP_BUILD": "1"},
        )


class OpenconfThreeArmSeed:
    params = list(MOLECULES)
    param_names = ["molecule"]
    timeout = 3600

    def setup(self, molecule):
        if molecule not in MOLECULES:
            raise RuntimeError(f"no molecule {molecule}")
        if not (BASELINE / "anneal").is_dir():
            raise RuntimeError(f"no base-revision anneal in {BASELINE}")

    def time_three_arms(self, molecule):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "paired",
                "--pair",
                molecule,
                "2000",
                "--baseline",
                str(BASELINE),
            ],
            check=True,
            env={**os.environ, "OPENCONF_H2H_SKIP_BUILD": "1"},
        )


class Smoke:
    """One seed at a small budget, outside the bench namespace. Not a paper row."""

    timeout = 1800

    def setup(self):
        need_modules("anneal", "rdkit")

    def time_smoke(self):
        smoke_script(
            ROOT / "scripts" / "openconf_h2h" / "h2h_job.py",
            ["anneal", "xld", "2000", "--out", "{work}/anneal", "--budget", "200"],
            modules=("anneal", "rdkit"),
        )
