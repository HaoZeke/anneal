# Copyright (c) anneal developers
"""Airspeed Velocity timing of one paired anneal and openconf seed.

openconf runs first and anneal then gets the energy and gradient calls
openconf spent, as in ``snakemake openconf_h2h_n12``, inside
``sudo unshare --uts`` with hostname bench, on core 0.
"""

from pathlib import Path
import csv
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "openconf_h2h" / "enter_bench.sh"
with (ROOT / "bench" / "openconf-h2h" / "molecules.csv").open(newline="") as handle:
    MOLECULES = tuple(row["name"] for row in csv.DictReader(handle))


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
