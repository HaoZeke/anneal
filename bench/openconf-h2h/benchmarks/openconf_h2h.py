# Copyright (c) anneal developers
"""Airspeed Velocity timing of one seeded conformer-search head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
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
ARMS = ("anneal", "openconf", "crest")


class OpenconfHeadToHead:
    params = (list(MOLECULES), list(ARMS))
    param_names = ["molecule", "arm"]
    timeout = 86400

    def setup(self, molecule, arm):
        if molecule not in MOLECULES:
            raise RuntimeError(f"no molecule {molecule}")
        if arm not in ARMS:
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, molecule, arm):
        subprocess.run(
            ["bash", str(ENTER), "campaign", "--only", molecule, "2000", arm],
            check=True,
            env={**os.environ, "OPENCONF_H2H_SKIP_BUILD": "1"},
        )
