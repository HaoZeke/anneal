# Copyright (c) anneal developers
"""Airspeed Velocity timing of one LJ38 seed per decision variant.

The scientific comparison is ``bench/decide/summary.csv``. These
benchmarks time the binaries the Snakemake rule runs, one seed each.
"""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[3]
BUILD = Path(os.environ.get("DECIDE_WORK", "/work/decide"))
SPECS = {"baseline": None, "hyperband": "hyperband", "thompson": "thompson", "ducb": "ducb"}


class DecideLJ38:
    timeout = 3600
    params = list(SPECS)
    param_names = ["variant"]

    def setup(self, variant):
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.binary = BUILD / f"bin-{revision}" / "h2h_gmin_proof"
        if not self.binary.is_file():
            raise NotImplementedError(f"build {self.binary} with enter_bench.sh first")

    def time_seed(self, variant):
        env = dict(os.environ)
        env.pop("H2H_DECIDE", None)
        if SPECS[variant] is not None:
            env["H2H_DECIDE"] = SPECS[variant]
        subprocess.run(
            [str(self.binary), "run", "lj38", "2000"],
            check=True,
            capture_output=True,
            env=env,
        )

    def track_calls(self, variant):
        env = dict(os.environ)
        env.pop("H2H_DECIDE", None)
        if SPECS[variant] is not None:
            env["H2H_DECIDE"] = SPECS[variant]
        out = subprocess.run(
            [str(self.binary), "run", "lj38", "2000"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        ).stdout.splitlines()[-1]
        return int(out.split(",")[5])

    track_calls.unit = "calls"
