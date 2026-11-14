# Copyright (c) anneal developers
"""Airspeed Velocity timing of the paired-gap summary.

The clock is scripts/h2o16_crest/summarize.py on the committed seeds.
It does not start a search.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "h2o16_crest" / "summarize.py"


class H2o16Crest:
    timeout = 120

    def setup(self):
        if not SCRIPT.is_file():
            raise RuntimeError(f"missing {SCRIPT}")

    def time_summary(self):
        subprocess.run(["python3", str(SCRIPT)], check=True)
