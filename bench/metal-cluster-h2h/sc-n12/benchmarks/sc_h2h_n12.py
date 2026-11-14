"""Airspeed Velocity timing of one seeded Sutton–Chen head-to-head run.

The clock is the twelve-seed serial arm. The stored table is the
Snakemake output and is not rewritten here.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_sc_n12.sh"
SYSTEMS = (
    "sc126-80",
    "sc96-80",
    "sc108-80",
    "sc126-38",
    "sc126-55",
    "sc108-38",
    "sc108-55",
    "sc96-38",
    "sc96-55",
)


class SuttonChenTwelveSeed:
    params = (list(SYSTEMS), ["anneal", "gmin"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in SYSTEMS:
            raise RuntimeError(f"no Sutton-Chen system {system}")
        if arm not in ("anneal", "gmin"):
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            ["bash", str(ENTER), "--only", system, "2000", arm],
            check=True,
        )
