"""Airspeed Velocity timing of one seeded Gupta head-to-head run.

The clock is the same serial arm the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. The call budget is the
one ``h2h_metal budget`` prints for that Cambridge system.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_bench.sh"


class MetalHeadToHead:
    params = (["zn38", "zn55", "zn75", "zn80", "cu19au19"], ["anneal", "ase"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if system not in ("zn38", "zn55", "zn75", "zn80", "cu19au19"):
            raise RuntimeError(f"no Cambridge system {system}")
        if arm not in ("anneal", "ase"):
            raise RuntimeError(f"unknown arm {arm}")
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

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
            ],
            check=True,
        )
