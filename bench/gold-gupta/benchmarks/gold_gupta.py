"""Airspeed Velocity timing of one Gupta gold seed.

The clock is the serial arm in ``scripts/metal_h2h/enter_gold.sh``,
inside ``sudo unshare --uts`` with hostname bench, pinned to core 0.
The full twelve-seed table is ``bench/gold-gupta/seeds.csv``. This
benchmark times Au38 seed 2000, the smallest paper budget in that table.
"""

from pathlib import Path
import os
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "metal_h2h" / "enter_gold.sh"


class GoldGupta:
    params = (["au38"], ["anneal"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        env = {key: value for key, value in os.environ.items() if not key.startswith("H2H_")}
        subprocess.run(
            [
                "sudo",
                "-n",
                "unshare",
                "--uts",
                "bash",
                "-c",
                f"hostname bench; exec bash {ENTER} --one {system} {arm} 2000",
            ],
            check=True,
            env=env,
        )
