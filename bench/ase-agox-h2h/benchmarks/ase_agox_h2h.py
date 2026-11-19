"""One seeded ASE or AGOX arm, the same serial command the Snakemake rule runs."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "ase_gofee_h2h" / "enter_bench.sh"


class AseAgoxHeadToHead:
    params = (["lj38", "lj55", "lj75", "au55", "cu6au32"], ["paper", "ase_bh", "ase_mh", "agox_bh", "agox_rss", "agox_gofee"])
    param_names = ["system", "arm"]
    timeout = 86400

    def setup(self, system, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, system, arm):
        subprocess.run(
            ["bash", str(ENTER), "--system", system, "--seed", "2000", "--arm", arm],
            check=True,
        )
