"""Airspeed Velocity timing of one paper-budget Lennard-Jones seed.

The clock is the same serial binary the Snakemake rule runs, inside
``sudo unshare --uts`` with hostname bench. ``posterior`` is the paper
preset. ``uniform`` is that preset with the depth posterior left unused.
This times a single seed; the interleaved campaign is the Snakemake rule.
"""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[3]
ENTER = ROOT / "scripts" / "depth_posterior_ablation" / "enter_bench.sh"


class DepthPosteriorAblation:
    params = ([38, 55, 75, 98, 104], ["posterior", "uniform"])
    param_names = ["size", "arm"]
    timeout = 86400

    def setup(self, size, arm):
        if not ENTER.is_file():
            raise RuntimeError(f"missing {ENTER}")

    def time_seed(self, size, arm):
        subprocess.run(
            [
                "bash",
                str(ENTER),
                "--size",
                str(size),
                "--seed",
                "2000",
                "--arm",
                arm,
            ],
            check=True,
        )
