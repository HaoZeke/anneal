"""Airspeed velocity deposit for the population strong scaling runs.

The value is the wall time of one fixed problem at one worker count.
A raw log from the Snakemake cell is reused when it is already present.
"""

import os
import subprocess

SCALE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(SCALE_DIR, "bin", "pop_strong_scale")
METHODS = ["population_annealing", "dmc_population", "successive_halving"]
WORKERS = [1, 2, 4, 8, 16, 32]
REPEATS = [0, 1, 2, 3, 4, 5]


def _wall_seconds(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("wall_s:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError("wall_s missing")


class PopulationStrong:
    params = (METHODS, WORKERS, REPEATS)
    param_names = ["method", "workers", "repeat"]
    unit = "seconds"

    def track_wall_seconds(self, method, workers, repeat):
        path = os.path.join(
            SCALE_DIR,
            "results",
            "raw",
            str(method),
            f"w{workers}",
            f"r{repeat}.txt",
        )
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return _wall_seconds(handle.read())
        completed = subprocess.run(
            [
                BINARY,
                "--method",
                str(method),
                "--workers",
                str(workers),
                "--repeat",
                str(repeat),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return _wall_seconds(completed.stdout)
