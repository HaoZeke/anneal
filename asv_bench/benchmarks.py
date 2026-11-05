"""Airspeed-velocity tracks for the weak-scaling table.

Each track reads one number the table cites. The snakemake rule writes
the summary first, and these tracks deposit that number again.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "weak_scale" / "summary.csv"


def _row(family: str, workers: int) -> dict[str, str]:
    with SUMMARY.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if record["family"] == family and int(record["workers"]) == int(workers):
                return record
    raise KeyError((family, workers))


class WeakScaling:
    """Cited weak-scaling statistics."""

    params = (["pt", "population", "dmc"], [1, 2, 4, 8, 16])
    param_names = ["family", "workers"]

    def track_median_wall_s(self, family, workers):
        return float(_row(family, workers)["median_wall_s"])

    def track_iqr_s(self, family, workers):
        return float(_row(family, workers)["iqr_s"])

    def track_weak_efficiency(self, family, workers):
        return float(_row(family, workers)["weak_efficiency"])

    def track_charged(self, family, workers):
        return float(_row(family, workers)["charged"])

    def track_replicas(self, family, workers):
        return float(_row(family, workers)["replicas"])

    def track_atoms(self, family, workers):
        return float(_row(family, workers)["atoms"])
