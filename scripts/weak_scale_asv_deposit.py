#!/usr/bin/env python3
"""Write every asv track the weak-scaling table cites."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "weak_scale" / "asv" / "tracks.csv"


def load_bench():
    path = ROOT / "asv_bench" / "benchmarks.py"
    spec = importlib.util.spec_from_file_location("weak_scale_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeakScaling()


def main() -> int:
    bench = load_bench()
    families, workers = bench.params
    tracks = [
        ("median_wall_s", bench.track_median_wall_s),
        ("iqr_s", bench.track_iqr_s),
        ("weak_efficiency", bench.track_weak_efficiency),
        ("charged", bench.track_charged),
        ("replicas", bench.track_replicas),
        ("atoms", bench.track_atoms),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["track", "family", "workers", "value"]
        )
        writer.writeheader()
        for family in families:
            for count in workers:
                for name, fn in tracks:
                    writer.writerow(
                        {
                            "track": name,
                            "family": family,
                            "workers": count,
                            "value": f"{fn(family, count):.16g}",
                        }
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
