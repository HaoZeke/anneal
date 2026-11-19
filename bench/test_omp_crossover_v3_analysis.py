#!/usr/bin/env python3
"""Corrected crossover analysis ignores equivalent stock-team sizes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "bench" / "run_omp_crossover_v3.py"


def load_harness():
    spec = importlib.util.spec_from_file_location("rgpot_omp_crossover_v3", HARNESS)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load benchmark harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(atoms: int, threads: int, speedup: float, p_value: float) -> dict:
    return {
        "arm": "team-1",
        "potential": "synthetic",
        "stage": "build",
        "atoms": atoms,
        "threads": threads,
        "speedup": speedup,
        "speedup_ci95": [speedup, speedup],
        "mw_p_team_faster": p_value,
        "aa_band": [0.95, 1.05],
        "n_per_binary": 12,
        "stock_ns": [1.0],
        "team_ns": [1.0 / speedup],
        "order": "ABBA/BAAB",
    }


def payload(one_thread_speedup: float = 1.0) -> dict:
    measurements = [
        row(atoms, 1, one_thread_speedup, 0.5)
        for atoms in (512, 1024, 4096, 8192, 16384)
    ]
    measurements += [
        row(512, 2, 1.19, 0.001),
        row(1024, 2, 1.30, 0.001),
        row(4096, 2, 1.40, 0.001),
        row(8192, 2, 1.00, 0.5),
        row(16384, 2, 1.00, 0.5),
        row(512, 8, 1.99, 0.001),
        row(1024, 8, 2.10, 0.001),
        row(4096, 8, 2.20, 0.001),
        row(8192, 8, 1.00, 0.5),
        row(16384, 8, 1.00, 0.5),
    ]
    return {
        "metadata": {
            "arms": [
                {"name": "stock", "threshold": 8192},
                {"name": "team-1", "threshold": 1},
            ]
        },
        "measurements": measurements,
    }


def main() -> None:
    harness = load_harness()
    result = payload()
    harness.add_crossovers(result)
    found = {
        row["threads"]: row["atoms"]
        for row in result["crossovers"]
        if row["potential"] == "synthetic" and row["stage"] == "build"
    }
    if found != {2: 1024, 8: 1024}:
        raise SystemExit(f"wrong persistent crossovers: {found}")

    harness.add_recommendation(result)
    expected = [
        {
            "potential": "synthetic",
            "stage": "build",
            "team_min_atoms": 1024,
            "supporting_arm": "team-1",
        }
    ]
    if result.get("recommendations") != expected:
        raise SystemExit(f"wrong recommendation: {result.get('recommendations')}")

    regressed = payload(one_thread_speedup=0.90)
    harness.add_crossovers(regressed)
    harness.add_recommendation(regressed)
    if regressed["recommendations"][0]["team_min_atoms"] != 8192:
        raise SystemExit("one-thread regression did not retain the stock threshold")


if __name__ == "__main__":
    main()
