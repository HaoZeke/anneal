#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Reprint the sealed campaign numbers the article prints.

The lines are the type 7 median and interquartile range, and the sealed
means, from the committed tables. Run from paper/jctc.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lj75_gmin_table
import water_results_table


def iqr(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [25, 75])
    return float(np.median(values)), float(high - low)


def main() -> int:
    _, arms = lj75_gmin_table.load(lj75_gmin_table.DATA)
    for arm in arms:
        calls = np.array([run.calls for run in arm.runs], dtype=float)
        walls = np.array([run.wall for run in arm.runs], dtype=float)
        call_median, call_iqr = iqr(calls)
        wall_median, wall_iqr = iqr(walls)
        print(f"lj75 label {arm.label}")
        print(f"lj75 hits {arm.hits} of {len(arm.runs)}")
        print(f"lj75 par2 mean {arm.sealed_par2}")
        print(f"lj75 call median {call_median:.6f} iqr {call_iqr:.6f}")
        print(f"lj75 wall median {wall_median:.6f} iqr {wall_iqr:.6f}")
    _, water, _ = water_results_table.load(water_results_table.DATA)
    for arm in water:
        print(f"water label {arm.label}")
        print(f"water hits {arm.summary['hits']} of {arm.summary['n']}")
        if arm.contrast is not None:
            p = arm.contrast["mcnemar_exact_p"]
            print(f"water mcnemar p {p}")
            print(f"water mcnemar p3 {water_results_table._p3(p)}")
            low, high = arm.contrast["newcombe_95"]
            print(f"water newcombe {low:.6f} {high:.6f}")
    fair = water_results_table.polish_vs_crest_fair(water)
    print(f"water polish vs crest_fair p {fair['mcnemar_exact_p']}")
    print(f"water polish vs crest_fair p3 {water_results_table._p3(fair['mcnemar_exact_p'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
