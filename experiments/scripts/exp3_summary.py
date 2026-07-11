"""Summarises exp3 (and exp4) trajectory CSVs: paired f16-vs-f64
best-position shift plus bootstrap CI. Used to verify the manuscript
Section 5.3/5.4 precision claims at run time."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np


def load_rows(path):
    rows_by_seed = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            seed = int(r["seed"])
            rows_by_seed.setdefault(seed, {})[r["dtype"]] = r
    return rows_by_seed


def paired_bias(rows_by_seed):
    """Per-seed f16-vs-f64 best-position shift.

    Returns the L2 norm of the paired coordinate difference for each seed.
    """
    diffs = []
    for seed, by_dtype in sorted(rows_by_seed.items()):
        if "float16" not in by_dtype or "float64" not in by_dtype:
            continue
        x16 = np.array(
            [
                float(by_dtype["float16"]["mean_pos_x"]),
                float(by_dtype["float16"]["mean_pos_y"]),
            ]
        )
        x64 = np.array(
            [
                float(by_dtype["float64"]["mean_pos_x"]),
                float(by_dtype["float64"]["mean_pos_y"]),
            ]
        )
        diffs.append(np.linalg.norm(x16 - x64))
    return np.array(diffs)


def bootstrap_ci(samples, n_boot=10_000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    means = np.array(
        [
            rng.choice(samples, size=len(samples), replace=True).mean()
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(samples.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument(
        "--manuscript-bias",
        type=float,
        default=2.0582e-1,
        help="Reference value from the manuscript.",
    )
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=10.0,
        help="Pass if measured bias is within +/- this percent.",
    )
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    rows = load_rows(args.csv_path)
    diffs = paired_bias(rows)
    if len(diffs) == 0:
        print("No paired (f16, f64) seeds found in CSV.")
        return 1
    mean, lo, hi = bootstrap_ci(diffs)

    print(f"Loaded {len(diffs)} paired seeds from {args.csv_path}")
    print(
        f"  paired f16-vs-f64 best-position shift: {mean:.4e}  (95% CI: [{lo:.4e}, {hi:.4e}])"
    )
    print(f"  manuscript reference:                {args.manuscript_bias:.4e}")
    if args.manuscript_bias != 0:
        rel = abs(mean - args.manuscript_bias) / args.manuscript_bias * 100
        print(f"  relative deviation:                  {rel:.1f}%")
        if args.check:
            assert rel <= args.tolerance_pct, (
                f"bias deviation {rel:.1f}% exceeds tolerance {args.tolerance_pct}%"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
