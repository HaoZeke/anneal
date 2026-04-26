"""Experiment 1: underflow grid (manuscript Section 5.1).

Sweeps the Metropolis acceptance kernel `p = exp(-delta_e / T)` over a grid
of `(delta_e, T, dtype)` cells. For each cell we draw `--samples` Bernoulli
trials at the explicit dtype and record the empirical accept rate plus the
standard error. The expected outcome: f16 underflows to zero for moderate
`delta_e / T` ratios where f64 still gives a small but non-zero probability.

The script writes one CSV row per `(delta_e, T, dtype)` cell with columns
delta_e, temp, dtype, n_samples, accept_rate, std_err. The `--check` flag
asserts the manuscript bound: `|accept_rate(f16) - accept_rate(f64)| <= 3e-3`
for every cell where the f64 rate exceeds 1e-3 (below that the f16 underflow
is genuine and the bound does not apply).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

DELTAS = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
TEMPS = [0.01, 0.1, 1.0, 10.0, 100.0]
DTYPES = ["float16", "float32", "float64"]


def empirical_accept_rate(delta_e, temp, dtype, n_samples, rng):
    """Bernoulli draw of `u < exp(-delta_e/T)` with all arithmetic at dtype."""
    delta_d = np.dtype(dtype).type(delta_e)
    temp_d = np.dtype(dtype).type(temp)
    if delta_d <= np.dtype(dtype).type(0):
        return 1.0, 0.0
    p = np.exp(-delta_d / temp_d)
    u = rng.random(size=n_samples).astype(dtype)
    accepted = (u < p).sum()
    rate = float(accepted / n_samples)
    se = float(np.sqrt(rate * (1.0 - rate) / n_samples))
    return rate, se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true",
                   help="Assert the manuscript's |bias|<=3e-3 bound.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for delta_e in DELTAS:
        for temp in TEMPS:
            cell = {}
            for dtype in DTYPES:
                rate, se = empirical_accept_rate(delta_e, temp, dtype, args.samples, rng)
                rows.append(dict(delta_e=delta_e, temp=temp, dtype=dtype,
                                 n_samples=args.samples,
                                 accept_rate=rate, std_err=se))
                cell[dtype] = rate

            if args.check and cell["float64"] >= 1e-3:
                bias = abs(cell["float16"] - cell["float64"])
                assert bias <= 3e-3, (
                    f"f16/f64 bias {bias} > 3e-3 at delta_e={delta_e} T={temp}"
                )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["delta_e", "temp", "dtype",
                                          "n_samples", "accept_rate", "std_err"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
