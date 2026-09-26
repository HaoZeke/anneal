"""Experiment 1: underflow grid (manuscript Section 5.1).

Sweeps the Metropolis acceptance kernel `p = exp(-delta_e / T)` over a grid
of `(delta_e, T, dtype)` cells. For each cell we draw `--samples` Bernoulli
trials at the explicit dtype and record the empirical accept rate plus the
standard error. The expected outcome: f16 underflows to zero for moderate
`delta_e / T` ratios where f64 still gives a small but non-zero probability.

Each cell uses an independent RNG (derived from the base seed via
SeedSequence) and shares one uniform stream across dtypes, so the reported
float16-vs-float64 acceptance-rate difference is the paired kernel-rounding
bias rather than Monte Carlo noise between independent streams (at 2e5
samples the per-stream standard error is ~1.1e-3, which would otherwise
swamp the signal).

The script writes one CSV row per `(delta_e, T, dtype)` cell with columns
delta_e, temp, dtype, n_samples, accept_rate, std_err. The `--check` flag
guards the empirical maximum: over cells where the f64 rate exceeds 1e-3
(below that the f16 underflow is genuine and the comparison does not apply),
the paired bias stays under MAX_F16_F64_BIAS, with the observed maximum
about 3.15e-4.
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
# Regression guard, set above the observed paired maximum (~3.15e-4) with a
# Monte Carlo margin; not a derived analytic bound.
MAX_F16_F64_BIAS = 5e-4


def empirical_accept_rate(delta_e, temp, dtype, u_base):
    """Bernoulli accept rate of `u < exp(-delta_e/T)` with all arithmetic at
    dtype, using the shared uniform draws `u_base` so the comparison across
    dtypes is paired (same stream, cast to each precision); the reported
    float16-vs-float64 bias then reflects the kernel rounding rather than
    Monte Carlo noise between independent streams."""
    delta_d = np.dtype(dtype).type(delta_e)
    temp_d = np.dtype(dtype).type(temp)
    if delta_d <= np.dtype(dtype).type(0):
        return 1.0, 0.0
    p = np.exp(-delta_d / temp_d)
    u = u_base.astype(dtype)
    n = u_base.size
    accepted = int((u < p).sum())
    rate = accepted / n
    se = float(np.sqrt(rate * (1.0 - rate) / n))
    return float(rate), se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--check",
        action="store_true",
        help=f"Assert the manuscript's |bias|<={MAX_F16_F64_BIAS:g} bound.",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # One independent RNG per (delta_e, T) cell, derived from the base seed via
    # SeedSequence, so the grid is reproducible and order-independent. Within a
    # cell the uniform stream is shared across dtypes (paired comparison).
    cells = [(delta_e, temp) for delta_e in DELTAS for temp in TEMPS]
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(cells))
    rows = []
    max_bias = 0.0
    max_bias_at = None
    for (delta_e, temp), child in zip(cells, child_seeds):
        rng = np.random.default_rng(child)
        u_base = rng.random(size=args.samples)
        cell = {}
        for dtype in DTYPES:
            rate, se = empirical_accept_rate(delta_e, temp, dtype, u_base)
            rows.append(
                dict(
                    delta_e=delta_e,
                    temp=temp,
                    dtype=dtype,
                    n_samples=args.samples,
                    accept_rate=rate,
                    std_err=se,
                )
            )
            cell[dtype] = rate

        if cell["float64"] >= 1e-3:
            bias = abs(cell["float16"] - cell["float64"])
            if bias > max_bias:
                max_bias, max_bias_at = bias, (delta_e, temp)
            if args.check:
                assert bias <= MAX_F16_F64_BIAS, (
                    f"f16/f64 bias {bias} > {MAX_F16_F64_BIAS} "
                    f"at delta_e={delta_e} T={temp}"
                )

    print(
        f"Max float16-vs-float64 bias {max_bias:.3e} at "
        f"(delta_e, T) = {max_bias_at} over cells with f64 rate >= 1e-3"
    )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "delta_e",
                "temp",
                "dtype",
                "n_samples",
                "accept_rate",
                "std_err",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
