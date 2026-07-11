"""Experiment 2: cancellation table (manuscript Section 5.2).

Computes `delta_e = f(x + h e0) - f(x)` on the Rosenbrock 2D objective at
`x = [0.5, 0.5]` with all arithmetic forced to each dtype. Step
`h = 10 * eps(dtype)` puts the subtraction into the lower-bit regime where
catastrophic cancellation dominates: the two evaluations are O(1) (here
f(x) = 6.5) and their difference is O(1e-13) at f64, so the subtraction
discards most significant digits.

The reference is the EXACT real-valued delta_e at the same real step h,
not a first-order Taylor term. Rosenbrock is a polynomial and x0, x1, and
h = 10 * eps are all dyadic rationals, so the exact delta_e is computed in
`fractions.Fraction` arithmetic. Referencing the exact delta_e (rather than
the analytic derivative df/dx0 * h) isolates the floating-point error of the
subtraction from the O(h) truncation of a forward difference; the latter is
identical across dtypes whose h scales with eps and would otherwise mask the
precision channel.

Schema: dtype, h, delta_e_dtype, delta_e_ref, rel_err. The `--check` flag
asserts the reported precision per dtype.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from fractions import Fraction

import numpy as np


def rosenbrock(x, dtype):
    """Rosenbrock with all arithmetic at dtype."""
    x = np.asarray(x, dtype=dtype)
    scalar = np.dtype(dtype).type
    dx = scalar(1.0) - x[0]
    x0_sq = x[0] * x[0]
    residual = x[1] - x0_sq
    a = dx * dx
    residual_sq = residual * residual
    b = scalar(100.0) * residual_sq
    return a + b


def rosenbrock_exact(x):
    """Rosenbrock evaluated exactly over rationals (no rounding)."""
    a = (Fraction(1) - x[0]) ** 2
    b = Fraction(100) * (x[1] - x[0] ** 2) ** 2
    return a + b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    x0 = [0.5, 0.5]
    h_dim = 0  # perturb the first coordinate
    rows = []
    rel_errs = {}

    for dtype_name in ["float64", "float32", "float16"]:
        dtype = np.dtype(dtype_name)
        eps = float(np.finfo(dtype).eps)
        h = 10.0 * eps
        x_plus = list(x0)
        x_plus[h_dim] = x0[h_dim] + h
        delta_e = float(rosenbrock(x_plus, dtype) - rosenbrock(x0, dtype))
        # Exact delta_e at the same real step h. h = 10 * eps is dyadic, so
        # Fraction(h) is the exact value of the float h; x0 is exact.
        x0_q = [Fraction(v) for v in x0]
        x_plus_q = list(x0_q)
        x_plus_q[h_dim] = x0_q[h_dim] + Fraction(h)
        delta_e_ref = float(rosenbrock_exact(x_plus_q) - rosenbrock_exact(x0_q))
        rel_err = abs(delta_e - delta_e_ref) / abs(delta_e_ref) if delta_e_ref != 0 else 0.0
        rel_errs[dtype_name] = rel_err
        rows.append(dict(dtype=dtype_name, h=h, delta_e_dtype=delta_e,
                         delta_e_ref=delta_e_ref, rel_err=rel_err))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dtype", "h", "delta_e_dtype",
                                          "delta_e_ref", "rel_err"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
    for r in rows:
        print(f"  {r['dtype']:8s}: rel_err = {r['rel_err']:.6f}")

    if args.check:
        # Against the exact delta_e, the f64/f32 relative error is the
        # scale-invariant cancellation constant eps * |f| / |delta_e| (the eps
        # cancels because h scales with eps); f16 carries an extra input- and
        # eval-rounding contribution at its larger step.
        assert 0.0035 <= rel_errs["float64"] <= 0.0045, f"f64 rel_err {rel_errs['float64']}"
        assert 0.0035 <= rel_errs["float32"] <= 0.0045, f"f32 rel_err {rel_errs['float32']}"
        assert 0.008 <= rel_errs["float16"] <= 0.011, f"f16 rel_err {rel_errs['float16']}"


if __name__ == "__main__":
    sys.exit(main())
