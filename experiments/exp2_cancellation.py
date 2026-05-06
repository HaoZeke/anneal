"""Experiment 2: cancellation table (manuscript Section 5.2).

Computes a finite-difference proxy for `delta_e = f(x + h) - f(x)` on the
Rosenbrock 2D objective at `x = [0.5, 0.5]`. Step `h = 10 * eps(dtype)`
chosen to put the difference into the lower-bit regime where catastrophic
cancellation dominates. For each dtype we record the relative error
against an f64 reference computed analytically.

Schema: dtype, h, delta_e_dtype, delta_e_ref, rel_err. The `--check` flag
asserts the manuscript's reported precision: f64 and f32 rel-err are
approximately 3.92e-3, and f16 rel-err is approximately 1.96e-2.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def rosenbrock(x, dtype):
    """Rosenbrock with all arithmetic at dtype."""
    x = np.asarray(x, dtype=dtype)
    a = (np.dtype(dtype).type(1.0) - x[0]) ** 2
    b = np.dtype(dtype).type(100.0) * (x[1] - x[0] ** 2) ** 2
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

    # f64 reference: analytical derivative of Rosenbrock at x = [0.5, 0.5].
    # df/dx0 = -2(1 - x0) - 400 * x0 * (x1 - x0^2)
    #        = -2*0.5      - 400 * 0.5 * (0.5 - 0.25)
    #        = -1.0        - 50.0
    #        = -51.0
    df_dx0_analytic = -51.0

    for dtype_name in ["float64", "float32", "float16"]:
        dtype = np.dtype(dtype_name)
        eps = float(np.finfo(dtype).eps)
        h = 10.0 * eps
        x_plus = list(x0)
        x_plus[h_dim] = x0[h_dim] + h
        delta_e = float(rosenbrock(x_plus, dtype) - rosenbrock(x0, dtype))
        delta_e_ref = df_dx0_analytic * h
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
        assert 0.003 <= rel_errs["float64"] <= 0.005, f"f64 rel_err {rel_errs['float64']}"
        assert 0.003 <= rel_errs["float32"] <= 0.005, f"f32 rel_err {rel_errs['float32']}"
        assert 0.015 <= rel_errs["float16"] <= 0.025, f"f16 rel_err {rel_errs['float16']}"


if __name__ == "__main__":
    sys.exit(main())
