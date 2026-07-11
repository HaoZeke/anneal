"""Experiment 3: paired finite-run best-position shift (manuscript Section 5.3).

Runs 32-seed paired f16 vs f64 SA on Styblinski-Tang 2D. Reports the
per-seed best position and best value for both dtypes. The summary script
(`scripts/exp3_summary.py`) computes the paired best-position shift and a
bootstrap CI; this script just emits the per-(seed, dtype) row.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


from experiments.shared.runner import sa_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--variant", default="boltzmann")
    p.add_argument("--objective", default="styb_tang_2d")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # exp3 explicitly disables log-domain acceptance so the f16 path
    # is the naive `u < exp(-delta/T)` form that underflows for moderate
    # delta/T ratios. exp4 (compensated) flips both compensated_delta_e
    # and log_domain_accept on so the comparison surfaces the precision
    # gap the manuscript Section 5.4 ablation targets.
    rows = []
    for seed in range(args.seeds):
        for dtype in ("float64", "float16"):
            r = sa_run(objective=args.objective, variant=args.variant,
                       dtype=dtype, seed=seed,
                       n_epochs=args.n_epochs,
                       steps_per_epoch=args.steps_per_epoch,
                       compensated_delta_e=False,
                       log_domain_accept=False)
            rows.append(dict(seed=seed, dtype=dtype,
                             mean_pos_x=float(r.best_pos[0]),
                             mean_pos_y=float(r.best_pos[1]),
                             best_val=r.best_val,
                             n_calls=r.n_calls))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "dtype", "mean_pos_x",
                                          "mean_pos_y", "best_val", "n_calls"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
