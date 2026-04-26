"""Experiment 4: compensated DeltaE ablation (manuscript Section 5.4).

Same as exp3 but with `compensated_delta_e=True`. The manuscript claim is
that Kahan compensation roughly halves the f16 first-moment bias without
eliminating it. We emit the per-(seed, dtype) row in the same schema as
exp3 so the summary script can do a paired comparison.
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

    # exp4 enables BOTH compensated_delta_e (running Kahan on cur_val)
    # AND log_domain_accept (avoids exp() underflow at f16). The
    # comparison vs exp3 surfaces the bias reduction from the
    # combined precision-correctness pair, matching manuscript Sec 5.4.
    rows = []
    for seed in range(args.seeds):
        for dtype in ("float64", "float16"):
            r = sa_run(objective=args.objective, variant=args.variant,
                       dtype=dtype, seed=seed,
                       n_epochs=args.n_epochs,
                       steps_per_epoch=args.steps_per_epoch,
                       compensated_delta_e=True,
                       log_domain_accept=True)
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
