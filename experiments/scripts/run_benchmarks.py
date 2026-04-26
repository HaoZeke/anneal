"""Drive the benchmark catalog across the three preset variants and write
the long-form CSV consumed by the plot scripts."""

from __future__ import annotations

import argparse
import sys

from experiments.benchmarks.runner import run_benchmarks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--problems", default=None,
                   help="Comma-separated subset of problem ids; default is all.")
    args = p.parse_args()
    problems = args.problems.split(",") if args.problems else None
    rows = run_benchmarks(
        problems=problems,
        seeds=args.seeds,
        n_epochs=args.n_epochs,
        steps_per_epoch=args.steps_per_epoch,
        out_path=args.out,
    )
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
