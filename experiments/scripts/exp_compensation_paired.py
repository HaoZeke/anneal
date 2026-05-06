"""Paired exp3-vs-exp4 statistical test for the manuscript's compensated
log-domain precision ablation.

Reads both `exp3_trajectory.csv` (uncompensated) and `exp4_compensated.csv`
(compensated). For each seed, computes the f16-vs-f64 best-position shift.
Pairs the per-seed shifts and runs a Wilcoxon signed-rank test on the
paired differences (compensated - uncompensated). Reports:
  - mean shift before / after
  - paired delta + 95 percent bootstrap CI
  - Wilcoxon W, p-value, effect size (rank-biserial)

Used by the verify env's `compensation-paired` task and cited in the
manuscript Section 5.4 narrative."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from scipy import stats


def load_paired_biases(uncomp_path, comp_path):
    """Load both CSVs and return per-seed (uncomp_bias, comp_bias) pairs."""
    def _by_seed(path):
        rows = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.setdefault(int(r["seed"]), {})[r["dtype"]] = r
        return rows

    uncomp = _by_seed(uncomp_path)
    comp = _by_seed(comp_path)

    pairs = []
    for seed in sorted(uncomp.keys()):
        if seed not in comp:
            continue
        if "float16" not in uncomp[seed] or "float64" not in uncomp[seed]:
            continue
        if "float16" not in comp[seed] or "float64" not in comp[seed]:
            continue
        uc = np.linalg.norm([
            float(uncomp[seed]["float16"]["mean_pos_x"]) - float(uncomp[seed]["float64"]["mean_pos_x"]),
            float(uncomp[seed]["float16"]["mean_pos_y"]) - float(uncomp[seed]["float64"]["mean_pos_y"]),
        ])
        c = np.linalg.norm([
            float(comp[seed]["float16"]["mean_pos_x"]) - float(comp[seed]["float64"]["mean_pos_x"]),
            float(comp[seed]["float16"]["mean_pos_y"]) - float(comp[seed]["float64"]["mean_pos_y"]),
        ])
        pairs.append((uc, c))
    return np.asarray(pairs)


def bootstrap_ci(values, n_boot=10_000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uncomp", default="data/exp3_trajectory.csv")
    p.add_argument("--comp", default="data/exp4_compensated.csv")
    p.add_argument("--halving-tolerance", type=float, default=0.05,
                   help="Allow mean(comp) up to (0.5 + tol) * mean(uncomp).")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    pairs = load_paired_biases(args.uncomp, args.comp)
    if len(pairs) == 0:
        print(f"No paired seeds in {args.uncomp} and {args.comp}", file=sys.stderr)
        return 1
    uncomp_b, comp_b = pairs[:, 0], pairs[:, 1]
    deltas = comp_b - uncomp_b

    print(f"Paired f16-vs-f64 best-position shift on {len(pairs)} seeds:")
    m_uc, lo_uc, hi_uc = bootstrap_ci(uncomp_b)
    m_c, lo_c, hi_c = bootstrap_ci(comp_b)
    print(f"  uncompensated:  mean = {m_uc:.4e}  95% CI = [{lo_uc:.4e}, {hi_uc:.4e}]")
    print(f"  compensated:    mean = {m_c:.4e}  95% CI = [{lo_c:.4e}, {hi_c:.4e}]")

    if np.allclose(deltas, 0):
        # Wilcoxon undefined when all differences are zero. This happens at
        # f64 reference (Kahan correction is exact zero); the experiments
        # run at f16 so deltas should never all be zero, but guard anyway.
        print("  paired deltas are identically zero; Wilcoxon undefined")
        ratio = m_c / m_uc if m_uc != 0 else float("nan")
    else:
        w_stat, p_val = stats.wilcoxon(deltas, alternative="less")
        # Rank-biserial effect size for paired Wilcoxon.
        n = len(deltas)
        rb = 1.0 - 2.0 * w_stat / (n * (n + 1) / 2.0)
        print(f"  Wilcoxon W = {w_stat:.2f}  p (one-sided, comp < uncomp) = {p_val:.4g}")
        print(f"  rank-biserial effect size = {rb:.3f}")
        ratio = m_c / m_uc if m_uc != 0 else float("nan")

    print(f"  ratio mean(comp) / mean(uncomp) = {ratio:.3f}")
    halving_target = 0.5 + args.halving_tolerance
    halved = ratio <= halving_target
    print(f"  legacy halving check (ratio <= {halving_target}): {'PASS' if halved else 'FAIL'}")

    if args.check:
        assert halved, (
            f"legacy halving check fails: ratio {ratio:.3f} > {halving_target}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
