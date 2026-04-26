"""Empirical witness for Theorem 7 (q-coherent MetaD effective heating).

Theorem 7 (design pass 18): the q-coherent well-tempered MetaD chain
at gamma = 1/(q_v - 1) has stationary distribution
    p_inf(x; T) = exp_q(-(q_v - 1) F(x) / T) / Z_inf(T)
in the vanishing-deposition limit.

This script witnesses the prediction empirically:

  1. Run a fixed-T bGSA-MetaD-q_v chain for T_TOTAL steps.
  2. Record F(x_t) values after burn-in.
  3. Histogram F values; fit the survival function
       P(F > f) ~ exp_q(-(q_v - 1) f / T)
     log-linearly to extract the empirical effective-beta beta_emp.
  4. Compare beta_emp to the predicted beta_pred = (q_v - 1) / T.

If Theorem 7 is correct, beta_emp / beta_pred = 1 +/- MC error
across q_v in the productive range. Any systematic deviation
falsifies the theorem (or the q-coherent gamma derivation).

Output: data/theorem7_witness_<landscape>.csv with columns
(q_v, beta_pred, beta_emp, beta_emp_se, ratio).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def fit_q_canonical_beta(samples_f, q_v, f_min=None, f_max=None):
    """Fit exp_q(-beta f) to the survival function of the F samples.

    Survival: S(f) = P(F > f). For exp_q-distributed F:
        S(f) ~ exp_q(-beta f) = (1 + (q_v - 1) beta f)^(1/(1-q_v))
        log S(f) = -1/(q_v - 1) * log(1 + (q_v - 1) beta f)
    Linearise: y = -log S(f), then y = 1/(q_v - 1) * log(1 + (q_v - 1) beta f).
    Define u = exp((q_v - 1) y) - 1; then u = (q_v - 1) beta f, so
    beta = u / ((q_v - 1) f) at each f-bin. Average over f-bins for an
    estimator with a per-sample SE.
    """
    samples_f = np.asarray(samples_f, dtype=float)
    samples_f = samples_f[np.isfinite(samples_f)]
    if f_min is None:
        f_min = float(np.quantile(samples_f, 0.10))
    if f_max is None:
        f_max = float(np.quantile(samples_f, 0.90))
    bins = np.linspace(f_min, f_max, 30)
    # Survival at each bin edge.
    n = len(samples_f)
    survival = np.array([np.mean(samples_f > b) for b in bins])
    # Need strictly positive survival to take log.
    keep = (survival > 1.0 / n) & (survival < 1.0 - 1.0 / n) & (bins > 0)
    if keep.sum() < 5:
        return float("nan"), float("nan")
    f_use = bins[keep]
    s_use = survival[keep]
    y = -np.log(s_use)
    # u = exp((q_v - 1) y) - 1
    u = np.exp((q_v - 1.0) * y) - 1.0
    # beta_per_bin = u / ((q_v - 1) f)
    beta_per_bin = u / ((q_v - 1.0) * f_use)
    return float(np.median(beta_per_bin)), \
           float(np.std(beta_per_bin) / max(np.sqrt(len(beta_per_bin)), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", default="rastrigin_5d")
    p.add_argument("--T", type=float, default=2.0,
                   help="Fixed temperature (no cooling)")
    p.add_argument("--qv-list", default="1.05,1.10,1.20")
    p.add_argument("--n-steps", type=int, default=50000)
    p.add_argument("--burn-in", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--w0-frac", type=float, default=0.05,
                   help="metad_w0 = w0_frac * T (default 0.05)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import demo_bgsa as d
    from metad_helpers import WellTemperedBias

    d.OBJ_FN, d.OBJ_GRAD, d.LOW, d.HIGH, _ = d.OBJECTIVES[args.objective]
    qv_list = [float(q) for q in args.qv_list.split(",")]
    out_csv = args.out or f"data/theorem7_witness_{args.objective}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    T = args.T
    rows = []
    # Theorem 7 prediction for the chain's mean F under
    # p_inf(x; T) = exp_q(-(q_v - 1) F / T) / Z:
    # for a q-Gaussian-like F-density, <F> ~ T / (q_v - 1) * c
    # where c is a landscape-dependent constant of O(1).
    # The KEY DIAGNOSTIC: <F>(q_v=q1) / <F>(q_v=q2) should equal
    # (q2 - 1) / (q1 - 1).
    # For p_inf(x) = exp_q(-(q_v-1) F / T) / Z, the median F is at
    # F_med ~ T/(q_v - 1) * (2^((q_v-1)) - 1) / (q_v - 1)
    # which simplifies to <F> ~ T * c(q) / (q_v - 1) for c(q) of O(1).
    # We report the predicted F_typical = T / (q_v - 1) for orientation;
    # the empirical <F> should match this scaling at moderate q_v where
    # the bias has converged to V_inf (large enough q_v that
    # gamma = 1/(q_v-1) is small enough for V to equilibrate).
    print(f"\nMean-F diagnostic (Theorem 7: typical F ~ T/(q_v-1)):")
    print(f"{'q_v':>6} {'gamma':>7} {'F_typ_pred':>11} {'<F>':>9} {'F/F_pred':>10}")
    for q_v in qv_list:
        gamma_q = d.metad_gamma_from_qv(q_v)
        beta_pred = (q_v - 1.0) / T
        all_betas = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed * 1009 + 17)
            sigma_rw = 0.4
            metad_sigma = sigma_rw
            metad_w0 = args.w0_frac * T
            bias = WellTemperedBias(d.LOW, d.HIGH, sigma=metad_sigma,
                                    w0=metad_w0, gamma=gamma_q)
            cur = rng.uniform(d.LOW, d.HIGH).astype(np.float64)
            cur_v = d.OBJ_FN(cur)
            f_samples = []
            accept_count = 0
            deposit_period = 20
            for step in range(args.n_steps):
                # q_a annealed-to-Metropolis schedule isn't applicable
                # at fixed T (we pick q_a = q_v throughout).
                q_a = q_v
                prop = np.clip(d.gaussian_propose(rng, cur, sigma_rw),
                               d.LOW, d.HIGH)
                pv = d.OBJ_FN(prop)
                cur_aug = cur_v + bias.potential(bias.cv(cur))
                prop_aug = pv + bias.potential(bias.cv(prop))
                if rng.random() < d.tsallis_accept_prob(
                        prop_aug - cur_aug, T, q_a):
                    cur, cur_v = prop, pv
                    accept_count += 1
                    if accept_count % deposit_period == 0:
                        bias.deposit(bias.cv(cur), T)
                if step >= args.burn_in:
                    f_samples.append(cur_v)
            f_arr = np.asarray(f_samples)
            beta_emp, beta_se = fit_q_canonical_beta(f_arr, q_v)
            mean_f = float(np.mean(f_arr))
            all_betas.append(beta_emp)
            rows.append({
                "q_v": q_v,
                "seed": seed,
                "T": T,
                "beta_pred": beta_pred,
                "beta_emp": beta_emp,
                "beta_emp_se": beta_se,
                "ratio_emp_to_pred": beta_emp / beta_pred if beta_pred > 0 else float("nan"),
                "mean_F": mean_f,
                "mean_F_times_qm1": mean_f * (q_v - 1.0),
                "n_accepts": accept_count,
            })
        mean_ratio = np.nanmean([r["ratio_emp_to_pred"] for r in rows
                                  if r["q_v"] == q_v])
        std_ratio = np.nanstd([r["ratio_emp_to_pred"] for r in rows
                                if r["q_v"] == q_v])
        mean_f_avg = np.nanmean([r["mean_F"] for r in rows
                                  if r["q_v"] == q_v])
        f_pred = T / (q_v - 1.0)
        ratio = mean_f_avg / f_pred
        print(f"{q_v:>6.3f} {gamma_q:>7.2f} {f_pred:>11.3f} "
              f"{mean_f_avg:>9.3f} {ratio:>10.3f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
