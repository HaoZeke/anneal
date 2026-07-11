"""Empirical convergence-rate sweep for the Tsallis-coherence
corollary (design pass 16 Corollary 2). Runs bGSA-q_v at a sweep of
q_v values and records the per-feval best_val trajectory; produces
the log-log convergence plot showing the q_v fibres' rate scaling.

The expected story (Tsallis-Stariolo 1996 + design pass 16):

  - q_v = 1: classical Boltzmann SA, T(k) ~ ln 2 / ln(1+k) (log
    cooling). Hajek 1988: best_val converges to global min only as
    k -> infinity, with O(exp(c/eps)) compute to reach within eps.
  - q_v in (1, 1 + 2/d): polynomial cooling T(k) ~ k^{-(q-1)/(2-q)},
    polynomial visiting heavy-tail. Polynomial mixing time.
  - q_v > 1 + 2/d - epsilon: visiting density's first moment
    diverges; chain takes "infinite jumps", fails to mix.

The pilot's MAP q_v is typically 1.10-1.30, well inside the
polynomial regime. This script provides the empirical witness that
the convergence rate IS polynomial at the pilot's MAP.

Outputs:
  - data/qv_convergence_<landscape>.csv: per-(seed, q_v, k) records.
  - data/figs/qv_convergence_<landscape>.png: log-log plot.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--objective",
        default="rosenbrock_5d",
        choices=["rastrigin_5d", "rosenbrock_5d", "schwefel_20d"],
    )
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--k-per-epoch", type=int, default=100)
    p.add_argument("--qv-list", default="1.05,1.20,1.40,1.60,1.80")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-fig", default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import demo_bgsa as d

    d.OBJ_FN, d.OBJ_GRAD, d.LOW, d.HIGH, _ = d.OBJECTIVES[args.objective]
    qv_list = [float(q) for q in args.qv_list.split(",")]
    out_csv = args.out_csv or f"data/qv_convergence_{args.objective}.csv"
    out_fig = args.out_fig or f"data/figs/qv_convergence_{args.objective}.png"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_fig) or ".", exist_ok=True)

    # Run the trajectory: at each epoch we record best_val and fevals.
    # We use the q-HMC kernel directly (hmc_sa) at a fixed pilot-derived
    # epsilon and L so the only varying parameter is q_v (-> Tsallis
    # cooling shape, q-Gaussian visiting, Tsallis acceptance via the
    # default q_a = 1.0 inside hmc_sa).
    rows = []
    for seed in range(args.seeds):
        # Fit pilot once per seed, reuse hyperparameters across q_v.
        out = d.run_pilot(seed, 16, 60, dim=len(d.LOW))
        t_map, e_map, L_map, q_map_pilot, sigma_map, bp, pc, t_hot, t_rw_map, _ = out
        # Use sane t_init / eps / L from pilot; vary q_v across sweep.
        for q_v in qv_list:
            # Run the chain in 10 segments, recording best_val at each.
            cur = (
                bp.copy()
                if bp is not None
                else np.random.default_rng(seed)
                .uniform(d.LOW, d.HIGH)
                .astype(np.float64)
            )
            cur_v = d.OBJ_FN(cur)
            best = cur_v
            n_calls = 1
            n_segments = 10
            seg_epochs = max(1, args.n_epochs // n_segments)
            for seg in range(n_segments):
                bv_seg, nc_seg, cur_new = d.hmc_sa(
                    seed * 9001 + seg,
                    seg_epochs,
                    args.k_per_epoch,
                    t_map,
                    e_map,
                    L_map,
                    x0=cur,
                    q=q_v,
                )
                cur = cur_new
                cur_v = d.OBJ_FN(cur)
                n_calls += nc_seg
                if bv_seg < best:
                    best = bv_seg
                rows.append(
                    {
                        "seed": seed,
                        "q_v": q_v,
                        "segment": seg + 1,
                        "fevals": n_calls,
                        "best_val": best,
                    }
                )
        # Also record classical SA baseline (q_v -> 1 fibre + log cooling).
        cur = np.random.default_rng(seed).uniform(d.LOW, d.HIGH).astype(np.float64)
        cur_v = d.OBJ_FN(cur)
        best = cur_v
        n_calls = 1
        n_segments = 10
        seg_epochs = max(1, args.n_epochs // n_segments)
        for seg in range(n_segments):
            bv_seg, nc_seg, _ = d.classical_sa(
                seed * 9001 + seg,
                seg_epochs,
                args.k_per_epoch,
                t_map,
                sigma=sigma_map,
                x0=cur,
            )
            n_calls += nc_seg
            if bv_seg < best:
                best = bv_seg
            rows.append(
                {
                    "seed": seed,
                    "q_v": -1.0,  # sentinel for classical
                    "segment": seg + 1,
                    "fevals": n_calls,
                    "best_val": best,
                }
            )

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["seed", "q_v", "segment", "fevals", "best_val"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")

    # Render the figure.
    try:
        import matplotlib.pyplot as plt
        from chemparseplot.plot.theme import (
            RUHI_COLORS,
            apply_axis_theme,
            get_theme,
            setup_publication_theme,
        )

        setup_publication_theme(get_theme("ruhi"))
    except ImportError:
        print("matplotlib/chemparseplot unavailable; skipping figure")
        return 0

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = [
        RUHI_COLORS["teal"],
        RUHI_COLORS["coral"],
        RUHI_COLORS["sunshine"],
        RUHI_COLORS["sky"],
        RUHI_COLORS["magenta"],
    ]
    qv_groups = sorted({r["q_v"] for r in rows})
    for i, q_v in enumerate(qv_groups):
        sub = [r for r in rows if r["q_v"] == q_v]
        if not sub:
            continue
        # Mean across seeds at each segment.
        seg_to_pts = defaultdict(list)
        for r in sub:
            seg_to_pts[r["segment"]].append((r["fevals"], r["best_val"]))
        segs = sorted(seg_to_pts.keys())
        xs = [np.mean([p[0] for p in seg_to_pts[s]]) for s in segs]
        ys = [np.mean([p[1] for p in seg_to_pts[s]]) for s in segs]
        ys_lo = [np.quantile([p[1] for p in seg_to_pts[s]], 0.25) for s in segs]
        ys_hi = [np.quantile([p[1] for p in seg_to_pts[s]], 0.75) for s in segs]
        label = "classical SA (log cool)" if q_v < 0 else f"q_v = {q_v:.2f}"
        color = palette[i % len(palette)]
        ax.plot(xs, ys, "-o", label=label, color=color, markersize=4)
        ax.fill_between(xs, ys_lo, ys_hi, alpha=0.2, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective (median + IQR)")
    ax.set_title(f"q_v fibre convergence on {args.objective}")
    ax.legend(loc="best", fontsize=8)
    apply_axis_theme(ax, get_theme("ruhi"))
    fig.tight_layout()
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    fig.savefig(out_fig.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Wrote {out_fig} (and .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
