"""Cross-landscape benchmark figure for the IISE / unifier manuscript.

Reads the three demo_bgsa CSVs (Rastrigin 5D, Rosenbrock 5D,
Schwefel 20D), computes per-driver mean / std / 95%-upper, and
renders a 3-panel comparison using the ruhi colour palette. The
plot's headline statistic is the bGSA-flavoured 95%-upper bound on
best_val (design pass 10 Section 5).

Usage:
    pixi run -e verify python experiments/scripts/plot_cross_landscape.py \\
        --rastrigin data/bgsa_rastrigin.csv \\
        --rosenbrock data/bgsa_rosenbrock.csv \\
        --schwefel data/bgsa_schwefel.csv \\
        --out-dir data/figs

The driver list / order is fixed so the figure compares apples-to-
apples across landscapes; new drivers won't silently appear without
an explicit code change.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from chemparseplot.plot.theme import (
    RUHI_COLORS,
    apply_axis_theme,
    get_theme,
    setup_publication_theme,
)


# Driver display order matches the cross-landscape design pass 14
# table grouping: classical baselines first, gradient-driven bGSA
# variants next, then bias-augmented bGSA variants.
DRIVER_ORDER = [
    "classical_sa",
    "classical_sa_advanced",
    "bgsa",
    "bgsa_pt_hybrid_v2",
    "bgsa_metad",
    "bgsa_pt_metad",
    # bgsa_continuous_temper omitted: the Wu-Stoltz integrator
    # currently lands far from the cold-T target in the standard
    # configuration; revisit once issue 006's tuning is finalised.
    # bgsa_auto rows are handled separately (driver name is bgsa_auto[<chosen>]).
]

DRIVER_LABELS = {
    "classical_sa": "classical SA",
    "classical_sa_advanced": "classical+",
    "bgsa": "bGSA",
    "bgsa_pt_hybrid_v2": "bGSA-PT-hybrid+",
    "bgsa_metad": "bGSA-MetaD",
    "bgsa_pt_metad": "bGSA-PT-MetaD",
    "bgsa_continuous_temper": "bGSA-cont-temp",
}


def _driver_color(driver: str) -> str:
    """Map driver -> ruhi colour. Classical baselines use teal-family;
    gradient bGSA uses coral; bias-augmented uses sunshine/sky."""
    if driver.startswith("classical"):
        return RUHI_COLORS["teal"]
    if driver in ("bgsa", "bgsa_pt_hybrid"):
        return RUHI_COLORS["coral"]
    if "metad" in driver:
        return RUHI_COLORS["sunshine"]
    return RUHI_COLORS["sky"]


def load_csv(path: str) -> dict:
    """Returns {driver: {best_vals: list, fevals: list}}."""
    out: dict = defaultdict(lambda: {"best_vals": [], "fevals": []})
    with open(path) as f:
        for r in csv.DictReader(f):
            d = r["driver"]
            out[d]["best_vals"].append(float(r["best_val"]))
            out[d]["fevals"].append(int(r["fevals"]))
    return out


def summarise(rows: dict, driver: str) -> tuple[float, float, float, float]:
    """Returns (mean, std, q95_upper, mean_fevals) for the driver."""
    vals = np.asarray(rows[driver]["best_vals"], dtype=float)
    fevals = np.asarray(rows[driver]["fevals"], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q95 = float(np.quantile(vals, 0.95)) if len(vals) > 1 else float(vals[0])
    return float(vals.mean()), float(vals.std()), q95, float(fevals.mean())


def render_panel(ax, rows: dict, title: str, ylabel_units: str = "") -> None:
    drivers = [d for d in DRIVER_ORDER if d in rows]
    n = len(drivers)
    means = []
    stds = []
    q95s = []
    fevals = []
    for d in drivers:
        m, s, q95, fv = summarise(rows, d)
        means.append(m)
        stds.append(s)
        q95s.append(q95)
        fevals.append(fv)
    x = np.arange(n)
    colors = [_driver_color(d) for d in drivers]
    # Bar = mean, error bar = std, marker = 95%-upper.
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
           edgecolor="white", linewidth=0.5)
    ax.scatter(x, q95s, marker="v", color="black", s=40,
               zorder=10, label="95%-upper")
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVER_LABELS.get(d, d) for d in drivers],
                       rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"best objective {ylabel_units}".strip())
    ax.set_title(title, fontsize=11)
    apply_axis_theme(ax, get_theme("ruhi"))
    # Annotate fevals above the highest bar.
    for xi, fv in zip(x, fevals):
        ax.annotate(f"{int(fv/1000)}k", xy=(xi, ax.get_ylim()[1] * 0.96),
                    ha="center", va="top", fontsize=7, alpha=0.6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rastrigin", required=True,
                   help="bGSA demo CSV for Rastrigin 5D")
    p.add_argument("--rosenbrock", required=True,
                   help="bGSA demo CSV for Rosenbrock 5D")
    p.add_argument("--schwefel", required=True,
                   help="bGSA demo CSV for Schwefel 20D")
    p.add_argument("--out-dir", default="data/figs")
    args = p.parse_args()

    setup_publication_theme(get_theme("ruhi"))
    os.makedirs(args.out_dir, exist_ok=True)

    rast = load_csv(args.rastrigin)
    rosen = load_csv(args.rosenbrock)
    schwef = load_csv(args.schwefel)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    render_panel(axes[0], rast, "Rastrigin 5D (regular cups)")
    render_panel(axes[1], rosen, "Rosenbrock 5D (narrow valley)")
    render_panel(axes[2], schwef, "Schwefel 20D (deceptive)")
    fig.suptitle("Cross-landscape: bGSA vs classical SA "
                 "(8 seeds, mean ± std, 95%-upper triangles)",
                 fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)

    out_pdf = os.path.join(args.out_dir, "cross_landscape.pdf")
    out_png = os.path.join(args.out_dir, "cross_landscape.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")

    # Print headline numbers for the manuscript.
    print("\nHeadline summary:")
    for label, rows in [("Rastrigin 5D", rast),
                         ("Rosenbrock 5D", rosen),
                         ("Schwefel 20D", schwef)]:
        print(f"  {label}:")
        best_classical = summarise(rows, "classical_sa")
        for d in DRIVER_ORDER:
            if d in rows:
                m, s, q95, fv = summarise(rows, d)
                rel = (m - best_classical[0]) / max(abs(best_classical[0]), 1e-12)
                marker = "*" if d.startswith("bgsa") and m < best_classical[0] else " "
                print(f"    {marker} {DRIVER_LABELS[d]:<18} "
                      f"mean={m:8.3f}  std={s:6.3f}  "
                      f"95-upper={q95:8.3f}  rel={rel:+6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
