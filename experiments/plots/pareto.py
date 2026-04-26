"""Accuracy-vs-cost Pareto plot, MLIP-paper convention.

Each solver run is a point in `(cost, accuracy)` space. The Pareto front
is the staircase of non-dominated points: those not strictly worse than
any other point on both axes. Convention: lower cost is better, lower
objective value (closer to global minimum) is better.

The plot draws every run as a marker (one shape per solver), the Pareto
front as a step line, and tags points with the solver name on hover (in
matplotlib widgets) or via per-marker labels (static export).

Used in IISE Section 7 to compare the Boltzmann/Fast/GSA preset variants
across (n_epochs * steps_per_epoch) cost on the StybTang/Rosenbrock/
Rastrigin/Ackley benchmarks.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from chemparseplot.plot.theme import (
    RUHI_COLORS,
    apply_axis_theme,
    get_theme,
    setup_publication_theme,
)


_MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def _solver_color(idx: int) -> str:
    palette = [
        RUHI_COLORS["teal"],
        RUHI_COLORS["coral"],
        RUHI_COLORS["sky"],
        RUHI_COLORS["magenta"],
        RUHI_COLORS["sunshine"],
    ]
    return palette[idx % len(palette)]


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return the indices of non-dominated points in `(cost, value)` space.

    Both axes minimised. A point `i` is on the Pareto front iff no other
    point `j` satisfies `cost_j <= cost_i AND value_j <= value_i AND
    (cost_j < cost_i OR value_j < value_i)`.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {points.shape}")
    n = points.shape[0]
    front = []
    for i in range(n):
        ci, vi = points[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            cj, vj = points[j]
            if cj <= ci and vj <= vi and (cj < ci or vj < vi):
                dominated = True
                break
        if not dominated:
            front.append(i)
    front_idx = np.array(sorted(front, key=lambda k: points[k, 0]))
    return front_idx


def plot_pareto(
    runs: Sequence[tuple[str, np.ndarray]],
    *,
    cost_label: str = "Cost (function evaluations)",
    value_label: str = "Best objective value",
    title: str | None = None,
    out_path: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
):
    """Render an accuracy-vs-cost Pareto plot.

    Args:
        runs: iterable of `(solver_name, points_2d)` pairs where
              `points_2d` has shape `(n_runs, 2)` -- columns are
              `(cost, value)`.
        cost_label, value_label: axis labels.
        title: optional title.
        out_path: if set, savefig.
        log_x, log_y: log-scale toggles per axis.

    Returns: (fig, ax).
    """
    setup_publication_theme(get_theme("ruhi"))
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    apply_axis_theme(ax, get_theme("ruhi"))

    all_points = []
    for s_idx, (name, pts) in enumerate(runs):
        pts = np.asarray(pts, dtype=float)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=name,
            color=_solver_color(s_idx),
            marker=_MARKERS[s_idx % len(_MARKERS)],
            s=40,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.4,
        )
        all_points.append(pts)

    combined = np.vstack(all_points)
    front_idx = pareto_front(combined)
    front_pts = combined[front_idx]
    ax.step(
        front_pts[:, 0],
        front_pts[:, 1],
        where="post",
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Pareto front",
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(cost_label)
    ax.set_ylabel(value_label)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)

    if out_path:
        fig.savefig(out_path)

    return fig, ax
