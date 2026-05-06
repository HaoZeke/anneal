"""Dolan-Moré performance profile (doi:10.1007/s101070100263).

For solver `s` and problem `p`, define `t_{p,s}` as the cost
(objective-equivalent evaluations or wall time) needed to solve `p` to
within a tolerance.
The performance ratio is
  r_{p,s} = t_{p,s} / min_{s'} t_{p,s'}
and the performance profile is
  rho_s(tau) = (1/|P|) * |{p : r_{p,s} <= tau}|.

`rho_s(1)` is the fraction of problems on which solver `s` is the
fastest; `lim_{tau -> inf} rho_s(tau)` is the fraction of problems
solved at all. Failed runs get `r_{p,s} = +inf` so they never enter
the profile.

This module ships a single helper, `plot_performance_profile`, that
takes a 2D matrix of costs (rows = problems, cols = solvers) and a
list of solver names, and returns the matplotlib figure with the ruhi
theme applied. Failed runs are encoded as `numpy.nan` (not 0) in the
matrix; the helper translates them to `+inf` internally.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from chemparseplot.plot.theme import (
    RUHI_COLORS,
    apply_axis_theme,
    get_theme,
    setup_publication_theme,
)


def _solver_color(idx: int) -> str:
    palette = list(plt.get_cmap("tab20").colors)
    return palette[idx % len(palette)]


def _solver_linestyle(idx: int) -> str:
    styles = ["-", "--", "-.", ":"]
    return styles[(idx // 20) % len(styles)]


def _place_legend(ax, n_items: int):
    ncol = 2 if n_items <= 6 else 3
    return ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=ncol,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.4,
    )


def plot_performance_profile(
    costs: np.ndarray,
    solver_names: Sequence[str],
    *,
    tau_max: float = 8.0,
    n_tau: int = 256,
    log_x: bool = True,
    title: str | None = None,
    out_path: str | None = None,
):
    """Render a Dolan-Moré performance profile.

    Args:
        costs: shape `(n_problems, n_solvers)`. Failed runs are NaN.
        solver_names: one label per column of `costs`.
        tau_max: upper bound on the x-axis ratio (default 8x).
        n_tau: number of tau points sampled.
        log_x: x-axis on log2 scale (Dolan-Moré convention).
        title: optional axis title.
        out_path: if set, write the figure to this path with savefig.

    Returns: (fig, ax) for further customisation.
    """
    setup_publication_theme(get_theme("ruhi"))
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2:
        raise ValueError(f"costs must be 2D, got shape {costs.shape}")
    n_problems, n_solvers = costs.shape
    if len(solver_names) != n_solvers:
        raise ValueError(
            f"solver_names has {len(solver_names)} entries; costs has {n_solvers} columns"
        )

    work = costs.copy()
    work[~np.isfinite(work)] = np.inf  # NaN -> inf
    best_per_problem = work.min(axis=1, keepdims=True)
    # Guard: if every solver failed on a problem, that row drops out.
    valid_problems = np.isfinite(best_per_problem.ravel())
    if not valid_problems.any():
        raise ValueError("no problem was solved by any solver; profile is empty")
    positive_best = valid_problems & (best_per_problem.ravel() > 0.0)
    ratios = np.full_like(work, np.inf, dtype=float)
    np.divide(
        work,
        best_per_problem,
        out=ratios,
        where=np.isfinite(work) & positive_best[:, None],
    )

    if log_x:
        taus = np.logspace(0.0, np.log2(tau_max), n_tau, base=2.0)
    else:
        taus = np.linspace(1.0, tau_max, n_tau)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    apply_axis_theme(ax, get_theme("ruhi"))

    for s_idx, name in enumerate(solver_names):
        rho = np.array(
            [
                float(np.sum(ratios[valid_problems, s_idx] <= tau))
                / valid_problems.sum()
                for tau in taus
            ]
        )
        ax.plot(
            taus,
            rho,
            label=name,
            color=_solver_color(s_idx),
            linestyle=_solver_linestyle(s_idx),
            linewidth=2.0,
        )

    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlim(1.0, tau_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Performance ratio $\tau$")
    ax.set_ylabel(r"Fraction within $\tau$ of best")
    if title:
        ax.set_title(title)
    _place_legend(ax, n_solvers)
    fig.subplots_adjust(bottom=0.34)

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, ax
