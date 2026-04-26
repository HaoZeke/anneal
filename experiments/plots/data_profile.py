"""Moré-Wild data profile (doi:10.1137/080724083).

For solver `s` and problem `p`, define a "solved at budget kappa" predicate
  H_{p,s}(kappa) = 1 iff t_{p,s} <= kappa * (n_p + 1)
where `n_p` is the problem's dimension and `t_{p,s}` is function evals to
reach a tolerance. The data profile is
  d_s(kappa) = (1/|P|) * sum_p H_{p,s}(kappa).

Reading: d_s(kappa) is the fraction of problems solved by solver s within
`kappa` simplex-gradient-equivalent budgets. Complement to the Dolan-Moré
performance profile; the two together summarise solver performance per
the manuscript's IISE reporting conventions.
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
    palette = [
        RUHI_COLORS["teal"],
        RUHI_COLORS["coral"],
        RUHI_COLORS["sky"],
        RUHI_COLORS["magenta"],
        RUHI_COLORS["sunshine"],
    ]
    return palette[idx % len(palette)]


def plot_data_profile(
    fevals: np.ndarray,
    dims: np.ndarray,
    solver_names: Sequence[str],
    *,
    kappa_max: float = 200.0,
    n_kappa: int = 256,
    title: str | None = None,
    out_path: str | None = None,
):
    """Render a Moré-Wild data profile.

    Args:
        fevals: shape `(n_problems, n_solvers)` of function evaluations to
                reach the tolerance. NaN encodes a failed run.
        dims: shape `(n_problems,)` -- problem dimensions `n_p`.
        solver_names: one label per column.
        kappa_max: upper bound on the budget ratio.
        n_kappa: number of kappa points sampled.
        title: optional axis title.
        out_path: if set, write the figure to this path.

    Returns: (fig, ax).
    """
    setup_publication_theme(get_theme("ruhi"))
    fevals = np.asarray(fevals, dtype=float)
    dims = np.asarray(dims, dtype=float)
    n_problems, n_solvers = fevals.shape
    if dims.shape != (n_problems,):
        raise ValueError(f"dims shape {dims.shape} != ({n_problems},)")
    if len(solver_names) != n_solvers:
        raise ValueError(
            f"solver_names has {len(solver_names)} entries; fevals has {n_solvers} columns"
        )

    work = fevals.copy()
    work[~np.isfinite(work)] = np.inf
    budget = work / (dims[:, None] + 1.0)
    kappas = np.linspace(0.0, kappa_max, n_kappa)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    apply_axis_theme(ax, get_theme("ruhi"))

    for s_idx, name in enumerate(solver_names):
        d = np.array([float(np.sum(budget[:, s_idx] <= k)) / n_problems for k in kappas])
        ax.plot(kappas, d, label=name, color=_solver_color(s_idx), linewidth=2.0)

    ax.set_xlim(0.0, kappa_max)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Budget $\kappa$ (simplex gradients)")
    ax.set_ylabel(r"$d_s(\kappa)$ -- fraction of problems solved")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)

    if out_path:
        fig.savefig(out_path)

    return fig, ax
