#!/usr/bin/env python3
"""Relax 13 Lennard-Jones atoms held by a ChemFit Fitter inside a box of +-3.

The atomic positions are the fitted parameters, one ``(13, 3)`` leaf of the
Fitter's parameter tree. ``anneal.chemfit.fit`` starts each method at those
positions, evaluates only points inside the box and stops at the budget.

Run after: pip install anneal, and ChemFit from its ``next`` branch
  python examples/chemfit_lj_positions.py [budget]
"""

from __future__ import annotations

import sys

import chemfit
import numpy as np

import anneal.chemfit

N_ATOMS = 13
BOX = 3.0
LJ13_MINIMUM = -44.326801


def lj_energy(params: dict) -> float:
    pos = params["positions"]
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d * d).sum(-1)[np.triu_indices(len(pos), 1)]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inv6 = (1.0 / r2) ** 3
        return float(4.0 * np.sum(inv6 * inv6 - inv6))


def run_anneal(benchmark_context: dict, method: str = "portfolio") -> dict:
    """Fit the context's Fitter with one of anneal's methods."""
    return anneal.chemfit.fit(
        benchmark_context["fitter"],
        benchmark_context["budget"],
        method=method,
        bounds={"positions": (-BOX, BOX)},
    )


def main() -> None:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    start = np.random.default_rng(0).uniform(-1.5, 1.5, size=(N_ATOMS, 3))
    evaluated: list[np.ndarray] = []

    def loss(params: dict) -> float:
        evaluated.append(params["positions"].copy())
        return lj_energy(params)

    print(f"start energy {lj_energy({'positions': start}):.4f}")
    print(f"LJ13 global minimum {LJ13_MINIMUM}")
    for method in anneal.chemfit.METHODS:
        evaluated.clear()
        fitter = chemfit.Fitter(loss, initial_params={"positions": start})
        best = run_anneal({"fitter": fitter, "budget": budget}, method)
        inside = all(np.abs(pos).max() <= BOX for pos in evaluated)
        print(
            f"{method:>9}: best energy {lj_energy(best):9.4f} after "
            f"{len(evaluated)} evaluations, all inside the box: {inside}"
        )


if __name__ == "__main__":
    main()
