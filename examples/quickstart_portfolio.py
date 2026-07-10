#!/usr/bin/env python3
"""Minimal stand-alone entry: budget-only portfolio + classical preset.

Run after: pip install anneal
  python examples/quickstart_portfolio.py
"""
from __future__ import annotations

import numpy as np

from anneal import Boltzmann, global_optimize, run


def rastrigin(x: np.ndarray) -> float:
    return float(10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def main() -> None:
    low = np.full(5, -5.0)
    high = np.full(5, 5.0)
    out = global_optimize(rastrigin, low, high, budget=2000, seed=0)
    print("portfolio best_val", float(out["best_val"]))
    print("portfolio best_pos", np.asarray(out["best_pos"]))

    low2 = np.array([-5.0, -5.0])
    high2 = np.array([5.0, 5.0])

    def rosenbrock(x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    h = run(
        rosenbrock,
        low2,
        high2,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=25,
        steps_per_epoch=40,
        seed=42,
    )
    print("boltzmann best_val", float(h.best_val))


if __name__ == "__main__":
    main()
