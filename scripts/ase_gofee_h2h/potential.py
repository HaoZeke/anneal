# Copyright (c) anneal developers
"""Untruncated Lennard-Jones potential, the same pair term as anneal.

Energy is ``4 (r^-12 - r^-6)``. The gradient matches
``PairPotential::lennard_jones``: ``(dV/dr) / r`` times the separation.
"""

from __future__ import annotations

import time

import numpy as np
from ase.calculators.calculator import Calculator
from numba import njit


class BudgetReached(BaseException):
    """The force-call budget is spent. A BaseException so search loops do not catch it."""


class HitReached(BaseException):
    """A quenched energy is within the hit window."""


@njit(cache=True)
def lj_value_and_gradient(x):
    n = x.shape[0] // 3
    energy = 0.0
    grad = np.zeros(x.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue
            inv = 1.0 / r2
            inv6 = inv * inv * inv
            inv12 = inv6 * inv6
            energy += 4.0 * (inv12 - inv6)
            coef = -24.0 * inv * (2.0 * inv12 - inv6)
            grad[3 * i] += coef * dx
            grad[3 * i + 1] += coef * dy
            grad[3 * i + 2] += coef * dz
            grad[3 * j] -= coef * dx
            grad[3 * j + 1] -= coef * dy
            grad[3 * j + 2] -= coef * dz
    return energy, grad


def warmup():
    lj_value_and_gradient(np.zeros(6))


class LjCalculator(Calculator):
    """One charge per geometry. Stops on the budget or on a quenched hit."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, budget: int, reference: float, tol: float, fmax: float):
        super().__init__()
        self.budget = budget
        self.reference = reference
        self.tol = tol
        self.fmax = fmax
        self.calls = 0
        self.inside_s = 0.0
        self.hit_calls = None
        self.best = 1.0e300
        self.best_quenched = 1.0e300

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if self.calls >= self.budget:
            raise BudgetReached()
        positions = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64)
        started = time.perf_counter()
        energy, grad = lj_value_and_gradient(positions)
        self.inside_s += time.perf_counter() - started
        self.calls += 1
        forces = (-grad).reshape(-1, 3)
        energy = float(energy)
        self.results = {"energy": energy, "forces": forces}
        if energy < self.best:
            self.best = energy
        peak = float(np.max(np.abs(forces))) if forces.size else 0.0
        if peak <= self.fmax and energy < self.best_quenched:
            self.best_quenched = energy
        if self.hit_calls is None and peak <= self.fmax and energy <= self.reference + self.tol:
            self.hit_calls = self.calls
            raise HitReached()
        if self.calls % 20000 == 0:
            print(
                f"progress calls={self.calls} best={self.best:.6f}",
                flush=True,
            )
