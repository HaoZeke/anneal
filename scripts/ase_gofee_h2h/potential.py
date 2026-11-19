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


class CapReached(BaseException):
    """The wall cap for this seed is spent."""


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
    gupta_value_and_gradient(np.array([0.0, 0.0, 0.0, 2.8, 0.0, 0.0]), np.zeros(2, dtype=np.int64))


# Cleri–Rosato Cu–Au rows, the same numbers as `src/potentials/gupta.rs`.
# Order is Cu–Cu, Cu–Au, Au–Cu, Au–Au. Each row is A, xi, p, q, r0.
_GUPTA = np.array(
    [
        [0.0855, 1.224, 10.960, 2.278, 2.556],
        [0.1539, 1.5605, 11.05, 3.0475, 2.556],
        [0.1539, 1.5605, 11.05, 3.0475, 2.556],
        [0.2061, 1.790, 10.229, 4.036, 2.884],
    ],
    dtype=np.float64,
)


@njit(cache=True)
def gupta_value_and_gradient(x, slot):
    """All-pairs Gupta energy and gradient. `slot` is 0 for Cu and 1 for Au."""
    n = x.shape[0] // 3
    energy = 0.0
    rho = np.zeros(n)
    # i, j, dx, dy, dz, r, dphi_dr, dpsi_dr
    pairs_i = np.empty(n * (n - 1) // 2, dtype=np.int64)
    pairs_j = np.empty(n * (n - 1) // 2, dtype=np.int64)
    stored = np.empty((n * (n - 1) // 2, 6))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                return np.inf, np.zeros(x.shape[0])
            row = slot[i] * 2 + slot[j]
            a = _GUPTA[row, 0]
            xi = _GUPTA[row, 1]
            p = _GUPTA[row, 2]
            q = _GUPTA[row, 3]
            r0 = _GUPTA[row, 4]
            r = np.sqrt(r2)
            u = r / r0 - 1.0
            phi = 2.0 * a * np.exp(-p * u)
            psi = xi * xi * np.exp(-2.0 * q * u)
            energy += phi
            rho[i] += psi
            rho[j] += psi
            pairs_i[count] = i
            pairs_j[count] = j
            stored[count, 0] = dx
            stored[count, 1] = dy
            stored[count, 2] = dz
            stored[count, 3] = r
            stored[count, 4] = -(p / r0) * phi
            stored[count, 5] = -(2.0 * q / r0) * psi
            count += 1
    for i in range(n):
        energy -= np.sqrt(rho[i])
    grad = np.zeros(x.shape[0])
    for index in range(count):
        i = pairs_i[index]
        j = pairs_j[index]
        embed = -0.5 * (1.0 / np.sqrt(rho[i]) + 1.0 / np.sqrt(rho[j]))
        coef = (stored[index, 4] + embed * stored[index, 5]) / stored[index, 3]
        grad[3 * i] += coef * stored[index, 0]
        grad[3 * i + 1] += coef * stored[index, 1]
        grad[3 * i + 2] += coef * stored[index, 2]
        grad[3 * j] -= coef * stored[index, 0]
        grad[3 * j + 1] -= coef * stored[index, 1]
        grad[3 * j + 2] -= coef * stored[index, 2]
    return energy, grad


class LjCalculator(Calculator):
    """One charge per geometry. Stops on the budget or on a quenched hit."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        budget: int,
        reference: float,
        tol: float,
        fmax: float,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        deadline: float | None = None,
    ):
        super().__init__()
        self.budget = budget
        self.reference = reference
        self.tol = tol
        self.fmax = fmax
        self.sigma = sigma
        self.epsilon = epsilon
        self.deadline = deadline
        self.evaluate = None
        self.calls = 0
        self.inside_s = 0.0
        self.hit_calls = None
        self.best = 1.0e300
        self.best_quenched = 1.0e300

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise CapReached()
        if self.calls >= self.budget:
            raise BudgetReached()
        positions = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64)
        started = time.perf_counter()
        if self.evaluate is None:
            reduced_energy, grad = lj_value_and_gradient(positions / self.sigma)
            forces = (-grad * self.epsilon / self.sigma).reshape(-1, 3)
            energy = float(reduced_energy) * self.epsilon
        else:
            reduced_energy, grad = self.evaluate(positions)
            forces = (-grad).reshape(-1, 3)
            energy = float(reduced_energy)
        self.inside_s += time.perf_counter() - started
        self.calls += 1
        reduced_energy = float(reduced_energy)
        self.results = {"energy": energy, "forces": forces}
        if reduced_energy < self.best:
            self.best = reduced_energy
        peak = float(np.max(np.abs(forces))) if forces.size else 0.0
        if peak <= self.fmax and reduced_energy < self.best_quenched:
            self.best_quenched = reduced_energy
        if (
            self.hit_calls is None
            and peak <= self.fmax
            and reduced_energy <= self.reference + self.tol
        ):
            self.hit_calls = self.calls
            raise HitReached()
        if self.calls % 20000 == 0:
            print(
                f"progress calls={self.calls} best={self.best:.6f}",
                flush=True,
            )
