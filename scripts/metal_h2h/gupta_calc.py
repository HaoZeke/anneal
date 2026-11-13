# Copyright (c) anneal developers
"""Cleri–Rosato Gupta calculator, the same sums as ``anneal_core::potentials::Gupta``.

Zinc is Doye, Phys. Rev. B 68, 195418 (2003), Table 1, with lengths in
units of ``r0``. Copper, gold and the copper–gold cross term are the
Cleri–Rosato parameters that reproduce the Cambridge ``CuAu_38`` archive.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

# (A, xi, p, q, r0)
ZINC = (0.1477, 0.8900, 9.689, 4.602, 1.0)
COPPER = (0.0855, 1.224, 10.960, 2.278, 2.556)
GOLD = (0.2061, 1.790, 10.229, 4.036, 2.884)
COPPER_GOLD = (0.1539, 1.5605, 11.05, 3.0475, 2.556)

PAIRS = {
    (30, 30): ZINC,
    (29, 29): COPPER,
    (79, 79): GOLD,
    (29, 79): COPPER_GOLD,
    (79, 29): COPPER_GOLD,
}


def parameters(numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair tables ``A, xi, p, q, r0`` for the upper triangle of ``numbers``."""
    count = len(numbers)
    upper = np.triu_indices(count, 1)
    left = numbers[upper[0]]
    right = numbers[upper[1]]
    table = np.empty((upper[0].size, 5))
    covered = np.zeros(upper[0].size, dtype=bool)
    for (z_left, z_right), values in PAIRS.items():
        mask = (left == z_left) & (right == z_right)
        table[mask] = values
        covered |= mask
    if not covered.all():
        raise ValueError("a pair has no Gupta parameters")
    return table[:, 0], table[:, 1], table[:, 2], table[:, 3], table[:, 4]


def evaluate(numbers: np.ndarray, xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy and Cartesian gradient of a Gupta cluster."""
    numbers = np.asarray(numbers, dtype=int)
    xyz = np.asarray(xyz, dtype=float)
    count = len(numbers)
    if xyz.shape != (count, 3):
        raise ValueError("coordinates are three columns per atom")
    upper = np.triu_indices(count, 1)
    strength, xi, power, decay, r0 = parameters(numbers)
    difference = xyz[upper[0]] - xyz[upper[1]]
    distance = np.linalg.norm(difference, axis=1)
    if np.any(distance <= 0.0):
        return float("inf"), np.zeros_like(xyz)
    reduced = distance / r0 - 1.0
    phi = 2.0 * strength * np.exp(-power * reduced)
    psi = xi * xi * np.exp(-2.0 * decay * reduced)
    density = np.zeros(count)
    np.add.at(density, upper[0], psi)
    np.add.at(density, upper[1], psi)
    energy = float(phi.sum() - np.sqrt(density).sum())
    dphi = -(power / r0) * phi
    dpsi = -(2.0 * decay / r0) * psi
    embed = -0.5 * (1.0 / np.sqrt(density[upper[0]]) + 1.0 / np.sqrt(density[upper[1]]))
    coefficient = (dphi + embed * dpsi) / distance
    gradient = np.zeros_like(xyz)
    np.add.at(gradient, upper[0], coefficient[:, None] * difference)
    np.add.at(gradient, upper[1], -coefficient[:, None] * difference)
    return energy, gradient


class GuptaCalculator(Calculator):
    """ASE calculator for the Gupta clusters in this benchmark."""

    implemented_properties = ("energy", "forces")

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, gradient = evaluate(self.atoms.get_atomic_numbers(), self.atoms.get_positions())
        self.results = {"energy": energy, "forces": -gradient}
