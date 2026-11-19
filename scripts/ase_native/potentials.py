# Copyright (c) anneal developers
"""Independent eV/angstrom evaluators for the agreement test.

These are not the anneal kernels. The agreement test requires the two
implementations to match after the unit map. Lennard-Jones is the
untruncated 12-6 pair with the given epsilon and sigma. Morse is the
Doye-Wales pair with the given epsilon, r0 and rho, untruncated. Gold is
the Cleri-Rosato sum used by Gupta::copper_gold, already in eV and
angstroms.
"""

from __future__ import annotations

import numpy as np

# Darby gold row, the constants in src/potentials/gupta.rs.
GOLD = {"a": 0.2061, "xi": 1.790, "p": 10.229, "q": 4.036, "r0": 2.884}


def lennard_jones(positions: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0):
    """Untruncated Lennard-Jones energy and Cartesian gradient.

    ``positions`` is in angstroms, shape (n, 3). The reduced separation is
    r / sigma. Energy is in eV and the gradient is in eV/angstrom.
    """
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    energy = 0.0
    gradient = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            r2 = float(delta @ delta)
            if r2 <= 0.0:
                continue
            reduced = r2 / (sigma * sigma)
            inv = 1.0 / reduced
            inv6 = inv * inv * inv
            inv12 = inv6 * inv6
            energy += 4.0 * epsilon * (inv12 - inv6)
            # dV/dr_reduced * dr_reduced/dr = (dV/dr_reduced) / sigma,
            # and the anneal coefficient is (dV/dr) / r times the
            # separation. In reduced units that coefficient is
            # -24 * inv_r2 * (2 inv12 - inv6), with inv_r2 = 1/r_reduced^2.
            # Chain rule through r = sigma * r_reduced multiplies the
            # physical gradient by epsilon / sigma.
            coef = -24.0 * epsilon * inv * (2.0 * inv12 - inv6) / (sigma * sigma)
            gradient[i] += coef * delta
            gradient[j] -= coef * delta
    return float(energy), gradient


def morse(positions: np.ndarray, rho: float, epsilon: float = 1.0, r0: float = 1.0):
    """Untruncated Morse energy and Cartesian gradient, in eV and eV/angstrom."""
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    energy = 0.0
    gradient = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            r2 = float(delta @ delta)
            if r2 <= 0.0:
                continue
            distance = math_sqrt(r2)
            reduced = distance / r0
            amplitude = np.exp(rho * (1.0 - reduced))
            energy += float(epsilon * amplitude * (amplitude - 2.0))
            # dV/dr = (epsilon / r0) * 2 rho (a - a^2), then / r for the
            # separation coefficient. a - a^2 = a (1 - a).
            coef = (
                epsilon * 2.0 * rho * (amplitude - amplitude * amplitude) / (r0 * distance)
            )
            gradient[i] += coef * delta
            gradient[j] -= coef * delta
    return float(energy), gradient


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def gupta_gold(positions: np.ndarray):
    """Cleri-Rosato gold cluster. Energy in eV, gradient in eV/angstrom."""
    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    a = GOLD["a"]
    xi = GOLD["xi"]
    p = GOLD["p"]
    q = GOLD["q"]
    r0 = GOLD["r0"]
    rho = np.zeros(n, dtype=np.float64)
    pack = []
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            delta = pos[i] - pos[j]
            r2 = float(delta @ delta)
            if r2 <= 0.0:
                return float("inf"), np.zeros_like(pos)
            distance = float(np.sqrt(r2))
            u = distance / r0 - 1.0
            phi = 2.0 * a * np.exp(-p * u)
            psi = xi * xi * np.exp(-2.0 * q * u)
            energy += float(phi)
            rho[i] += psi
            rho[j] += psi
            pack.append((i, j, delta, distance, -(p / r0) * phi, -(2.0 * q / r0) * psi))
    for density in rho:
        energy -= float(np.sqrt(density))
    gradient = np.zeros_like(pos)
    inv_sqrt = 1.0 / np.sqrt(rho)
    for i, j, delta, distance, dphi, dpsi in pack:
        embed = -0.5 * (inv_sqrt[i] + inv_sqrt[j])
        coef = (dphi + embed * dpsi) / distance
        gradient[i] += coef * delta
        gradient[j] -= coef * delta
    return float(energy), gradient


def relative_error(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale


def gradient_relative_error(left: np.ndarray, right: np.ndarray) -> float:
    difference = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    if scale < 1.0e-12:
        return difference
    return difference / scale
