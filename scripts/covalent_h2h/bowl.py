#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Relax hand-built C20 candidates on the Tersoff library.

The corannulene skeleton is a pentagon, five spokes and a rim of ten,
puckered so the quench can curve it. The dodecahedron is the Ih fullerene.
The bowl is the C20 reference: no unbiased run in reference.csv reaches it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ase_bh import Covalent  # noqa: E402

BOND = 1.42


def corannulene() -> np.ndarray:
    angles = np.arange(5) * 2.0 * np.pi / 5.0
    hub_radius = BOND / (2.0 * np.sin(np.pi / 5.0))
    hub = np.c_[hub_radius * np.cos(angles), hub_radius * np.sin(angles), np.zeros(5)]
    spoke_radius = hub_radius + BOND
    spokes = np.c_[spoke_radius * np.cos(angles), spoke_radius * np.sin(angles), np.full(5, -0.5)]
    rim = []
    for angle in angles + np.pi / 5.0:
        for side in (-0.2, 0.2):
            rim.append([3.5 * np.cos(angle + side), 3.5 * np.sin(angle + side), -1.0])
    return np.r_[hub, spokes, np.array(rim)]


def dodecahedron() -> np.ndarray:
    phi = 0.5 * (1.0 + 5.0**0.5)
    points = [list(p) for p in itertools.product((-1.0, 1.0), repeat=3)]
    for a, b in itertools.product((-1.0, 1.0), repeat=2):
        points += [[0.0, a / phi, b * phi], [a / phi, b * phi, 0.0], [a * phi, 0.0, b / phi]]
    return np.array(points) * 1.45 / (2.0 / phi)


def relax(potential: Covalent, xyz: np.ndarray) -> tuple[float, np.ndarray]:
    def value(flat):
        energy, gradient = potential(flat.reshape(-1, 3))
        return energy, gradient.ravel()

    result = minimize(
        value, xyz.ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": 20000, "gtol": 1e-8, "ftol": 1e-15},
    )
    return float(result.fun), result.x.reshape(-1, 3)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv[1:])
    potential = Covalent(args.library, "tc20")
    for name, build in (("corannulene", corannulene), ("dodecahedron", dodecahedron)):
        energy, xyz = relax(potential, build())
        print(f"{name} {energy:.8f}")
        if name == "corannulene" and args.out:
            np.savetxt(args.out, xyz, fmt="%.10f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
