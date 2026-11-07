#!/usr/bin/env python3
"""Periodic D4 against dftd4 4.3.1 on one cell that library can finish."""

import os
import sys
import tempfile

import numpy as np
from dftd4.interface import DampingParam, DispersionModel

BOHR = 0.529177210903
HARTREE = 27.211386245988


def main():
    binary = sys.argv[1]
    rng = np.random.default_rng(1)
    n = 16
    side = 10.0
    pos = rng.uniform(0.4, side - 0.4, size=(n, 3))
    z = np.array([8 if i % 2 == 0 else 1 for i in range(n)])
    lattice = np.diag([side, side, side])
    model = DispersionModel(
        z, pos / BOHR, charge=0.0, lattice=lattice / BOHR, periodic=np.array([True, True, True])
    )
    ref = model.get_dispersion(DampingParam(method="pbe", atm=True), grad=True)
    ref_e = float(ref["energy"])
    ref_g = np.array(ref["gradient"], dtype=float).reshape(-1)

    geom = os.path.join(tempfile.gettempdir(), "d4_periodic_geom.txt")
    with open(geom, "w", encoding="utf-8") as handle:
        handle.write(f"{n} {side:.16g}\n")
        for zi, point in zip(z, pos):
            handle.write(f"{int(zi)} {point[0]:.16g} {point[1]:.16g} {point[2]:.16g}\n")

    import subprocess

    proc = subprocess.run([binary, geom], check=True, capture_output=True, text=True)
    energy = None
    forces = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts[0] == "energy_eV":
            energy = float(parts[1]) / HARTREE
        elif parts[0] == "force_eV_A":
            # eV/A to Ha/Bohr, and force to gradient.
            scale = BOHR / HARTREE
            forces.extend([-float(parts[1]) * scale, -float(parts[2]) * scale, -float(parts[3]) * scale])
    forces = np.array(forces)
    e_err = abs(energy - ref_e)
    g_err = float(np.max(np.abs(forces - ref_g)))
    print(f"periodic D4 n={n} side_A={side}")
    print(f"dftd4_energy_Ha {ref_e:.17g}")
    print(f"rgpot_energy_Ha {energy:.17g}")
    print(f"max_abs_energy_Ha {e_err:.17g}")
    print(f"max_abs_gradient_Ha_Bohr {g_err:.17g}")
    if not np.isfinite(e_err) or not np.isfinite(g_err):
        raise SystemExit("non-finite periodic D4 comparison")


if __name__ == "__main__":
    main()
