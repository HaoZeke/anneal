#!/usr/bin/env python3
"""Periodic D4 against dftd4 4.3.1.

Several cells must agree to 1e-8 Ha and 1e-8 Ha/Bohr. The check records
the binary's build stamp and the sha256 of that binary.
"""

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
from dftd4.interface import DampingParam, DispersionModel

BOHR = 0.529177210903
HARTREE = 27.211386245988
ENERGY_TOL = 1e-8
GRADIENT_TOL = 1e-8


def dftd4_version():
    import dftd4

    for name in ("__version__", "version"):
        if hasattr(dftd4, name):
            return str(getattr(dftd4, name))
    return "dftd4-python"  # pragma: no cover


def one_cell(binary, n, lengths, seed):
    rng = np.random.default_rng(seed)
    ax, ay, az = lengths
    pos = np.column_stack(
        (
            rng.uniform(0.4, ax - 0.4, size=n),
            rng.uniform(0.4, ay - 0.4, size=n),
            rng.uniform(0.4, az - 0.4, size=n),
        )
    )
    z = np.array([8 if i % 2 == 0 else 1 for i in range(n)])
    lattice = np.diag([ax, ay, az])
    model = DispersionModel(
        z, pos / BOHR, charge=0.0, lattice=lattice / BOHR, periodic=np.array([True, True, True])
    )
    ref = model.get_dispersion(DampingParam(method="pbe", atm=True), grad=True)
    ref_e = float(ref["energy"])
    ref_g = np.array(ref["gradient"], dtype=float).reshape(-1)

    geom = os.path.join(tempfile.gettempdir(), f"d4_periodic_{n}_{seed}.txt")
    with open(geom, "w", encoding="utf-8") as handle:
        handle.write(f"{n} {ax:.16g} {ay:.16g} {az:.16g}\n")
        for zi, point in zip(z, pos):
            handle.write(f"{int(zi)} {point[0]:.16g} {point[1]:.16g} {point[2]:.16g}\n")

    proc = subprocess.run([binary, geom], check=True, capture_output=True, text=True)
    provenance = ""
    energy = None
    forces = []
    for line in proc.stdout.splitlines():
        if line.startswith("provenance "):
            provenance = line
            continue
        parts = line.split()
        if not parts:
            continue  # pragma: no cover
        if parts[0] == "energy_eV":
            energy = float(parts[1]) / HARTREE
        elif parts[0] == "force_eV_A":
            scale = BOHR / HARTREE
            forces.extend([-float(parts[1]) * scale, -float(parts[2]) * scale, -float(parts[3]) * scale])
    forces = np.array(forces)
    e_err = abs(energy - ref_e)
    g_err = float(np.max(np.abs(forces - ref_g)))
    print(f"periodic D4 n={n} cell_A={ax:.16g} {ay:.16g} {az:.16g} seed={seed}")
    print(provenance)
    print(f"dftd4_version {dftd4_version()}")
    print(f"dftd4_energy_Ha {ref_e:.17g}")
    print(f"rgpot_energy_Ha {energy:.17g}")
    print(f"max_abs_energy_Ha {e_err:.17g}")
    print(f"max_abs_gradient_Ha_Bohr {g_err:.17g}")
    if not np.isfinite(e_err) or not np.isfinite(g_err):
        raise SystemExit("non-finite periodic D4 comparison")  # pragma: no cover
    if e_err > ENERGY_TOL or g_err > GRADIENT_TOL:
        raise SystemExit(  # pragma: no cover
            f"periodic D4 outside 1e-8: energy {e_err:.3e} gradient {g_err:.3e}"
        )


def main():
    binary = sys.argv[1]
    digest = hashlib.sha256(open(binary, "rb").read()).hexdigest()
    print(f"check_binary {binary}")
    print(f"check_binary_sha256 {digest}")
    cells = (
        (16, (10.0, 10.0, 10.0), 1),
        (8, (8.0, 7.5, 9.0), 2),
        (24, (12.0, 12.0, 12.0), 3),
        (32, (14.0, 11.0, 13.0), 4),
    )
    for n, lengths, seed in cells:
        one_cell(binary, n, lengths, seed)
    print("periodic D4 tolerances 1e-8 Ha and 1e-8 Ha/Bohr passed")


if __name__ == "__main__":
    main()
