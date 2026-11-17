# Copyright (c) anneal developers
import hashlib
import math
import sys
from pathlib import Path

A = 2510400.0
C = 2552.24
sig6 = A / C
sigma = sig6 ** (1 / 6)
eps = C / (4 * sig6)
ea, sa = 0.6485, 3.15365
c6 = 4 * ea * sa ** 6
c12 = 4 * ea * sa ** 12
g6, g12 = 2552.24, 2510.4e3
print(f"Jorgensen A = {A:.4f} kJ A^12/mol, C = {C:.6f} kJ A^6/mol")
print(f"exact epsilon = {eps:.7f} kJ/mol, sigma = {sigma:.6f} A")
print(f"anneal epsilon = {ea}, sigma = {sa}: C6 = {c6:.4f}, C12 = {c12:.1f}")
print(
    f"GMIN C6 = {g6}, C12 = {g12:.4e}: equal to Jorgensen C and A: "
    f"{abs(g6 - C) < 1e-9 and abs(g12 - A) < 1e-3}"
)
print(
    f"relative epsilon rounding (anneal/exact - 1) = {ea / eps - 1:.3e}; "
    f"sigma (anneal/exact - 1) = {sa / sigma - 1:.3e}"
)


def rounded_hexamer(path: Path) -> None:
    """Reproduce the historical rigid-TIP4P energy at a CCD geometry."""
    import numpy as np

    text = path.read_text()
    lines = text.splitlines()
    atoms = int(lines[0])
    symbols = [line.split()[0] for line in lines[2 : 2 + atoms]]
    xyz = np.array([[float(value) for value in line.split()[1:4]] for line in lines[2 : 2 + atoms]])
    if atoms != 18 or symbols != ["O", "H", "H"] * 6:
        raise SystemExit(f"{path} is not an O-H-H ordered water hexamer")

    half = math.radians(104.52 / 2.0)
    masses = np.array([15.9994, 1.008, 1.008])
    com_z = 2.0 * masses[1] * 0.9572 * math.cos(half) / masses.sum()
    body = np.array(
        [
            [0.0, 0.0, -com_z],
            [0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half) - com_z],
            [-0.9572 * math.sin(half), 0.0, 0.9572 * math.cos(half) - com_z],
            [0.0, 0.0, 0.15 - com_z],
        ]
    )
    placed = []
    for molecule in range(6):
        lab = xyz[3 * molecule : 3 * molecule + 3]
        centre = (lab * masses[:, None]).sum(axis=0) / masses.sum()
        left, _, right_t = np.linalg.svd(body[:3].T @ (lab - centre))
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        placed.append(body @ rotation.T + centre)

    charges = (0.0, 0.52, 0.52, -1.04)
    coulomb = 1389.35458
    energy = 0.0
    for first in range(6):
        for second in range(first + 1, 6):
            distance = np.linalg.norm(placed[first][0] - placed[second][0])
            energy += c12 / distance**12 - c6 / distance**6
            for site_a in range(1, 4):
                for site_b in range(1, 4):
                    distance = np.linalg.norm(placed[first][site_a] - placed[second][site_b])
                    energy += coulomb * charges[site_a] * charges[site_b] / distance

    print("source_revision = e51f38dd1df91f5f153c396349eda1a5009ef3ae")
    print(f"structure = {path}")
    print(f"structure_sha256 = {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"historical Coulomb prefactor = {coulomb}")
    print(f"energy = {energy:.10f} kJ/mol")
    print(f"reported energy = {energy:.4f} kJ/mol")


if "--hexamer" in sys.argv:
    index = sys.argv.index("--hexamer")
    if index + 1 >= len(sys.argv):
        raise SystemExit("usage: tip4p_constants.py [--hexamer TIP4P-6.xyz]")
    rounded_hexamer(Path(sys.argv[index + 1]))
