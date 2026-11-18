#!/usr/bin/env python3
"""Regenerate the DFT-D2 references in CppCore/tests/data/d2/.

The reference is Grimme's dftd3 program, V3.1
(https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz,
sha256 d97cf9758f61aa81fd85425448fbf4a6e8ce07c12e9236739831a3af32880f59),
run in its D2 mode, `dftd3 coord -func b-lyp -old -grad` and with `-pbc` on a
POSCAR for the cells. dftd3 prints the energy with eight decimals only, so
the build used here has the `.EDISP` write at the end of the main program
uncommented, written as `(ES25.17)`; nothing else changes:

    712,714c712,714
    < !      open(unit=1,file='.EDISP')
    < !      write(1,*) disp
    < !      close(1)
    ---
    >       open(unit=1,file='.EDISP')
    >       write(1,'(ES25.17)') disp
    >       close(1)

dftd3 reads numbers with its own READAA, which keeps ten decimals and drops
the rest, so every length is first rounded to ten decimals in bohr. Those
bohr values are what dftd3 sees: the TM coord file is in bohr, and the POSCAR
holds them with the scale factor 0.52917726, which dftd3 divides by its own
0.52917726 to get exactly 1. The reference files hold the same lengths times
rgpot's BOHR_TO_ANGSTROM, which rgpot divides back.

Usage: regen_d2_references.py --dftd3 PATH/TO/dftd3 [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path

BOHR_TO_ANGSTROM = 0.529177210903  # rgpot/units.hpp
DFTD3_AUTOANG = 0.52917726
SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O"}

WATER_DIMER = [  # S22 water dimer, Angstrom
    (8, -1.551007, -0.114520, 0.000000),
    (1, -1.934259, 0.762503, 0.000000),
    (1, -0.599677, 0.040712, 0.000000),
    (8, 1.350625, 0.111469, 0.000000),
    (1, 1.680398, -0.373741, -0.758561),
    (1, 1.680398, -0.373741, 0.758561),
]


def benzene():
    atoms = []
    for k in range(6):
        a = math.pi / 3.0 * k
        atoms.append((6, 1.397 * math.cos(a), 1.397 * math.sin(a), 0.0))
    for k in range(6):
        a = math.pi / 3.0 * k
        atoms.append((1, 2.481 * math.cos(a), 2.481 * math.sin(a), 0.0))
    return atoms


def rotation(rng):
    q = [rng.gauss(0.0, 1.0) for _ in range(4)]
    n = math.sqrt(sum(v * v for v in q))
    w, x, y, z = (v / n for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def water_box(cell, seed):
    """Eight waters on a 2x2x2 grid of the cell, randomly turned, O first."""
    rng = random.Random(seed)
    local = [(0.0, 0.0, 0.0), (0.7572, 0.5865, 0.0), (-0.7572, 0.5865, 0.0)]
    oxygens, hydrogens = [], []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = [(i + 0.25 + 0.1 * rng.random()) / 2.0,
                     (j + 0.25 + 0.1 * rng.random()) / 2.0,
                     (k + 0.25 + 0.1 * rng.random()) / 2.0]
                centre = [sum(s[r] * cell[r][a] for r in range(3)) for a in range(3)]
                rot = rotation(rng)
                for idx, p in enumerate(local):
                    x = [centre[a] + sum(rot[a][b] * p[b] for b in range(3)) for a in range(3)]
                    (oxygens if idx == 0 else hydrogens).append(x)
    return [(8, *x) for x in oxygens] + [(1, *x) for x in hydrogens]


def wrap_into(cell, atoms):
    """Wrapped to fractional [0, 1), as dftd3's image range assumes."""
    import numpy as np

    h = np.array(cell)
    out = []
    for z, *x in atoms:
        s = np.linalg.solve(h.T, np.array(x))
        s -= np.floor(s)
        out.append((z, *(s @ h)))
    return out


CUBIC = [[6.21, 0.0, 0.0], [0.0, 6.21, 0.0], [0.0, 0.0, 6.21]]
SHEARED = [[6.21, 0.0, 0.0], [1.35, 6.02, 0.0], [-0.92, 0.74, 6.33]]

SYSTEMS = {
    "water_dimer": (WATER_DIMER, None),
    "benzene": (benzene(), None),
    "water_box_cubic": (wrap_into(CUBIC, water_box(CUBIC, 2006)), CUBIC),
    "water_box_sheared": (wrap_into(SHEARED, water_box(SHEARED, 1787)), SHEARED),
}


def to_bohr(x):
    return round(x / BOHR_TO_ANGSTROM, 10)


def run(dftd3, atoms, cell, work):
    """dftd3 on the geometry in bohr (ten decimals); returns Ha, Ha/bohr."""
    if cell is None:
        with open(work / "coord", "w") as fh:
            fh.write("$coord\n")
            for z, *b in atoms:
                fh.write(f"{b[0]:.10f} {b[1]:.10f} {b[2]:.10f} {SYMBOL[z].lower()}\n")
            fh.write("$end\n")
        cmd = [dftd3, "coord", "-func", "b-lyp", "-old", "-grad"]
    else:
        order = []
        for z, *_ in atoms:
            if z not in order:
                order.append(z)
        counts = [sum(1 for a in atoms if a[0] == z) for z in order]
        assert [a[0] for a in atoms] == [z for z, c in zip(order, counts) for _ in range(c)]
        with open(work / "POSCAR", "w") as fh:
            fh.write(" ".join(SYMBOL[z] for z in order) + "\n")
            fh.write(f"{DFTD3_AUTOANG:.8f}\n")
            for row in cell:
                fh.write(" ".join(f"{v:.10f}" for v in row) + "\n")
            fh.write(" ".join(SYMBOL[z] for z in order) + "\n")
            fh.write(" ".join(str(c) for c in counts) + "\n")
            fh.write("Cartesian\n")
            for _, *b in atoms:
                fh.write(" ".join(f"{v:.10f}" for v in b) + "\n")
        cmd = [dftd3, "POSCAR", "-func", "b-lyp", "-old", "-pbc", "-grad"]
    done = subprocess.run(cmd, cwd=work, check=True, capture_output=True, text=True)
    if "normal termination" not in done.stdout:
        raise RuntimeError(f"dftd3 failed:\n{done.stdout}")
    energy = float((work / ".EDISP").read_text().split()[0])
    grad = [[float(v.replace("D", "E")) for v in line.split()]
            for line in (work / "dftd3_gradient").read_text().splitlines() if line.strip()]
    cellgrad = None
    if cell is not None:
        cellgrad = [[float(v.replace("D", "E")) for v in line.split()]
                    for line in (work / "dftd3_cellgradient").read_text().splitlines()
                    if line.strip()]
    return energy, grad, cellgrad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dftd3", required=True)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                             / "CppCore" / "tests" / "data" / "d2"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (atoms_ang, cell_ang) in SYSTEMS.items():
        atoms = [(z, *(to_bohr(v) for v in x)) for z, *x in atoms_ang]
        cell = [[to_bohr(v) for v in row] for row in cell_ang] if cell_ang else None
        with tempfile.TemporaryDirectory() as tmp:
            energy, grad, cellgrad = run(os.path.abspath(args.dftd3), atoms, cell, Path(tmp))
        lines = [
            f"# {name}: dftd3 V3.1 -func b-lyp -old{' -pbc' if cell else ''} -grad",
            "# atoms: Z x y z in Angstrom; energy Ha; grad Ha/bohr; cellgrad Ha/bohr,",
            "# row i is dE/d(lattice vector i) at fixed fractional coordinates.",
            f"natoms {len(atoms)}",
            "periodic " + ("1" if cell else "0"),
        ]
        box = cell if cell else [[0.0] * 3] * 3
        lines.append("box " + " ".join(f"{v * BOHR_TO_ANGSTROM:.17g}" for row in box for v in row))
        for z, *b in atoms:
            lines.append(f"atom {z} " + " ".join(f"{v * BOHR_TO_ANGSTROM:.17g}" for v in b))
        lines.append(f"energy {energy:.17g}")
        for g in grad:
            lines.append("grad " + " ".join(f"{v:.17g}" for v in g))
        if cellgrad:
            for g in cellgrad:
                lines.append("cellgrad " + " ".join(f"{v:.17g}" for v in g))
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(name, energy)


if __name__ == "__main__":
    main()
