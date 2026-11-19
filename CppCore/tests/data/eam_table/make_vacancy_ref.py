#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the reference run for the copper vacancy cell.

The cell is the 3 x 3 x 3 fcc copper crystal of make_lammps_ref.py with
each atom jittered by up to 0.12 Angstrom, seed 23, and the first atom
removed, so 107 atoms are left around one vacancy. The run is LAMMPS
pair_style eam on Cu_u3.eam, the same file the other copper case reads.
The inputs are the ones run_lammps in make_lammps_ref.py issues: units
metal, atom_style atomic, boundary p p p, read_data of the cell,
pair_style eam, pair_coeff * * Cu_u3.eam, neighbor 1.0 bin, run 0.

Usage: make_vacancy_ref.py [output directory].
It needs the lammps Python module with the MANYBODY package.
"""

import hashlib
import pathlib
import sys

import make_lammps_ref as base

NAME = "cu_u3_vacancy"
POTFILE = "Cu_u3.eam"
CUTOFF = 4.95
LATTICE = 3.615
CELLS = 3
SEED = 23


def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else base.HERE
    box, pos, types = base.crystal(LATTICE, CELLS, False, SEED, CUTOFF, 1)
    pos = pos[1:]
    types = types[1:]
    energy, x, f, t, stress, version = base.run_lammps(
        box, pos, types, POTFILE, "eam", ("Cu",)
    )
    digest = hashlib.sha256((base.HERE / POTFILE).read_bytes()).hexdigest()
    rows = "".join(
        "29 " + " ".join(f"{v:.17g}" for v in (*xi, *fi)) + "\n"
        for xi, fi in zip(x, f)
    )
    (outdir / f"{NAME}.ref").write_text(
        f"# LAMMPS {version} pair_style eam, {POTFILE} sha256 {digest}, "
        f"vacancy at the first of 108 sites\n"
        f"box {' '.join(f'{v:.17g}' for v in box.ravel())}\n"
        f"energy {energy:.17g}\nstress {' '.join(f'{v:.17g}' for v in stress)}\n"
        f"atoms {len(x)}\n{rows}"
    )
    print(NAME, len(x), "atoms, energy", f"{energy:.12f}", "LAMMPS", version)


if __name__ == "__main__":
    main()
