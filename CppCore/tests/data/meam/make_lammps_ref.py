#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the reference runs for the MEAM tests.

The runs use the Large-scale Atomic/Molecular Massively Parallel
Simulator (LAMMPS) with the MEAM package, pair_style meam, on potential
files copied unchanged from its potentials/ directory:

  library.meam    element table of the 1NN MEAM files.
  Ni.meam         parameter file for Ni, Baskes, Phys. Rev. B 46, 2727 (1992).
  SiC.meam        parameter file for the Si-C pair.
  library_2nn.meam, MoCoNiVFeAlCr_2nn.meam
                  second-neighbour (Lee-Baskes 2NN) set for Mo Co Ni V Fe Al
                  Cr, Wang et al., npj Comput. Mater. 9, 60 (2023).
  AlSiMgCuFe.meam 2NN parameter file for the AlS SiS MgS CuS FeS entries of
                  library.meam, Jelinek et al., Phys. Rev. B 85, 245102 (2012).

and on parameter files written for this repository (opts_*.meam), which
switch on the options the LAMMPS files do not reach: ialloy 0, 1 and 2,
mixture_ref_t, bkgd_dyn, emb_lin_neg, erose_form 1 and 2, the ZBL blend,
G(gamma) for ibar 1, -5, 0 and 3, and the L12, B1, B2, C11, dimer, CH4,
linear, zig-zag, triangle and diamond reference pairs.

Each case is a jittered crystal, cubic or sheared into a triclinic cell.
Positions are written as LAMMPS stores them after read_data. The virial is
the LAMMPS pressure virial times the volume, in eV, so it is the sum of
r_i . f_i; the stress is minus that over the volume, in eV/Angstrom^3.
Components are in the order xx yy zz yz xz xy.

Usage: make_lammps_ref.py [output directory] [case name ...].
It needs the lammps Python module built with the MEAM and MANYBODY
packages.
"""

import pathlib
import sys
import tempfile

import lammps
import numpy as np

HERE = pathlib.Path(__file__).parent
ZNUM = {"C": 6, "AlS": 13, "SiS": 14, "MgS": 12, "Si": 14, "FeS": 26, "CuS": 29,
        "Fe": 26, "Ni": 28, "Cu": 29, "WL": 74, "Mo": 42, "Co": 27, "V": 23,
        "Al": 13, "Cr": 24}

FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
BCC = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
ZINCBLENDE = np.concatenate([FCC, FCC + 0.25])

# name, library, library elements (type order), parameter file or None,
# basis, lattice constant, cells, shear, jitter, species rule, seed
# The species rule is "random", "zincblende" (types 1 and 2 by sublattice) or
# "only:N" (every atom of type N).
HEA = ("Mo", "Co", "Ni", "V", "Fe", "Al", "Cr")
CASES = (
    ("ni_cubic", "library.meam", ("Ni",), "Ni.meam", FCC, 3.52, 3, False, 0.10,
     "random", 31),
    ("ni_sheared_small", "library.meam", ("Ni",), "Ni.meam", FCC, 3.52, 2, True,
     0.12, "random", 32),
    ("fe_bcc", "library.meam", ("Fe",), None, BCC, 2.87, 4, False, 0.08,
     "random", 33),
    ("sic_zincblende", "library.meam", ("Si", "C"), "SiC.meam", ZINCBLENDE, 4.36,
     3, True, 0.08, "zincblende", 34),
    ("fe_2nn", "library_2nn.meam", HEA, "MoCoNiVFeAlCr_2nn.meam", BCC, 2.87, 4, False,
     0.08, "only:5", 41),
    ("ni_2nn_sheared", "library_2nn.meam", HEA, "MoCoNiVFeAlCr_2nn.meam", FCC, 3.52, 3,
     True, 0.09, "only:3", 42),
    ("hea_2nn", "library_2nn.meam", HEA, "MoCoNiVFeAlCr_2nn.meam", BCC, 2.9, 4, True,
     0.07, "random", 43),
    ("alsimgcufe_2nn", "library.meam", ("AlS", "SiS", "MgS", "CuS", "FeS"),
     "AlSiMgCuFe.meam", FCC, 4.0, 3, True, 0.08, "random", 44),
    ("alloy_opts_a", "library.meam", ("Ni", "AlS", "Cu", "Fe", "WL"), "opts_a.meam",
     FCC, 3.6, 3, True, 0.08, "random", 35),
    ("alloy_opts_b", "library.meam", ("Ni", "AlS", "Cu", "Fe", "WL"), "opts_b.meam",
     FCC, 3.6, 3, True, 0.08, "random", 36),
    ("alloy_opts_c", "library.meam", ("Ni", "AlS", "Cu", "Fe", "WL"), "opts_c.meam",
     FCC, 3.5, 3, True, 0.10, "random", 37),
)


def crystal(basis, lattice, cells, sheared, jitter, rule, seed, ntypes):
    grid = np.array(np.meshgrid(*[range(cells)] * 3, indexing="ij")).reshape(3, -1).T
    frac = (grid[:, None, :] + basis[None]).reshape(-1, 3) / cells
    box = np.diag([cells * lattice] * 3)
    if sheared:
        box[1, 0], box[2, 0], box[2, 1] = 0.9, 0.5, -0.7
    rng = np.random.default_rng(seed)
    pos = frac @ box + rng.uniform(-jitter, jitter, frac.shape)
    pos = np.mod(pos @ np.linalg.inv(box), 1.0) @ box
    if rule.startswith("only:"):
        types = np.full(len(pos), int(rule[5:]))
    elif rule == "zincblende":
        types = np.tile(np.repeat([1, 2], len(basis) // 2), cells ** 3)
    else:
        types = rng.integers(1, ntypes + 1, len(pos))
    return box, pos, types


def run_lammps(box, pos, types, library, elements, params):
    with tempfile.TemporaryDirectory() as tmp:
        data = pathlib.Path(tmp) / "case.data"
        data.write_text(
            f"lammps data\n\n{len(pos)} atoms\n{len(elements)} atom types\n"
            f"0 {box[0, 0]:.17g} xlo xhi\n0 {box[1, 1]:.17g} ylo yhi\n"
            f"0 {box[2, 2]:.17g} zlo zhi\n"
            f"{box[1, 0]:.17g} {box[2, 0]:.17g} {box[2, 1]:.17g} xy xz yz\n\n"
            "Masses\n\n"
            + "".join(f"{k + 1} 1.0\n" for k in range(len(elements)))
            + "\nAtoms # atomic\n\n"
            + "".join(f"{i + 1} {t} {x:.17g} {y:.17g} {z:.17g}\n"
                      for i, (t, (x, y, z)) in enumerate(zip(types, pos)))
        )
        names = " ".join(elements)
        coeff = f"* * {HERE / library} {names} {HERE / params if params else 'NULL'} {names}"
        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.commands_list([
            "units metal", "atom_style atomic", "boundary p p p",
            "atom_modify map array", f"read_data {data}", "pair_style meam",
            f"pair_coeff {coeff}", "neighbor 1.0 bin",
            "compute vir all pressure NULL virial", "thermo_style custom pe",
            "run 0",
        ])
    nat = len(pos)
    x = np.array(lmp.gather_atoms("x", 1, 3)[: 3 * nat]).reshape(nat, 3)
    f = np.array(lmp.gather_atoms("f", 1, 3)[: 3 * nat]).reshape(nat, 3)
    t = np.array(lmp.gather_atoms("type", 0, 1)[:nat])
    p = lmp.extract_compute("vir", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR)
    volume = box[0, 0] * box[1, 1] * box[2, 2]
    # LAMMPS order is xx yy zz xy xz yz; the file order is xx yy zz yz xz xy.
    virial = np.array([p[0], p[1], p[2], p[5], p[4], p[3]]) * volume / lmp.extract_global("nktv2p")
    result = (lmp.get_thermo("pe"), x, f, t, virial, volume, lmp.version())
    lmp.close()
    print("largest remap on read_data", np.abs(x - pos).max())
    return result


def main():
    args = sys.argv[1:]
    outdir = pathlib.Path(args[0]) if args else HERE
    only = set(args[1:])
    for (name, library, elements, params, basis, lattice, cells, sheared, jitter,
         rule, seed) in CASES:
        if only and name not in only:
            continue
        box, pos, types = crystal(basis, lattice, cells, sheared, jitter, rule, seed,
                                  len(elements))
        energy, x, f, t, virial, volume, version = run_lammps(box, pos, types, library,
                                                              elements, params)
        stress = -virial / volume
        rows = "".join(
            f"{ZNUM[elements[k - 1]]} " + " ".join(f"{v:.17g}" for v in (*xi, *fi)) + "\n"
            for k, xi, fi in zip(t, x, f))
        (outdir / f"{name}.ref").write_text(
            f"# LAMMPS {version} pair_style meam, {library} {params or 'NULL'}\n"
            f"elements {' '.join(elements)}\n"
            f"box {' '.join(f'{v:.17g}' for v in box.ravel())}\n"
            f"energy {energy:.17g}\n"
            f"virial {' '.join(f'{v:.17g}' for v in virial)}\n"
            f"stress {' '.join(f'{v:.17g}' for v in stress)}\n"
            f"atoms {len(x)}\n{rows}"
        )
        print(name, f"energy {energy:.12f} stress", " ".join(f"{v:.6e}" for v in stress))


if __name__ == "__main__":
    main()
