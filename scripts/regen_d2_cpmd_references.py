#!/usr/bin/env python3
"""Regenerate the CPMD empirical D2 references in CppCore/tests/data/d2cpmd/.

The reference is OpenCPMD at commit 062582b7cfd832d36f88f504cd08e4ead42eb404,
the tree cpmdc builds against, run as `cpmd.x deck.inp .` with
`VDW CORRECTION ON` in &CPMD and `ALL DFT-D2` in the EMPIRICAL CORRECTION
block of &VDW, so `vdw` in `src/vdw_utils.mod.F90` evaluates Grimme's D2 over
the minimum image of every pair. CPMD prints that energy as `VDW ENERGY` with
eight decimals and never prints its forces apart from the others, so the build
used here adds an output-only write at the end of the D2 branch of `vdw`:

    OPEN(unit=77,file='.EVDW',status='replace')
    WRITE(77,'(ES25.17)') evdw
    ! one line per atom, by species: is, ia and the D2 force in Ha/bohr
    WRITE(77,'(I4,1X,I4,3(1X,ES25.17))') is_,ia_,fion(:,ia_,is_)-fsave(:,ia_,is_)

with `fsave` a copy of `fion` taken before the loop. Nothing else changes.
The script checks the full precision energy against the printed one.

The D2 term does not depend on the electrons, so the decks use a low plane
wave cutoff and stop after two wavefunction steps. The pseudopotentials are
the GTH-BLYP set of CP2K's `data/GTH_POTENTIALS`, written in CPMD's GOEDECKER
format. Lengths are in bohr in the decks; the reference files hold them
times 0.529177210859, CPMD's own bohr, which is what cpmdc divides by when it
hands rgpot's Angstrom to CPMD.

Usage: regen_d2_cpmd_references.py --cpmd PATH/TO/cpmd.x
           --gth PATH/TO/GTH_POTENTIALS [--out DIR]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

CPMD_BOHR = 0.529177210859  # 1 / fbohr in OpenCPMD's cnst.mod.F90
SYMBOL = {1: "H", 6: "C", 8: "O"}
LMAX = {1: "S", 6: "P", 8: "P"}

WATER = [(8, 0.0, 0.0, 0.0), (1, 1.81, 0.0, 0.0), (1, -0.45, 1.75, 0.0)]
METHANE = [
    (6, 0.0, 0.0, 0.0),
    (1, 1.19, 1.19, 1.19),
    (1, -1.19, -1.19, 1.19),
    (1, -1.19, 1.19, -1.19),
    (1, 1.19, -1.19, -1.19),
]


def shifted(atoms, dx, dy, dz):
    return [(z, x + dx, y + dy, w + dz) for z, x, y, w in atoms]


def two_waters(ox):
    """Water at the origin and a second one whose oxygen sits at x = ox."""
    return WATER + shifted(
        [(8, 0.0, 0.0, 0.0), (1, 1.45, 0.9, 0.4), (1, -0.55, -1.4, 0.1)],
        ox,
        0.3,
        -0.2,
    )


# name: (cell, atoms). A cell is ("cubic", a), ("ortho", a, b, c) or
# ("vectors", 3x3 rows in bohr).
SYSTEMS = {
    "cubic_water": (("cubic", 15.0), two_waters(6.1)),
    # The two O atoms 7.5 -/+ 0.02 bohr apart along x in a 15 bohr cell:
    # the minimum image of the pair flips across L/2 between the two.
    "cubic_below_half": (("cubic", 15.0), two_waters(7.48)),
    "cubic_above_half": (("cubic", 15.0), two_waters(7.52)),
    "ortho_water_methane": (
        ("ortho", 16.0, 14.0, 15.0),
        WATER + shifted(METHANE, 7.9, -6.8, 7.2),
    ),
    "triclinic_water_methane": (
        ("vectors", [[14.0, 0.0, 0.0], [2.5, 13.5, 0.0], [-1.5, 1.0, 14.5]]),
        two_waters(8.3) + shifted(METHANE, 3.0, 9.6, -6.9),
    ),
    # H-H pairs either side of CPMD's 6 R0 truncation (22.70 bohr for H-H).
    "cubic_truncation": (("cubic", 50.0), two_waters(21.9)),
}


def gth_pp(gth: str, z: int) -> str:
    """CP2K's GTH-BLYP entry for z in CPMD's GOEDECKER format."""
    lines = gth.splitlines()
    head = f"{SYMBOL[z]} GTH-BLYP-q"
    i = next(k for k, l in enumerate(lines) if l.startswith(head))
    zv = sum(int(v) for v in lines[i + 1].split())
    loc = lines[i + 2].split()
    nl = int(lines[i + 3].split()[0])
    body = [str(nl), f"  {loc[0]}", "  " + " ".join(loc[1:])]
    k = i + 4
    for _ in range(nl):
        parts = lines[k].split()
        npro = int(parts[1])
        hmat = parts[2:]
        k += 1
        for _row in range(1, npro):
            hmat += lines[k].split()
            k += 1
        body.append("  " + " ".join(parts[:2] + hmat))
    return (
        f"&ATOM\n Z  = {z}\n ZV = {zv}\n XC = 1312        .666667\n"
        " TYPE = NORMCONSERVING GOEDECKER\n&END\n"
        f"&INFO\n {lines[i].strip()} from CP2K data/GTH_POTENTIALS\n&END\n"
        "&POTENTIAL\n    GOEDECKER\n" + "\n".join("  " + b for b in body) + "\n&END\n"
    )


def by_species(atoms):
    order = []
    for z, *_ in atoms:
        if z not in order:
            order.append(z)
    return [(z, [a for a in atoms if a[0] == z]) for z in order]


def deck(cell, atoms) -> str:
    if cell[0] == "cubic":
        system = f" SYMMETRY\n  1\n CELL\n  {cell[1]!r} 1.0 1.0 0.0 0.0 0.0\n"
    elif cell[0] == "ortho":
        a, b, c = cell[1:]
        system = f" SYMMETRY\n  8\n CELL\n  {a!r} {b / a!r} {c / a!r} 0.0 0.0 0.0\n"
    else:
        rows = "\n".join("  " + " ".join(repr(v) for v in r) for r in cell[1])
        system = f" CELL VECTORS\n{rows}\n"
    out = [
        "&INFO\n rgpot D2 reference\n&END\n",
        "&CPMD\n OPTIMIZE WAVEFUNCTION\n VDW CORRECTION ON\n MAXITER\n  2\n&END\n",
        f"&SYSTEM\n{system} CUTOFF\n  20.0\n&END\n",
        "&DFT\n FUNCTIONAL BLYP\n&END\n",
        "&VDW\n EMPIRICAL CORRECTION\n  ALL DFT-D2\n END EMPIRICAL CORRECTION\n&END\n",
        "&ATOMS\n",
    ]
    for z, group in by_species(atoms):
        out.append(f"*{SYMBOL[z]}_GTH_BLYP.psp\n LMAX={LMAX[z]}\n {len(group)}\n")
        for _, x, y, w in group:
            out.append(f"  {x!r} {y!r} {w!r}\n")
    out.append("&END\n")
    return "".join(out)


def box_rows(cell):
    if cell[0] == "cubic":
        a = cell[1]
        return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    if cell[0] == "ortho":
        a, b, c = cell[1:]
        return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
    return cell[1]


def run(cpmd: Path, gth: str, name: str, cell, atoms):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for z in {a[0] for a in atoms}:
            (work / f"{SYMBOL[z]}_GTH_BLYP.psp").write_text(gth_pp(gth, z))
        (work / "deck.inp").write_text(deck(cell, atoms))
        out = subprocess.run(
            [str(cpmd), "deck.inp", "."],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        printed = re.findall(r"VDW ENERGY =\s*(\S+) A\.U\.", out)
        dump = (work / ".EVDW").read_text().split("\n")
    energy = float(dump[0])
    if not printed or abs(float(printed[-1]) - energy) > 5e-9:
        raise SystemExit(f"{name}: printed VDW ENERGY {printed} disagrees with {energy}")
    forces = [[float(v) for v in line.split()[2:5]] for line in dump[1 : 1 + len(atoms)]]
    return energy, float(printed[-1]), forces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpmd", type=Path, required=True)
    parser.add_argument("--gth", type=Path, required=True)
    here = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", type=Path, default=here / "CppCore/tests/data/d2cpmd")
    args = parser.parse_args()
    gth = args.gth.read_text()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (cell, atoms) in SYSTEMS.items():
        energy, printed, forces = run(args.cpmd, gth, name, cell, atoms)
        ordered = [a for _, group in by_species(atoms) for a in group]
        lines = [
            f"# {name}: OpenCPMD 062582b empirical D2 (ALL DFT-D2, BLYP s6)",
            "# atoms: Z x y z in Angstrom (bohr times 0.529177210859), in CPMD's",
            "# species order; energy Ha (printed: CPMD's eight decimal line);",
            "# force Ha/bohr. box rows are the lattice vectors in Angstrom.",
            f"natoms {len(ordered)}",
            "box " + " ".join(repr(v * CPMD_BOHR) for r in box_rows(cell) for v in r),
        ]
        for z, x, y, w in ordered:
            lines.append(f"atom {z} {x * CPMD_BOHR!r} {y * CPMD_BOHR!r} {w * CPMD_BOHR!r}")
        lines.append(f"energy {energy!r}")
        lines.append(f"printed {printed!r}")
        for f in forces:
            lines.append("force " + " ".join(repr(v) for v in f))
        (args.out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {energy!r} Ha")


if __name__ == "__main__":
    main()
