#!/usr/bin/env python3
"""Regenerate the DFT-D4 BLYP references in CppCore/tests/data/d4/.

The reference is the dftd4 program (4.2.0, the pixi `dftd` environment's
conda-forge build), run as `dftd4 coord --func blyp --grad --json` with the
default D4 model, EEQ charges and the ATM term. The geometries are those of
the D2 references in CppCore/tests/data/d2/: a water dimer and benzene as
molecules, and the eight-water box in a cubic and a sheared cell. They go to
dftd4 as Turbomole coord files in bohr, with `$periodic 3` and `$lattice
bohr` for the cells, so dftd4 runs its periodic code path on the same lattice
rgpot passes through the C API.

Usage: regen_d4_references.py [--dftd4 PATH/TO/dftd4] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

BOHR_TO_ANGSTROM = 0.529177210903  # rgpot/units.hpp
SYMBOL = {1: "h", 6: "c", 7: "n", 8: "o"}
ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["water_dimer", "benzene", "water_box_cubic", "water_box_sheared"]


def load_d2(name):
    atoms, box, periodic = [], None, False
    for line in (ROOT / "CppCore" / "tests" / "data" / "d2" / f"{name}.txt").read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "periodic":
            periodic = parts[1] == "1"
        elif parts[0] == "box":
            v = [float(x) for x in parts[1:]]
            box = [v[0:3], v[3:6], v[6:9]]
        elif parts[0] == "atom":
            atoms.append((int(parts[1]), *(float(x) for x in parts[2:5])))
    return atoms, box if periodic else None


def run(dftd4, atoms, box, work):
    """dftd4 on the geometry in bohr; returns Ha, Ha/bohr and the virial."""
    with open(work / "coord", "w") as fh:
        fh.write("$coord\n")
        for z, *x in atoms:
            fh.write(" ".join(f"{v / BOHR_TO_ANGSTROM:.17g}" for v in x) + f" {SYMBOL[z]}\n")
        if box is not None:
            fh.write("$periodic 3\n$lattice bohr\n")
            for row in box:
                fh.write(" ".join(f"{v / BOHR_TO_ANGSTROM:.17g}" for v in row) + "\n")
        fh.write("$end\n")
    subprocess.run([dftd4, "coord", "-i", "tmol", "--func", "blyp", "--grad", "--json",
                    "out.json", "--noedisp", "-s"],
                   cwd=work, check=True, capture_output=True, text=True)
    data = json.loads((work / "out.json").read_text())
    return data["version"], data["energy"], data["gradient"], data["virial"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dftd4", default=shutil.which("dftd4"))
    parser.add_argument("--out", default=str(ROOT / "CppCore" / "tests" / "data" / "d4"))
    args = parser.parse_args()
    if not args.dftd4:
        raise SystemExit("no dftd4 on PATH; pass --dftd4")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SYSTEMS:
        atoms, box = load_d2(name)
        with tempfile.TemporaryDirectory() as tmp:
            version, energy, grad, virial = run(str(Path(args.dftd4).resolve()), atoms, box, Path(tmp))
        lines = [
            f"# {name}: dftd4 {version} --func blyp --grad"
            + (" ($periodic 3)" if box else ""),
            "# atoms: Z x y z in Angstrom; energy Ha; grad Ha/bohr; virial Ha,",
            "# dftd4's sigma, dE/dstrain at fixed fractional coordinates.",
            f"natoms {len(atoms)}",
            "periodic " + ("1" if box else "0"),
            "box " + " ".join(f"{v:.17g}" for row in (box or [[0.0] * 3] * 3) for v in row),
        ]
        for z, *x in atoms:
            lines.append(f"atom {z} " + " ".join(f"{v:.17g}" for v in x))
        lines.append(f"energy {energy:.17g}")
        for i in range(len(atoms)):
            lines.append("grad " + " ".join(f"{v:.17g}" for v in grad[3 * i:3 * i + 3]))
        if box:
            for i in range(3):
                lines.append("virial " + " ".join(f"{v:.17g}" for v in virial[3 * i:3 * i + 3]))
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(name, version, energy)


if __name__ == "__main__":
    main()
