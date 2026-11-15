#!/usr/bin/env python3
"""Run one LAMMPS case and write what it reports as JSON.

    lmp_case.py case.json out.json

case.json holds "commands", a list of LAMMPS input lines, and
optionally "strains", a list of lower-triangular 3x3 displacement
gradients e. After the commands, "run 0" is not issued again; the
script reads the potential energy, the pressure tensor of the compute
named "rgpress" when the commands define it, the forces in atom-ID
order and the cell. For each strain the cell rows and the positions go
to r (I + e) and the energy of that cell is recorded, then the
original cell is restored.

A LAMMPS error is written as {"error": message}. The script may run
under mpirun; rank 0 writes the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cell(lmp) -> list[list[float]]:
    boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
    return [
        [boxhi[0] - boxlo[0], 0.0, 0.0],
        [xy, boxhi[1] - boxlo[1], 0.0],
        [xz, yz, boxhi[2] - boxlo[2]],
    ]


def set_cell(lmp, h) -> None:
    boxlo = lmp.extract_box()[0]
    lmp.command(
        "change_box all "
        f"x final {boxlo[0]!r} {boxlo[0] + h[0][0]!r} "
        f"y final {boxlo[1]!r} {boxlo[1] + h[1][1]!r} "
        f"z final {boxlo[2]!r} {boxlo[2] + h[2][2]!r} "
        f"xy final {h[1][0]!r} xz final {h[2][0]!r} yz final {h[2][1]!r} "
        "remap units box"
    )


def strained_energy(lmp, e: list[list[float]]) -> float:
    """Energy with the cell rows and positions sent to r (I + e).

    change_box with remap keeps the fractional coordinates, so the
    atoms follow the cell. e is lower triangular, which keeps the cell
    a LAMMPS (restricted triclinic) cell.
    """
    h0 = cell(lmp)
    h1 = [[sum(h0[i][k] * ((1.0 if k == j else 0.0) + e[k][j]) for k in range(3))
           for j in range(3)] for i in range(3)]
    set_cell(lmp, h1)
    lmp.command("run 0 post no")
    energy = float(lmp.get_thermo("pe"))
    set_cell(lmp, h0)
    return energy


def main() -> None:
    from lammps import lammps

    case = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    rank = lmp.extract_setting("world_rank")
    result: dict = {"nprocs": lmp.extract_setting("world_size")}
    try:
        for command in case["commands"]:
            lmp.command(command)
        natoms = lmp.get_natoms()
        result["pe"] = float(lmp.get_thermo("pe"))
        result["forces"] = [
            float(v) for v in lmp.gather_atoms("f", 1, 3)[: 3 * natoms]
        ]
        result["x"] = [
            [float(v) for v in row]
            for row in zip(*[iter(lmp.gather_atoms("x", 1, 3)[: 3 * natoms])] * 3)
        ]
        result["types"] = [int(t) for t in lmp.gather_atoms("type", 0, 1)[:natoms]]
        result["boxlo"] = [float(v) for v in lmp.extract_box()[0]]
        result["cell"] = cell(lmp)
        result["volume"] = float(lmp.get_thermo("vol"))
        result["nktv2p"] = float(lmp.extract_global("nktv2p"))
        if any(c.startswith("compute rgpress ") for c in case["commands"]):
            press = lmp.numpy.extract_compute("rgpress", 0, 1)
            result["press"] = [float(v) for v in press[:6]]
        strains = case.get("strains", [])
        if strains and not lmp.extract_global("triclinic"):
            lmp.command("change_box all triclinic")
        result["strained"] = [strained_energy(lmp, e) for e in strains]
    except Exception as exc:  # LAMMPS raises its error text
        result = {"error": str(exc)}
    finally:
        lmp.close()
    if rank == 0:
        out_path.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
