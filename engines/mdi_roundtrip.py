#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""MDI TCP round trips for two concrete pot_spec potentials."""

from __future__ import annotations

import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import mdi

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988
COMMANDS = (
    ">NATOMS",
    "<NATOMS",
    ">COORDS",
    "<COORDS",
    ">CELL",
    "<CELL",
    ">ELEMENTS",
    ">TYPES",
    "<ENERGY",
    "<FORCES",
    "<PE",
    "EXIT",
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def mdi_library() -> Path:
    root = Path(mdi.__file__).parent
    for name in ("libmdi.so", "libmdi.dylib", "mdi.dll"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"official MDI Library not found in {root}")


def direct(executable: Path, spec: str, coords: list[float], cell: list[float]):
    tokens = spec.split()
    geometry = ["2", " ".join(str(value) for value in cell)]
    geometry.extend(
        f"0 {coords[index]} {coords[index + 1]} {coords[index + 2]}"
        for index in range(0, 6, 3)
    )
    result = subprocess.run(
        [str(executable), *tokens],
        input="\n".join(geometry) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    energy = float(lines[0].split()[1])
    forces = [
        float(value)
        for line in lines[1:3]
        for value in line.split()[2:]
    ]
    return energy, forces


def run_case(engine: Path, evaluator: Path, spec: str) -> None:
    port = free_port()
    library = mdi_library()
    mdi.MDI_Init(
        f"-role DRIVER -name TEST -method TCP -hostname 127.0.0.1 -port {port}"
    )
    process = subprocess.Popen(
        [
            str(engine),
            "--pot",
            spec,
            "--mdi-library",
            str(library),
            "-mdi",
            (
                "-role ENGINE -name RGPOT -method TCP "
                f"-hostname 127.0.0.1 -port {port}"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    comm = mdi.MDI_Accept_Communicator()
    for command in COMMANDS:
        assert mdi.MDI_Check_Command_Exists("@DEFAULT", command, comm) == 1

    mdi.MDI_Send_Command(">NATOMS", comm)
    mdi.MDI_Send(2, 1, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command("<NATOMS", comm)
    assert mdi.MDI_Recv(1, mdi.MDI_INT, comm) == 2
    mdi.MDI_Send_Command(">ELEMENTS", comm)
    mdi.MDI_Send([18, 18], 2, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command(">TYPES", comm)
    mdi.MDI_Send([0, 0], 2, mdi.MDI_INT, comm)

    coords_angstrom = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    cell_angstrom = [20.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 22.0]
    coords = [value / BOHR_TO_ANGSTROM for value in coords_angstrom]
    cell = [value / BOHR_TO_ANGSTROM for value in cell_angstrom]
    mdi.MDI_Send_Command(">COORDS", comm)
    mdi.MDI_Send(coords, 6, mdi.MDI_DOUBLE, comm)
    mdi.MDI_Send_Command("<COORDS", comm)
    assert list(mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm)) == coords
    mdi.MDI_Send_Command(">CELL", comm)
    mdi.MDI_Send(cell, 9, mdi.MDI_DOUBLE, comm)
    mdi.MDI_Send_Command("<CELL", comm)
    assert list(mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)) == cell
    mdi.MDI_Send_Command("<ENERGY", comm)
    energy = float(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))
    mdi.MDI_Send_Command("<FORCES", comm)
    forces = list(mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm))
    mdi.MDI_Send_Command("<PE", comm)
    pe = float(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))

    direct_energy, direct_forces = direct(
        evaluator, spec, coords_angstrom, cell_angstrom
    )
    tolerance = sys.float_info.epsilon * 128.0
    assert math.isclose(
        energy * HARTREE_TO_EV,
        direct_energy,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    assert energy == pe
    for actual, expected in zip(forces, direct_forces, strict=True):
        assert math.isclose(
            actual * HARTREE_TO_EV / BOHR_TO_ANGSTROM,
            expected,
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    smoke_steps = int(os.environ.get("RGPOT_MDI_SMOKE_STEPS", "0"))
    if smoke_steps > 0:
        started = time.perf_counter()
        for step in range(smoke_steps):
            coords[3] = (4.0 + float(step % 2)) / BOHR_TO_ANGSTROM
            mdi.MDI_Send_Command(">COORDS", comm)
            mdi.MDI_Send(coords, 6, mdi.MDI_DOUBLE, comm)
            mdi.MDI_Send_Command("<FORCES", comm)
            mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm)
        elapsed = time.perf_counter() - started
        print(f"{spec.split()[0]} MDI {elapsed * 1_000_000 / smoke_steps:.3f} us/step")
    mdi.MDI_Send_Command("EXIT", comm)
    _, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--case":
        run_case(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return 0
    if len(sys.argv) != 3:
        raise SystemExit("usage: mdi_roundtrip.py RGPOT_MDI RGPOT_EVAL")
    script = Path(__file__).resolve()
    for spec in (
        "LJ u0 0.0104 psi 3.4 cutoff 8.5",
        "Morse de 0.7102 a 1.6047 re 2.897 cutoff 8.5",
    ):
        subprocess.run(
            [sys.executable, str(script), "--case", sys.argv[1], sys.argv[2], spec],
            env=os.environ.copy(),
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
