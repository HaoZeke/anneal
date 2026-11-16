# Copyright (c) anneal developers
"""Replay three Si10 geometries through rgpot's xtb engine, warm and cold.

Usage: reproduce.py <libxtb_engine.so> [calls.xyz]

The engine is created as anneal creates it for the GFN2 Si10 bench: GFN2,
accuracy 0.01, 500 iterations, 3000 K, neutral, closed shell, default warm
move. After a reset each frame of ``calls.xyz`` is evaluated in order, so
each reuses the previous frame's wavefunction when the engine allows it;
then each frame is evaluated alone after a reset. Energies are in eV.
xtb's shared library must be on LD_LIBRARY_PATH.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

GFN2 = 3
SILICON = 14


class Config(ctypes.Structure):
    _fields_ = [
        ("method", ctypes.c_int),
        ("accuracy", ctypes.c_double),
        ("electronic_temperature", ctypes.c_double),
        ("max_iterations", ctypes.c_int),
        ("charge", ctypes.c_double),
        ("uhf", ctypes.c_int),
    ]


def frames(path: Path) -> list[tuple[str, list[float]]]:
    lines = path.read_text().splitlines()
    out = []
    while lines:
        n = int(lines[0])
        xyz = [float(v) for line in lines[2 : 2 + n] for v in line.split()[1:4]]
        out.append((lines[1], xyz))
        lines = lines[2 + n :]
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__.splitlines()[2])
    here = Path(__file__).resolve().parent
    path = Path(sys.argv[2]) if len(sys.argv) == 3 else here / "calls.xyz"
    lib = ctypes.CDLL(sys.argv[1])
    lib.rgpot_xtb_create.restype = ctypes.c_void_p
    lib.rgpot_xtb_create.argtypes = [
        ctypes.POINTER(Config),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.rgpot_xtb_force.argtypes = [ctypes.c_void_p, ctypes.c_long] + [
        ctypes.c_void_p
    ] * 6
    lib.rgpot_xtb_reset.argtypes = [ctypes.c_void_p]
    lib.rgpot_xtb_reused_wavefunction.argtypes = [ctypes.c_void_p]
    lib.rgpot_xtb_destroy.argtypes = [ctypes.c_void_p]
    err = ctypes.create_string_buffer(512)
    pot = lib.rgpot_xtb_create(
        ctypes.byref(Config(GFN2, 0.01, 3000.0, 500, 0.0, 0)), err, len(err)
    )
    if not pot:
        raise SystemExit(err.value.decode())

    def force(xyz: list[float]) -> tuple[float, int]:
        n = len(xyz) // 3
        energy = ctypes.c_double()
        rc = lib.rgpot_xtb_force(
            pot,
            n,
            (ctypes.c_double * (3 * n))(*xyz),
            (ctypes.c_int * n)(*([SILICON] * n)),
            (ctypes.c_double * (3 * n))(),
            ctypes.byref(energy),
            None,
            (ctypes.c_double * 9)(),
        )
        if rc:
            raise SystemExit(f"rgpot_xtb_force returned {rc}")
        return energy.value, lib.rgpot_xtb_reused_wavefunction(pot)

    sequence = frames(path)
    lib.rgpot_xtb_reset(pot)
    warm = [force(xyz) for _, xyz in sequence]
    print("frame,in_sequence_eV,reused,alone_eV")
    for (title, xyz), (energy, reused) in zip(sequence, warm):
        lib.rgpot_xtb_reset(pot)
        alone, _ = force(xyz)
        print(f"{title.split()[1]},{energy:.10f},{reused},{alone:.10f}")
    lib.rgpot_xtb_destroy(pot)


if __name__ == "__main__":
    main()
