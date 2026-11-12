#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fail unless ASE and anneal agree to 1e-8 on Cambridge Gupta geometries.

The comparison is the energy and the Cartesian gradient. ASE reports forces,
which are the negative gradient. A displaced zinc geometry is included so the
gradient is not the near-zero gradient of a minimum.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

from gupta_calc import GuptaCalculator, evaluate

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(os.environ.get("METAL_FIXTURES", ROOT / "tests" / "fixtures" / "ccd"))
TOLERANCE = 1e-8

CASES = (
    ("zn38", FIXTURES / "gupta_zn" / "38", None, -50.499983),
    ("zn55", FIXTURES / "gupta_zn" / "55", None, -73.310032),
    ("zn80", FIXTURES / "gupta_zn" / "80", None, -106.927940),
    ("cu19au19", FIXTURES / "cuau38" / "Cu19Au19.txt", "header", -122.99148391),
)


def read_geometry(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Coordinates, and species atomic numbers when the file carries a type column.

    Cambridge copper–gold files interleave the species. Type 0 is copper
    and type 1 is gold. The returned coordinates are reordered copper then
    gold, which is the order ``h2h_metal`` evaluates.
    """
    rows = []
    kinds = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 3:
            try:
                rows.append([float(word) for word in fields])
            except ValueError:
                continue
        elif len(fields) == 4 and fields[0] in {"0", "1"}:
            kinds.append(29 if fields[0] == "0" else 79)
            rows.append([float(word) for word in fields[1:]])
    xyz = np.asarray(rows, dtype=float)
    if not kinds:
        return xyz, None
    order = np.argsort(np.array(kinds), kind="stable")
    return xyz[order], np.array(kinds, dtype=int)[order]


def anneal_gradient(binary: Path, system: str, path: Path) -> tuple[float, np.ndarray]:
    proc = subprocess.run(
        [str(binary), "grad", system, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    energy = float(lines[0])
    gradient = np.array([[float(word) for word in line.split()] for line in lines[1:]], dtype=float)
    return energy, gradient


def compare(binary: Path, system: str, xyz: np.ndarray, numbers: list[int], listed: float | None) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geom.xyz"
        path.write_text("".join(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n" for row in xyz))
        rust_energy, rust_gradient = anneal_gradient(binary, system, path)
    ase_energy, ase_gradient = evaluate(np.array(numbers), xyz)
    atoms = Atoms(numbers=numbers, positions=xyz)
    atoms.calc = GuptaCalculator()
    forces = atoms.get_forces()
    gaps = [
        abs(rust_energy - ase_energy),
        abs(rust_energy - float(atoms.get_potential_energy())),
        float(np.max(np.abs(rust_gradient - ase_gradient))),
        float(np.max(np.abs(forces + rust_gradient))),
    ]
    if listed is not None and abs(ase_energy - listed) >= 1e-6:
        print(f"{system} energy {ase_energy:.10f} is not the Cambridge {listed:.8f}", file=sys.stderr)
        return 1.0
    gap = max(gaps)
    print(f"{system} energy {ase_energy:.8f} gap {gap:.3e}")
    return gap


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("check_match.py BINARY", file=sys.stderr)
        return 2
    binary = Path(argv[1])
    worst = 0.0
    for system, path, _marker, listed in CASES:
        xyz, labels = read_geometry(path)
        numbers = labels.tolist() if labels is not None else [30] * len(xyz)
        worst = max(worst, compare(binary, system, xyz, numbers, listed))
        if worst > TOLERANCE:
            print(f"gap {worst:.3e} exceeds {TOLERANCE}", file=sys.stderr)
            return 1
    xyz, _labels = read_geometry(FIXTURES / "gupta_zn" / "38")
    xyz = xyz.copy()
    xyz[0, 0] += 0.05
    xyz[3, 1] -= 0.02
    worst = max(worst, compare(binary, "zn38", xyz, [30] * 38, None))
    print(f"worst {worst:.3e}")
    if worst > TOLERANCE:
        print(f"gap {worst:.3e} exceeds {TOLERANCE}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
