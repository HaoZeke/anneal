"""Persistent objective server for molecular-cluster searches.

Speaks a line protocol on stdin/stdout so the Rust driver pays engine startup
once rather than per evaluation:

    request:  N\n  then N lines  "symbol x y z"\n  then "EVAL"\n
    reply:    "E <energy_eV>"\n  then N lines "fx fy fz" (eV/Angstrom)\n "DONE"\n

Engine by environment:
    ASE_ENGINE=xtb   GFN2-xTB (fast smoke, real chemistry)
    ASE_ENGINE=cp2k  PBE through CP2K (set ASE_CP2K_COMMAND to cp2k_shell)

Energies and forces pass through unchanged; unit handling and convergence
settings live here, visible, rather than inside the driver.
"""

import os
import sys

from ase import Atoms


def make_calculator():
    engine = os.environ.get("ASE_ENGINE", "xtb")
    if engine == "xtb":
        from xtb.ase.calculator import XTB

        return XTB(method="GFN2-xTB")
    if engine == "cp2k":
        from ase.calculators.cp2k import CP2K

        return CP2K(
            command=os.environ.get("ASE_CP2K_COMMAND", "cp2k_shell.psmp"),
            xc="PBE",
            cutoff=400 * 13.605693,  # 400 Ry in eV, ASE takes eV
            basis_set="DZVP-MOLOPT-SR-GTH",
            pseudo_potential="GTH-PBE",
            stress_tensor=False,
        )
    raise SystemExit(f"unknown ASE_ENGINE {engine!r}")


def main():
    calc = make_calculator()
    stdin = sys.stdin
    out = sys.stdout
    while True:
        header = stdin.readline()
        if not header:
            return
        n = int(header.strip())
        symbols = []
        positions = []
        for _ in range(n):
            parts = stdin.readline().split()
            symbols.append(parts[0])
            positions.append([float(v) for v in parts[1:4]])
        tag = stdin.readline().strip()
        if tag != "EVAL":
            raise SystemExit(f"protocol error: expected EVAL, got {tag!r}")
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            out.write(f"E {energy:.10f}\n")
            for f in forces:
                out.write(f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")
        except Exception as exc:  # a failed SCF is a refused evaluation
            out.write(f"FAIL {exc.__class__.__name__}\n")
            for _ in range(n):
                out.write("0 0 0\n")
        out.write("DONE\n")
        out.flush()


if __name__ == "__main__":
    main()
