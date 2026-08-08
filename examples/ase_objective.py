"""Persistent objective server for molecular-cluster searches.

Speaks a line protocol on stdin/stdout so the Rust driver pays engine startup
once rather than per evaluation:

    request:  N\n  then N lines  "symbol x y z"\n  then "EVAL"\n
    reply:    "E <energy_eV>"\n  then N lines "fx fy fz" (eV/Angstrom)\n "DONE"\n

Engine by environment:
    ASE_ENGINE=xtb      GFN2-xTB through the Python bindings
    ASE_ENGINE=xtb-cli  GFN2-xTB through the xtb executable (no bindings)
    ASE_ENGINE=cp2k     PBE through CP2K (set ASE_CP2K_COMMAND to cp2k_shell)

Energies and forces pass through unchanged; unit handling and convergence
settings live here, visible, rather than inside the driver.
"""

import os
import subprocess
import sys
import tempfile

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903


class XtbCli:
    """GFN2-xTB through the executable: xyz in, Turbomole gradient out."""

    def __init__(self):
        self.exe = os.environ.get("XTB_EXE", "xtb")
        self.dir = tempfile.mkdtemp(prefix="xtbcli-")

    def eval(self, symbols, positions):
        xyz = os.path.join(self.dir, "geo.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(symbols)}\n\n")
            for s, p in zip(symbols, positions):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        subprocess.run(
            [self.exe, "geo.xyz", "--grad", "--gfn", "2"],
            cwd=self.dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=300,
        )
        grad_path = os.path.join(self.dir, "gradient")
        energy = None
        grads = []
        with open(grad_path) as f:
            for line in f:
                parts = line.split()
                if "SCF energy" in line or "energy =" in line:
                    for i, tok in enumerate(parts):
                        if tok == "energy" and i + 2 < len(parts):
                            energy = float(parts[i + 2])
                if len(parts) == 3:
                    try:
                        g = [float(v.replace("D", "E")) for v in parts]
                        grads.append(g)
                    except ValueError:
                        pass
        os.remove(grad_path)
        n = len(symbols)
        grads = grads[-n:]
        e_ev = energy * HARTREE_EV
        forces = [
            [-g[0] * HARTREE_EV / BOHR_ANG, -g[1] * HARTREE_EV / BOHR_ANG, -g[2] * HARTREE_EV / BOHR_ANG]
            for g in grads
        ]
        return e_ev, forces


def make_calculator():
    engine = os.environ.get("ASE_ENGINE", "xtb")
    if engine == "xtb-cli":
        return XtbCli()
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
        try:
            if isinstance(calc, XtbCli):
                energy, forces = calc.eval(symbols, positions)
            else:
                from ase import Atoms

                atoms = Atoms(symbols=symbols, positions=positions)
                atoms.calc = calc
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
