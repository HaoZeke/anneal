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

    def eval(self, symbols, positions, cell=None):
        xyz = os.path.join(self.dir, "geo.xyz")
        with open(xyz, "w") as f:
            f.write(f"{len(symbols)}\n\n")
            for s, p in zip(symbols, positions):
                f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
        xcontrol = os.path.join(self.dir, "xcontrol")
        if cell is not None:
            a = (cell[0] ** 2 + cell[1] ** 2 + cell[2] ** 2) ** 0.5
            b = (cell[3] ** 2 + cell[4] ** 2 + cell[5] ** 2) ** 0.5
            c = (cell[6] ** 2 + cell[7] ** 2 + cell[8] ** 2) ** 0.5
            with open(xcontrol, "w") as f:
                f.write("$periodic\n$cell angs\n")
                f.write(f"  {a:.10f} {b:.10f} {c:.10f} 90.0 90.0 90.0\n")
        elif os.path.exists(xcontrol):
            os.remove(xcontrol)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        cmd = [self.exe, "geo.xyz", "--grad", "--gfn", "2", "--norestart"]
        if cell is not None:
            cmd.extend(["--input", "xcontrol"])
        subprocess.run(
            cmd,
            cwd=self.dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=300,
            env=env,
        )
        for leftover in ("xtbrestart", "charges", "wbo", "xtbtopo.mol", ".xtboptok"):
            p = os.path.join(self.dir, leftover)
            if os.path.exists(p):
                os.remove(p)
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

        if "CP2K_DATA_DIR" not in os.environ and "CONDA_PREFIX" in os.environ:
            os.environ["CP2K_DATA_DIR"] = os.path.join(
                os.environ["CONDA_PREFIX"], "share", "cp2k", "data"
            )
        return CP2K(
            command=os.environ.get("ASE_CP2K_COMMAND", "cp2k.ssmp --shell"),
            xc="PBE",
            print_level="LOW",
            basis_set="DZVP-MOLOPT-SR-GTH",
            pseudo_potential="GTH-PBE",
            max_scf=200,
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
        cell = None
        if tag.startswith("CELL"):
            vals = [float(v) for v in tag.split()[1:]]
            if len(vals) != 9:
                raise SystemExit(f"protocol error: CELL needs 9 numbers, got {tag!r}")
            cell = vals
            tag = stdin.readline().strip()
        if tag != "EVAL":
            raise SystemExit(f"protocol error: expected EVAL, got {tag!r}")
        try:
            if isinstance(calc, XtbCli):
                energy, forces = calc.eval(symbols, positions, cell)
            else:
                from ase import Atoms

                atoms = Atoms(symbols=symbols, positions=positions)
                if cell is not None:
                    atoms.set_cell(
                        [
                            [cell[0], cell[1], cell[2]],
                            [cell[3], cell[4], cell[5]],
                            [cell[6], cell[7], cell[8]],
                        ]
                    )
                    atoms.pbc = True
                elif os.environ.get("ASE_ENGINE") == "cp2k":
                    # CP2K needs a cell. Resize-every-call (center(vacuum=...))
                    # changes the Poisson grid between hops and the restart
                    # guess then fails to converge, aborting the shell.
                    atoms.set_cell([24.0, 24.0, 24.0])
                    atoms.center()
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
            out.write(f"E {energy:.10f}\n")
            for f in forces:
                out.write(f"{f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")
        except BrokenPipeError:
            return
        except Exception as exc:  # a failed SCF is a refused evaluation
            try:
                out.write(f"FAIL {exc.__class__.__name__}\n")
                for _ in range(n):
                    out.write("0 0 0\n")
            except BrokenPipeError:
                return
            if os.environ.get("ASE_ENGINE") == "cp2k":
                # An unconverged SCF aborts cp2k_shell. Recreate so later
                # evaluations are not counted as failures of a dead engine.
                try:
                    calc = make_calculator()
                except Exception:
                    pass
        try:
            out.write("DONE\n")
            out.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
