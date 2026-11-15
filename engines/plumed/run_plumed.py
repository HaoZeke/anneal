#!/usr/bin/env python3
"""Checks of the RGPOT collective variable through ``plumed driver``.

    run_plumed.py dimer rgpot-lj libRgpotPlumed.so
    run_plumed.py forces libRgpotPlumed.so [CuNi.eam.alloy]
    run_plumed.py box libRgpotPlumed.so [CuNi.eam.alloy]
    run_plumed.py units libRgpotPlumed.so
    run_plumed.py api libRgpotPlumed.so librgpot.so [python build dir]

``dimer`` compares the LJ dimer with ``rgpot-lj``. ``forces`` checks the
atomic derivatives and ``box`` the nine box derivatives against the finite
differences of ``--debug-forces``. ``units`` evaluates one cell in nm and
kJ/mol and again in Angstrom and eV. ``api`` compares energies, forces and
virials with the rgpot C and Python interfaces. Each subcommand writes
under plumed-<subcommand> in the current directory. Exit status 77 is a
skip, with the reason printed.
"""

from __future__ import annotations

import ctypes
import importlib
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy is not importable")
    sys.exit(77)


KJ_PER_EV = 96.48533212331002
# PLUMED's eV (tools/Units.cpp), which differs from rgpot's CODATA 2018 value.
PLUMED_EV_IN_KJ = 96.48530749925792

PT = "SuttonChen", "epsilon 0.019833 c 34.408 a 3.92 n 10 m 8 cutoff 5.0"
LJ = "LJ", "u0 0.0104 psi 3.40 cutoff 8.5"


def fcc(a: float, n: int, tilt: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Fractional fcc sites of n^3 cells and the cell rows, lower triangular."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    side = n * a
    h = np.array([[side, 0, 0], [tilt[0], side, 0], [tilt[1], tilt[2], side]])
    return frac, h


def diamond(a: float, n: int, tilt: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    frac, h = fcc(a, n, tilt)
    return np.vstack([frac, frac + 0.25 / n]), h


def jiggle(frac: np.ndarray, h: np.ndarray, amplitude: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return frac @ h + rng.uniform(-amplitude, amplitude, size=frac.shape)


def cluster(frac: np.ndarray, h: np.ndarray, count: int, seed: int) -> np.ndarray:
    """The count sites closest to the cell centre, jiggled, without a cell."""
    pos = jiggle(frac, h, 0.08, seed)
    centre = 0.5 * h.sum(axis=0)
    order = np.argsort(np.linalg.norm(pos - centre, axis=1))
    return pos[order[:count]]


# Each case: potential, options, TYPES, symbols written to the xyz file,
# positions and cell in Angstrom (cell None for a cluster under NOPBC).
def cases(eam_table: str) -> dict:
    out = {}
    frac, h = fcc(5.26, 4, (1.2, -0.8, 1.0))
    out["lj_triclinic"] = (*LJ, "Ar", ["Ar"] * len(frac), jiggle(frac, h, 0.15, 1), h)
    out["lj_cluster"] = (*LJ, "Ar", ["Ar"] * 19, cluster(frac, h, 19, 2), None)
    frac, h = fcc(3.92, 3, (0.8, 0.5, -0.6))
    out["suttonchen_triclinic"] = (*PT, "Pt", ["Pt"] * len(frac), jiggle(frac, h, 0.06, 3), h)
    out["suttonchen_cluster"] = (*PT, "Pt", ["Pt"] * 13, cluster(frac, h, 13, 4), None)
    frac, h = diamond(5.431, 2, (0.5, -0.4, 0.3))
    out["swsi1985_triclinic"] = (
        "SWSi1985", "", "14", ["Si"] * len(frac), jiggle(frac, h, 0.08, 5), h)
    frac, h = fcc(3.6, 3, (0.0, 0.0, 0.0))
    out["emt_cluster"] = ("EMT", "", "Cu", ["Cu"] * 13, cluster(frac, h, 13, 6), None)
    if eam_table:
        frac, h = fcc(3.6, 4, (0.5, -0.4, 0.6))
        rng = np.random.default_rng(7)
        symbols = list(rng.choice(["Cu", "Ni"], size=len(frac)))
        out["eamtable_triclinic"] = (
            "EAMTable", f"file {eam_table}", ",".join(symbols), symbols,
            jiggle(frac, h, 0.06, 8), h)
    return out


def write_xyz(path: Path, symbols, pos: np.ndarray, cell, scale: float) -> None:
    """scale converts Angstrom to the file's length unit.

    The driver needs a cell line; a cluster gets a cube that NOPBC ignores.
    """
    if cell is None:
        cell = np.eye(3) * 100.0
    lines = [str(len(pos))]
    lines.append(" ".join(f"{v * scale:.15g}" for v in cell.reshape(-1)))
    lines += [f"{s} {x * scale:.15g} {y * scale:.15g} {z * scale:.15g}" for s, (x, y, z) in
              zip(symbols, pos)]
    path.write_text("\n".join(lines) + "\n")


def plumed_input(plugin: str, potential: str, options: str, types: str, natoms: int,
                 nopbc: bool, units: str = "") -> str:
    action = (f"ene: RGPOT ATOMS=1-{natoms} POTENTIAL={potential} TYPES={types}"
              + (f" OPTIONS={{{options}}}" if options else "") + (" NOPBC" if nopbc else ""))
    return "\n".join(
        [f"LOAD FILE={Path(plugin).resolve()}"] + ([units] if units else [])
        + [action, "BIASVALUE ARG=ene", "PRINT ARG=ene FILE=COLVAR FMT=%.17g", ""])


def driver(work: Path, tag: str, plumed_dat: str, symbols, pos, cell, *, length_units: str = "",
           debug: bool = False, check: bool = True) -> dict:
    """Run plumed driver on one frame. Lengths in the file follow length_units."""
    scale = 1.0 if length_units == "A" else 0.1
    case = work / tag
    case.mkdir(parents=True, exist_ok=True)
    (case / "plumed.dat").write_text(plumed_dat)
    write_xyz(case / "frame.xyz", symbols, pos, cell, scale)
    for name in ("COLVAR", "forces.txt", "debug.txt"):
        (case / name).unlink(missing_ok=True)
    cmd = ["plumed", "driver", "--plumed", "plumed.dat", "--ixyz", "frame.xyz",
           "--dump-forces", "forces.txt", "--dump-forces-fmt", "%.17g", "--dump-full-virial"]
    if length_units:
        cmd += ["--length-units", length_units]
    if debug:
        cmd += ["--debug-forces", "debug.txt"]
    proc = subprocess.run(cmd, cwd=case, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(f"{tag}: plumed driver failed\n{proc.stdout}\n{proc.stderr}")
    out = {"returncode": proc.returncode, "log": proc.stdout + proc.stderr}
    if proc.returncode != 0:
        return out
    values = [line for line in (case / "COLVAR").read_text().splitlines()
              if line and not line.startswith("#")]
    out["value"] = float(values[-1].split()[-1])
    dump = (case / "forces.txt").read_text().splitlines()
    out["virial"] = np.array([float(v) for v in dump[1].split()]).reshape(3, 3)
    out["forces"] = np.array([[float(v) for v in line.split()[1:4]] for line in dump[2:]])
    if debug:
        out["debug"] = np.loadtxt(case / "debug.txt", comments="#")
    return out


def check_fd(tag: str, rows: np.ndarray) -> None:
    """Analytic against numerical derivatives.

    The numerical column has six decimals and, on the cells of a few
    hundred atoms, cancellation noise of a few 1e-5 of the largest
    derivative. A central difference with a chosen step agrees to 1e-8.
    """
    analytic, numeric = rows[:, 1], rows[:, 2]
    scale = float(np.max(np.abs(analytic)))
    if scale < 1.0:
        raise SystemExit(f"{tag}: derivatives too small to test ({scale})")
    worst = float(np.max(np.abs(analytic - numeric)))
    print(f"{tag}: max |analytic| {scale:.6g}, max |analytic - numerical| {worst:.3g}")
    if worst > 5e-5 * scale + 1e-3:
        raise SystemExit(f"{tag}: finite difference disagrees")


def cmd_dimer(tool: str, plugin: str) -> None:
    work = Path.cwd() / "plumed-dimer"
    factor = float(subprocess.check_output([tool, "--kJ-per-eV"], text=True).strip())
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    cell = np.diag([20.0, 20.0, 20.0])
    dat = plumed_input(plugin, "LJ", "u0 1.0 psi 1.0 cutoff 8.0", "Ar", 2, False)
    out = driver(work, "dimer", dat, ["Ar", "Ar"], pos, cell, debug=True)
    check_fd("dimer forces", out["debug"][:-9])
    check_fd("dimer box", out["debug"][-9:])
    energy = out["value"] / factor
    direct = subprocess.check_output(
        [tool], input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True)
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(energy - ref) > 1e-8:
        raise SystemExit(f"plumed {energy} eV, rgpot {ref} eV")
    print(f"plumed energy {energy:.10f} eV")


def cmd_forces(plugin: str, eam_table: str) -> None:
    work = Path.cwd() / "plumed-forces"
    if not eam_table:
        print("SKIP eamtable: no EAM table (configure with -Dwith_eam_table=true)")
    for tag, (pot, opts, types, symbols, pos, cell) in cases(eam_table).items():
        dat = plumed_input(plugin, pot, opts, types, len(pos), cell is None)
        out = driver(work, tag, dat, symbols, pos, cell, debug=True)
        check_fd(f"{tag} forces", out["debug"][:-9])
    # A potential without a stress stops on a periodic cell. EMT rejects
    # the 3x3x3 cell, so this one is 5x5x5.
    frac, h = fcc(3.6, 5, (0.0, 0.0, 0.0))
    pos = jiggle(frac, h, 0.05, 9)
    dat = plumed_input(plugin, "EMT", "", "Cu", len(pos), False)
    out = driver(work, "emt_periodic", dat, ["Cu"] * len(pos), pos, h, check=False)
    if out["returncode"] == 0 or "returns no stress" not in out["log"]:
        raise SystemExit(f"EMT on a periodic cell did not stop with the stress error\n{out['log']}")
    print("emt_periodic: stops with the missing stress error")


def cmd_box(plugin: str, eam_table: str) -> None:
    work = Path.cwd() / "plumed-box"
    for tag, (pot, opts, types, symbols, pos, cell) in cases(eam_table).items():
        if cell is None:
            continue
        if not np.any(np.abs(np.tril(cell, -1)) > 0):
            raise SystemExit(f"{tag}: the box check needs a triclinic cell")
        dat = plumed_input(plugin, pot, opts, types, len(pos), False)
        out = driver(work, tag, dat, symbols, pos, cell, debug=True)
        check_fd(f"{tag} box", out["debug"][-9:])


def cmd_units(plugin: str) -> None:
    """One cell in PLUMED's default units and in Angstrom and eV."""
    work = Path.cwd() / "plumed-units"
    pot, opts, types, symbols, pos, cell = cases("")["suttonchen_triclinic"]
    natoms = len(pos)
    default = driver(work, "units_nm", plumed_input(plugin, pot, opts, types, natoms, False),
                     symbols, pos, cell)
    metal = driver(work, "units_A", plumed_input(plugin, pot, opts, types, natoms, False,
                                                 "UNITS LENGTH=A ENERGY=eV"),
                   symbols, pos, cell, length_units="A")
    mixed = driver(work, "units_mixed", plumed_input(plugin, pot, opts, types, natoms, False,
                                                     "UNITS LENGTH=A ENERGY=eV"),
                   symbols, pos, cell)
    kj = default["value"]
    for tag, value in (("A and eV", metal["value"]), ("nm file, A and eV", mixed["value"])):
        print(f"{tag}: {value:.12f} PLUMED eV = {value * PLUMED_EV_IN_KJ:.10f} kJ/mol, "
              f"default units {kj:.10f} kJ/mol")
        if abs(value * PLUMED_EV_IN_KJ - kj) > 1e-10 * abs(kj):
            raise SystemExit(f"{tag}: the value does not convert")
    # Driver forces are in kJ/mol per its length unit; virials in kJ/mol.
    df = np.max(np.abs(metal["forces"] * 10.0 - default["forces"]))
    dv = np.max(np.abs(metal["virial"] - default["virial"]))
    print(f"forces differ by {df:.3g} kJ/mol/nm, virials by {dv:.3g} kJ/mol")
    if df > 1e-9 * np.max(np.abs(default["forces"])) or dv > 1e-9 * np.max(
            np.abs(default["virial"])):
        raise SystemExit("forces or virials do not convert")


class ScConfig(ctypes.Structure):
    _fields_ = [("epsilon", ctypes.c_double), ("c", ctypes.c_double), ("a", ctypes.c_double),
                ("n", ctypes.c_int32), ("m", ctypes.c_int32), ("cutoff", ctypes.c_double),
                ("periodic", ctypes.c_int32)]


def sutton_chen_c(lib: str, pos: np.ndarray, cell: np.ndarray) -> tuple[float, np.ndarray]:
    rg = ctypes.CDLL(str(Path(lib).resolve()))
    fn = rg.rgpot_sutton_chen_energy_forces
    dp = ctypes.POINTER(ctypes.c_double)
    fn.argtypes = [ctypes.POINTER(ScConfig), ctypes.c_int32, dp, dp, dp, dp]
    fn.restype = ctypes.c_int
    words = dict(zip(*[iter(PT[1].split())] * 2))
    cfg = ScConfig(float(words["epsilon"]), float(words["c"]), float(words["a"]),
                   int(words["n"]), int(words["m"]), float(words["cutoff"]), 1)
    x = np.ascontiguousarray(pos, dtype=np.float64)
    box = np.ascontiguousarray(cell, dtype=np.float64)
    forces = np.zeros_like(x)
    energy = ctypes.c_double()
    status = fn(ctypes.byref(cfg), len(x), x.ctypes.data_as(dp), box.ctypes.data_as(dp),
                ctypes.byref(energy), forces.ctypes.data_as(dp))
    if status != 0:
        raise SystemExit(f"rgpot_sutton_chen_energy_forces returned {status}")
    return energy.value, forces


def python_core(pydir: str):
    """The rgpot Python module: the built extension, or an installed rgpot."""
    if pydir and Path(pydir).is_dir():
        sys.path.insert(0, str(Path(pydir).resolve()))
        try:
            return importlib.import_module("_core")
        except ImportError as err:
            print(f"cannot import _core from {pydir}: {err}")
    try:
        return importlib.import_module("rgpot")
    except ImportError:
        return None


def compare(tag: str, out: dict, energy: float, forces: np.ndarray, stress) -> None:
    """PLUMED output (kJ/mol, Angstrom) against an rgpot interface (eV)."""
    value = out["value"] / KJ_PER_EV
    f_pl = out["forces"] / KJ_PER_EV
    de = abs(value - energy)
    df = float(np.max(np.abs(f_pl - forces)))
    line = f"{tag}: energy {value:.12f} eV vs {energy:.12f} eV (diff {de:.2g}), max force diff {df:.2g} eV/A"
    bad = de > 1e-11 * max(1.0, abs(energy)) or df > 1e-10 * max(1.0, np.max(np.abs(forces)))
    if stress is not None:
        cell = stress[1]
        volume = abs(np.linalg.det(cell))
        expected = volume * np.asarray(stress[0]).T
        dv = float(np.max(np.abs(out["virial"] / KJ_PER_EV - expected)))
        line += f", max virial diff {dv:.2g} eV"
        bad = bad or dv > 1e-10 * max(1.0, np.max(np.abs(expected)))
    print(line)
    if bad:
        raise SystemExit(f"{tag}: PLUMED and the rgpot interface disagree")


def cmd_api(plugin: str, lib: str, pydir: str) -> None:
    work = Path.cwd() / "plumed-api"
    units = "UNITS LENGTH=A"
    all_cases = cases("")

    def run(tag):
        pot, opts, types, symbols, pos, cell = all_cases[tag]
        dat = plumed_input(plugin, pot, opts, types, len(pos), cell is None, units)
        return driver(work, f"api_{tag}", dat, symbols, pos, cell, length_units="A"), pos, cell

    out, pos, cell = run("suttonchen_triclinic")
    energy, forces = sutton_chen_c(lib, pos, cell)
    compare("suttonchen_triclinic, C API", out, energy, forces, None)

    core = python_core(pydir)
    if core is None:
        print("SKIP Python API: rgpot is not importable (configure with -Dwith_python=true)")
        return
    out, pos, cell = run("lj_triclinic")
    words = dict(zip(*[iter(LJ[1].split())] * 2))
    pot = core.LJPot(u0=float(words["u0"]), cutoff=float(words["cutoff"]), psi=float(words["psi"]))
    res = pot.evaluate(np.ascontiguousarray(pos), [18] * len(pos), np.ascontiguousarray(cell))
    compare("lj_triclinic, Python API", out, res["energy"], np.asarray(res["forces"]),
            (res["stress"], cell))
    out, pos, _ = run("lj_cluster")
    res = pot.evaluate(np.ascontiguousarray(pos), [18] * len(pos), np.eye(3) * 200.0)
    compare("lj_cluster, Python API", out, res["energy"], np.asarray(res["forces"]), None)
    out, pos, cell = run("swsi1985_triclinic")
    res = core.StillingerWeberSi().evaluate(np.ascontiguousarray(pos), [14] * len(pos),
                                           np.ascontiguousarray(cell))
    compare("swsi1985_triclinic, Python API", out, res["energy"], np.asarray(res["forces"]),
            (res["stress"], cell))


def main() -> None:
    args = sys.argv[1:]
    usage = __doc__.split("\n\n")[1]
    if not args:
        raise SystemExit(usage)
    sub, rest = args[0], args[1:]
    if sub == "dimer" and len(rest) == 2:
        cmd_dimer(*rest)
    elif sub in ("forces", "box") and len(rest) in (1, 2):
        (cmd_forces if sub == "forces" else cmd_box)(rest[0], rest[1] if len(rest) == 2 else "")
    elif sub == "units" and len(rest) == 1:
        cmd_units(rest[0])
    elif sub == "api" and len(rest) in (2, 3):
        cmd_api(rest[0], rest[1], rest[2] if len(rest) == 3 else "")
    else:
        raise SystemExit(usage)


if __name__ == "__main__":
    main()
