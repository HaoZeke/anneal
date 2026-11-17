"""Shared pieces of the librgpot_gromacs tests and benchmark.

:func:`write_system` writes a GROMACS run whose QM group is evaluated by
rgpot, :func:`run` runs grompp and mdrun, :func:`read_trr` and
:func:`energies` read the results at full precision, and
:func:`rgpot_eval` is the direct rgpot evaluation they are compared with.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# rgpot::md::kKjPerEv, the SI definition GROMACS also uses (c_faraday / c_kilo).
KJ_PER_EV = 1.602176634e-19 * 6.02214076e23 / 1000.0

# GROMACS' src/gromacs/math/units.h, written as there so the bits match.
C_AVOGADRO = 6.02214076e23
C_PLANCK = 6.62607015e-34 * C_AVOGADRO / (1e-12 * 1e3)
C_HARTREE2KJ = (2.0 * 1.0973731568160e-02 * C_PLANCK * 2.99792458e05) / C_AVOGADRO
C_BOHR2NM = 0.0529177210903
C_HARTREEBOHR2MD = C_HARTREE2KJ * C_AVOGADRO / C_BOHR2NM

SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
           "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
           "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
           "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
           "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
           "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]
MASSES = {"Ar": 39.948, "Si": 28.0855, "Pt": 195.08, "Cu": 63.546, "Ni": 58.6934}


def qmi(rgpot: str, charge: int = 0, multiplicity: int = 1, pdb: str = "rgpot.pdb") -> str:
    """A -qmi file: the keys grompp checks and an &RGPOT section."""
    return f"""&GLOBAL
  RUN_TYPE ENERGY_FORCE
&END GLOBAL
&FORCE_EVAL
  &DFT
    CHARGE {charge}
    MULTIPLICITY {multiplicity}
  &END DFT
  &SUBSYS
    &TOPOLOGY
      COORD_FILE_NAME {pdb}
      COORD_FILE_FORMAT PDB
      CHARGE_EXTENDED TRUE
    &END TOPOLOGY
  &END SUBSYS
  &RGPOT
{rgpot}
  &END RGPOT
&END FORCE_EVAL
"""


def gmx_problem() -> tuple[str | None, str]:
    """The gmx to run, or None and why it cannot run."""
    gmx = os.environ.get("RGPOT_GMX") or shutil.which("gmx_d")
    if not gmx:
        return None, "no GROMACS: set RGPOT_GMX or put gmx_d (GMX_DOUBLE, GMX_CP2K) on PATH"
    try:
        done = subprocess.run([gmx, "-quiet", "--version"], capture_output=True, text=True,
                              timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{gmx} --version failed: {exc}"
    out = done.stdout + done.stderr
    if done.returncode != 0:
        return None, f"{gmx} --version failed: {out.strip()[-300:]}"
    if "double" not in next((line for line in out.splitlines() if "Precision" in line), ""):
        return None, f"{gmx} is not a double-precision build (GMX_DOUBLE=ON)"
    return gmx, ""


def gmx_version(gmx: str) -> str:
    out = subprocess.run([gmx, "-quiet", "--version"], capture_output=True, text=True,
                         check=False).stdout
    return next((line.split(":", 1)[1].strip() for line in out.splitlines()
                 if line.startswith("GROMACS version")), "unknown")


def gro_box(rows: np.ndarray) -> str:
    """The .gro box line: v1x v2y v3z v1y v1z v2x v2z v3x v3y."""
    r = np.asarray(rows, dtype=float)
    values = [r[0, 0], r[1, 1], r[2, 2], r[0, 1], r[0, 2], r[1, 0], r[1, 2], r[2, 0], r[2, 1]]
    if not any(values[3:]):
        values = values[:3]
    return " ".join(f"{v:.12f}" for v in values)


def write_system(work: Path, *, symbols: Sequence[str], positions: np.ndarray,
                 box: np.ndarray, qmi_text: str | None, mdp: dict[str, Any],
                 qm: Sequence[int] | None = None, mm_charges: dict[int, float] | None = None,
                 velocities: np.ndarray | None = None) -> None:
    """One single-atom molecule per atom, with zero Lennard-Jones and charge.

    ``qm`` lists the QM atoms (0-based, default all). GROMACS adds nothing
    of its own, so the rgpot energy is the whole potential and an MM atom
    feels no force. ``mm_charges`` gives chosen MM atoms a charge.
    """
    work.mkdir(parents=True, exist_ok=True)
    for old in work.iterdir():
        if old.is_file():
            old.unlink()
    x = np.asarray(positions, dtype=float)
    n = len(x)
    qm = list(range(n)) if qm is None else sorted(qm)
    mm_charges = mm_charges or {}
    lines = ["rgpot drop-in", str(n)]
    for i, row in enumerate(x):
        name = symbols[i]
        line = f"{(i + 1) % 100000:5d}{name:<5s}{name:>5s}{(i + 1) % 100000:5d}"
        line += "".join(f"{v:20.12f}" for v in row)
        if velocities is not None:
            line += "".join(f"{v:20.12f}" for v in velocities[i])
        lines.append(line)
    lines.append(gro_box(box))
    (work / "conf.gro").write_text("\n".join(lines) + "\n")

    kinds = sorted({(s, mm_charges.get(i, 0.0)) for i, s in enumerate(symbols)})
    names = {kind: f"{kind[0]}{k}" for k, kind in enumerate(kinds)}
    top = ["[ defaults ]", "1 2 yes 0.5 0.8333", "", "[ atomtypes ]"]
    for el in sorted({s for s in symbols}):
        top.append(f"{el} {SYMBOLS.index(el)} {MASSES[el]} 0.0 A 0.0 0.0")
    for kind, mol in names.items():
        top += ["", "[ moleculetype ]", f"{mol} 1", "", "[ atoms ]",
                f"1 {kind[0]} 1 {mol} {kind[0]} 1 {kind[1]}"]
    top += ["", "[ system ]", "rgpot drop-in", "", "[ molecules ]"]
    run_name, run_len = None, 0
    for i, s in enumerate(symbols):
        mol = names[(s, mm_charges.get(i, 0.0))]
        if mol == run_name:
            run_len += 1
            continue
        if run_name is not None:
            top.append(f"{run_name} {run_len}")
        run_name, run_len = mol, 1
    top.append(f"{run_name} {run_len}")
    (work / "topol.top").write_text("\n".join(top) + "\n")

    def group(name: str, atoms: Sequence[int]) -> str:
        body = [" ".join(str(a + 1) for a in atoms[k:k + 15]) for k in range(0, len(atoms), 15)]
        return f"[ {name} ]\n" + "\n".join(body) + "\n"

    (work / "index.ndx").write_text(group("System", list(range(n))) + group("QM", qm))
    settings: dict[str, Any] = {
        "integrator": "md",
        "nsteps": 0,
        "dt": 0.001,
        "cutoff-scheme": "Verlet",
        "pbc": "xyz",
        "coulombtype": "Cut-off",
        "rcoulomb": 0.4,
        "rvdw": 0.4,
        "rlist": 0.4,
        "verlet-buffer-tolerance": -1,
        "vdw-modifier": "None",
        "DispCorr": "no",
        "nstxout": 1,
        "nstvout": 1,
        "nstfout": 1,
        "nstenergy": 1,
        "nstcalcenergy": 1,
        "tcoupl": "no",
        "pcoupl": "no",
        "continuation": "yes",
        "qmmm-cp2k-active": "true",
        "qmmm-cp2k-qmgroup": "QM",
        "qmmm-cp2k-qmmethod": "INPUT",
    }
    settings.update(mdp)
    (work / "md.mdp").write_text("".join(f"{k} = {v}\n" for k, v in settings.items()))
    if qmi_text is not None:
        (work / "rgpot.inp").write_text(qmi_text)


def run(gmx: str, work: Path, *, check: bool = True, ntomp: int = 1,
        extra: Sequence[str] = ()) -> subprocess.CompletedProcess:
    """grompp, then mdrun on one thread-MPI rank; returns mdrun's process."""
    env = dict(os.environ, GMX_MAXBACKUP="-1")
    with_qmi = ["-qmi", "rgpot.inp"] if (work / "rgpot.inp").exists() else []
    grompp = subprocess.run(
        [gmx, "-quiet", "grompp", "-f", "md.mdp", "-c", "conf.gro", "-p", "topol.top",
         "-n", "index.ndx", *with_qmi, "-o", "md.tpr", "-maxwarn", "10"],
        cwd=work, capture_output=True, text=True, env=env, check=False)
    if grompp.returncode != 0:
        raise RuntimeError(f"grompp failed:\n{grompp.stdout}\n{grompp.stderr}")
    mdrun = subprocess.run(
        [gmx, "-quiet", "mdrun", "-deffnm", "md", "-ntmpi", "1", "-ntomp", str(ntomp),
         "-nb", "cpu", *extra],
        cwd=work, capture_output=True, text=True, env=env, check=False)
    if check and mdrun.returncode != 0:
        raise RuntimeError(f"mdrun failed:\n{mdrun.stdout}\n{mdrun.stderr}")
    return mdrun


def read_trr(path: Path) -> list[dict[str, Any]]:
    """Frames of a .trr: step, box rows, x, v, f (nm, kJ/mol/nm)."""
    data = path.read_bytes()
    pos = 0

    def ints(n: int) -> tuple[int, ...]:
        nonlocal pos
        out = struct.unpack_from(f">{n}i", data, pos)
        pos += 4 * n
        return out

    frames = []
    while pos < len(data):
        magic, _slen, length = ints(3)
        if magic != 1993:
            raise ValueError(f"{path}: not a trr frame at byte {pos}")
        pos += (length + 3) // 4 * 4
        (_ir, _e, box_size, vir_size, pres_size, _top, _sym, x_size, v_size, f_size,
         natoms, step, _nre) = ints(13)
        size = 8 if box_size == 72 or 24 * natoms in (x_size, v_size, f_size) else 4
        fmt = ">f8" if size == 8 else ">f4"

        def reals(n: int) -> np.ndarray:
            nonlocal pos
            out = np.frombuffer(data, dtype=fmt, count=n, offset=pos).astype(float)
            pos += size * n
            return out

        reals(2)
        frame: dict[str, Any] = {"step": step, "double": size == 8}
        if box_size:
            frame["box"] = reals(9).reshape(3, 3)
        if vir_size:
            reals(9)
        if pres_size:
            reals(9)
        for key, nbytes in (("x", x_size), ("v", v_size), ("f", f_size)):
            if nbytes:
                frame[key] = reals(3 * natoms).reshape(natoms, 3)
        frames.append(frame)
    return frames


def energies(gmx: str, work: Path, *terms: str) -> dict[str, np.ndarray]:
    """The named terms of md.edr, printed by gmx energy -dp."""
    done = subprocess.run(
        [gmx, "-quiet", "energy", "-f", "md.edr", "-o", "energy.xvg", "-dp"],
        cwd=work, input="\n".join(terms) + "\n\n", capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"gmx energy failed:\n{done.stdout}\n{done.stderr}")
    legend, rows = [], []
    for line in (work / "energy.xvg").read_text().splitlines():
        if line.startswith("@ s") and "legend" in line:
            legend.append(line.split('"')[1])
        elif line and line[0] not in "#@":
            rows.append([float(v) for v in line.split()])
    table = np.array(rows)
    return {name: table[:, k + 1] for k, name in enumerate(legend)}


def rgpot_eval(tool: str, spec: Sequence[str], symbols: Sequence[str], x_ang: np.ndarray,
               box_ang: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy (eV) and forces (eV/Angstrom) from rgpot-eval, the direct path."""
    text = [str(len(x_ang)), " ".join(f"{v:.17g}" for v in np.asarray(box_ang).reshape(-1))]
    text += [f"{SYMBOLS.index(s)} {a:.17g} {b:.17g} {c:.17g}"
             for s, (a, b, c) in zip(symbols, x_ang)]
    out = subprocess.run([tool, *spec], input="\n".join(text) + "\n", capture_output=True,
                         text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"rgpot-eval failed: {out.stderr}")
    energy = None
    forces = np.zeros((len(x_ang), 3))
    for line in out.stdout.splitlines():
        w = line.split()
        if w[0] == "energy":
            energy = float(w[1])
        elif w[0] == "force":
            forces[int(w[1])] = [float(v) for v in w[2:5]]
    if energy is None:
        raise RuntimeError(f"rgpot-eval printed no energy\n{out.stdout}")
    return energy, forces


def wrap(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Positions put into the cell whose lattice vectors are ``rows``."""
    s = np.asarray(x) @ np.linalg.inv(rows)
    return (s - np.floor(s)) @ rows


def fcc(a: float, n: int, jiggle: float = 0.0, seed: int = 3,
        tilt: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """An n x n x n fcc crystal (nm); ``tilt`` is (b_x, c_x, c_y) of the box."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    side = n * a
    box = np.array([[side, 0, 0], [tilt[0], side, 0], [tilt[1], tilt[2], side]])
    x = frac @ box
    if jiggle:
        x = x + np.random.default_rng(seed).uniform(-jiggle, jiggle, size=x.shape)
    return wrap(x, box), box


def diamond(a: float, n: int, jiggle: float = 0.0, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """An n x n x n diamond crystal (nm) in a cubic box."""
    fcc_basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    basis = np.vstack([fcc_basis, fcc_basis + 0.25])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    box = np.eye(3) * n * a
    x = frac @ box
    if jiggle:
        x = x + np.random.default_rng(seed).uniform(-jiggle, jiggle, size=x.shape)
    return wrap(x, box), box
