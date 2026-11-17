#!/usr/bin/env python3
"""GROMACS runs whose only energy is the RGPOT bias.

GROMACS 2026 in this recipe has no NNPot module. It does accept
``gmx mdrun -plumed``, and the PLUMED collective variable calls rgpot.
The topologies have zero Lennard-Jones and zero charge, so the potential
GROMACS prints is the bias.

The first run is one step of the LJ dimer, checked against ``rgpot-lj``.
The second is a short NPT run of a Sutton-Chen platinum crystal in a
triclinic box. Every frame's bias, virial and pressure are checked
against ``rgpot-eval`` on the positions PLUMED saw.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("SKIP: numpy is not importable")
    sys.exit(77)


KJ_PER_EV = 96.48533212331002
# kJ mol^-1 nm^-3 in bar.
BAR_PER_KJ_MOL_NM3 = 16.605390666
PT_MASS = 195.08
PT_SPEC = ["SuttonChen", "preset", "platinum", "cutoff", "5.0"]

TOPOL = """
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
{name} {mass} 0.0 A 0.10 0.0

[ moleculetype ]
MOL 1

[ atoms ]
1 {name} 1 MOL {name} 1 0.0

[ system ]
rgpot bias only

[ molecules ]
MOL {count}
"""

GRO = """two atoms
2
    1MOL     Ar    1   0.000   0.000   0.000
    2MOL     Ar    2   0.150   0.000   0.000
   4.00000   4.00000   4.00000
"""

MDP = """
integrator = md
dt = 0.001
nsteps = 0
continuation = yes
cutoff-scheme = Verlet
nstlist = 1
pbc = xyz
coulombtype = Cut-off
rcoulomb = 1.2
rvdw = 1.2
rlist = 1.2
vdw-modifier = None
DispCorr = no
"""

# md-vv reports the full-step kinetic energy, so each frame's pressure is
# 2/V (Ekin - Xi) with the velocities written for that frame.
NPT_MDP = """
integrator = md-vv
dt = 0.002
nsteps = 10
continuation = yes
cutoff-scheme = Verlet
nstlist = 10
pbc = xyz
coulombtype = Cut-off
rcoulomb = 0.6
rvdw = 0.6
vdw-modifier = None
DispCorr = no
nstcalcenergy = 1
nstenergy = 1
nstvout = 1
tcoupl = no
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 2.0
ref-p = 1.0
compressibility = 4.5e-6
ensemble-temperature-setting = constant
ensemble-temperature = 300
"""


def gmx_env() -> dict:
    env = os.environ.copy()
    prefix = env.get("CONDA_PREFIX")
    if prefix:
        kernel = Path(prefix) / "lib" / "libplumedKernel.so"
        if kernel.is_file():
            env["PLUMED_KERNEL"] = str(kernel)
    return env


def gmx(work: Path, *args: str, stdin: str | None = None) -> str:
    proc = subprocess.run(["gmx", *args], cwd=work, env=gmx_env(), input=stdin,
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"gmx {args[0]} failed\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def run(work: Path, mdp: str, gro: str, top: str, plumed: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*"):
        if old.is_file():
            old.unlink()
    (work / "md.mdp").write_text(mdp)
    (work / "conf.gro").write_text(gro)
    (work / "topol.top").write_text(top)
    (work / "plumed.dat").write_text(plumed)
    gmx(work, "grompp", "-f", "md.mdp", "-c", "conf.gro", "-p", "topol.top", "-o", "run.tpr",
        "-maxwarn", "5")
    gmx(work, "mdrun", "-s", "run.tpr", "-deffnm", "run", "-plumed", "plumed.dat", "-ntmpi", "1",
        "-ntomp", "1")


def colvar(work: Path) -> list[float]:
    rows = [line for line in (work / "COLVAR").read_text().splitlines()
            if line and not line.startswith("#")]
    if not rows:
        raise SystemExit("PLUMED wrote no energy")
    return [float(line.split()[1]) for line in rows]


def lj_dimer(tool: str, plugin: str, work: Path) -> None:
    plumed = "\n".join([
        f"LOAD FILE={Path(plugin).resolve()}",
        "ene: RGPOT ATOMS=1,2 POTENTIAL=LJ OPTIONS={u0 1.0 psi 1.0 cutoff 8.0} TYPES=Ar",
        "b: BIASVALUE ARG=ene",
        "PRINT ARG=ene FILE=COLVAR FMT=%.17g",
        "",
    ])
    run(work, MDP, GRO, TOPOL.format(name="Ar", mass=39.948, count=2), plumed)
    ev = colvar(work)[-1] / KJ_PER_EV
    direct = subprocess.check_output(
        [tool], input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True)
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(ev - ref) > 1e-6:
        raise SystemExit(f"gromacs bias {ev} eV, rgpot {ref} eV")
    print(f"LJ dimer: gromacs bias {ev:.10f} eV, rgpot {ref:.10f} eV")


def platinum_gro() -> tuple[str, int]:
    """A jiggled 4x4x4 fcc platinum crystal in a triclinic box, in nm."""
    a, n = 0.3924, 4
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) / n
    side = n * a
    h = np.array([[side, 0, 0], [0.10, side, 0], [0.06, -0.08, side]])
    pos = frac @ h + np.random.default_rng(3).uniform(-0.006, 0.006, size=frac.shape)
    lines = ["platinum", str(len(pos))]
    lines += [f"{i + 1:5d}MOL     Pt{(i + 1) % 100000:5d}{x:15.9f}{y:15.9f}{z:15.9f}"
              for i, (x, y, z) in enumerate(pos)]
    lines.append(f" {h[0, 0]:.9f} {h[1, 1]:.9f} {h[2, 2]:.9f} 0 0 {h[1, 0]:.9f} 0 "
                 f"{h[2, 0]:.9f} {h[2, 1]:.9f}")
    return "\n".join(lines) + "\n", len(pos)


def xyz_frames(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        cell = np.array([float(v) for v in lines[i + 1].split()]).reshape(3, 3)
        pos = np.array([[float(v) for v in row.split()[1:4]] for row in lines[i + 2:i + 2 + n]])
        yield cell, pos
        i += n + 2


def velocity_frames(dump: str) -> list[np.ndarray]:
    frames, current = [], None
    for line in dump.splitlines():
        if re.match(r"\s+v \(\d+x3\)", line):
            current = []
            frames.append(current)
        elif current is not None and line.strip().startswith("v["):
            current.append([float(t) for t in line.split("{")[1].split("}")[0].split(",")])
        else:
            current = None
    return [np.array(v) for v in frames]


def energy_tensors(work: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    terms = [f"{p}-{a}{b}" for p in ("Pres", "Vir") for a, b in
             ("XX", "YY", "ZZ", "XY", "XZ", "YZ")]
    gmx(work, "energy", "-f", "run.edr", "-o", "energy.xvg", stdin="\n".join(terms) + "\n")
    legend, rows = [], []
    for line in (work / "energy.xvg").read_text().splitlines():
        if line.startswith("@ s") and "legend" in line:
            legend.append(line.split('"')[1])
        elif line and line[0] not in "#@":
            rows.append([float(t) for t in line.split()])
    table = {name: [row[k + 1] for row in rows] for k, name in enumerate(legend)}

    def tensor(prefix: str, frame: int) -> np.ndarray:
        out = np.zeros((3, 3))
        for i, a in enumerate("XYZ"):
            for j, b in enumerate("XYZ"):
                key = f"{prefix}-{a}{b}" if f"{prefix}-{a}{b}" in table else f"{prefix}-{b}{a}"
                out[i, j] = table[key][frame]
        return out

    count = len(rows)
    return [tensor("Pres", k) for k in range(count)], [tensor("Vir", k) for k in range(count)]


def rgpot_eval(tool: str, cell_nm: np.ndarray, pos_nm: np.ndarray) -> tuple[float, np.ndarray]:
    text = [str(len(pos_nm)), " ".join(f"{v * 10:.17g}" for v in cell_nm.reshape(-1))]
    text += [f"78 {x * 10:.17g} {y * 10:.17g} {z * 10:.17g}" for x, y, z in pos_nm]
    out = subprocess.check_output([tool, *PT_SPEC], input="\n".join(text) + "\n", text=True)
    energy = stress = None
    for line in out.splitlines():
        if line.startswith("energy"):
            energy = float(line.split()[1])
        elif line.startswith("stress"):
            stress = np.array([float(v) for v in line.split()[1:]]).reshape(3, 3)
    if energy is None or stress is None:
        raise SystemExit(f"rgpot-eval printed no energy or stress\n{out}")
    return energy, stress


def platinum_npt(plugin: str, evaluator: str, work: Path) -> None:
    gro, count = platinum_gro()
    plumed = "\n".join([
        f"LOAD FILE={Path(plugin).resolve()}",
        f"ene: RGPOT ATOMS=1-{count} POTENTIAL={PT_SPEC[0]} "
        f"OPTIONS={{{' '.join(PT_SPEC[1:])}}} TYPES=Pt",
        "b: BIASVALUE ARG=ene",
        "PRINT ARG=ene FILE=COLVAR FMT=%.17g",
        f"DUMPATOMS ATOMS=1-{count} FILE=frames.xyz PRECISION=12",
        "",
    ])
    run(work, NPT_MDP, gro, TOPOL.format(name="Pt", mass=PT_MASS, count=count), plumed)
    bias = colvar(work)
    pressures, virials = energy_tensors(work)
    velocities = velocity_frames(gmx(work, "dump", "-f", "run.trr"))
    frames = list(xyz_frames(work / "frames.xyz"))
    if not (len(frames) == len(bias) == len(pressures) == len(velocities)):
        raise SystemExit(f"frame counts differ: {len(frames)} xyz, {len(bias)} COLVAR, "
                         f"{len(pressures)} energy, {len(velocities)} velocity")
    worst_e = worst_v = worst_p = 0.0
    for k, (cell, pos) in enumerate(frames):
        energy, stress = rgpot_eval(evaluator, cell, pos)
        volume = abs(np.linalg.det(cell))
        # GROMACS's virial is -1/2 sum r (x) F = V sigma / 2.
        virial = 0.5 * volume * 1000.0 * stress.T * KJ_PER_EV
        v = velocities[k]
        kinetic = 0.5 * PT_MASS * np.einsum("ia,ib->ab", v, v)
        pressure = BAR_PER_KJ_MOL_NM3 * 2.0 / volume * (kinetic - virial)
        de = abs(bias[k] - energy * KJ_PER_EV) / abs(energy * KJ_PER_EV)
        dv = np.max(np.abs(virials[k] - virial)) / np.max(np.abs(virial))
        dp = np.max(np.abs(pressures[k] - pressure))
        worst_e, worst_v, worst_p = max(worst_e, de), max(worst_v, dv), max(worst_p, dp)
        print(f"step {k:2d}: V {volume:.6f} nm^3, bias {bias[k]:.6f} kJ/mol "
              f"(rgpot {energy * KJ_PER_EV:.6f}), P {np.trace(pressures[k]) / 3:.3f} bar "
              f"(rgpot {np.trace(pressure) / 3:.3f})")
    print(f"worst: bias {worst_e:.2g} relative, virial {worst_v:.2g} relative, "
          f"pressure component {worst_p:.3g} bar")
    # Energies come from PLUMED's own double positions. The virial and
    # pressure pass through the single precision energy file, and the
    # kinetic term through six-digit velocities.
    if worst_e > 1e-10 or worst_v > 1e-5 or worst_p > 0.5:
        raise SystemExit("GROMACS and rgpot disagree on the platinum NPT run")


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: run_gromacs.py rgpot-lj libRgpotPlumed.so work rgpot-eval")
    tool, plugin, work, evaluator = sys.argv[1:]
    lj_dimer(tool, plugin, Path(work) / "lj")
    platinum_npt(plugin, evaluator, Path(work) / "platinum")


if __name__ == "__main__":
    main()
