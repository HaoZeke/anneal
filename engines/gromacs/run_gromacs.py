#!/usr/bin/env python3
"""Run a two-atom GROMACS step whose only energy is the RGPOT bias.

GROMACS 2026 in this recipe has no NNPot module. It does accept
``gmx mdrun -plumed``, and the PLUMED collective variable calls rgpot.
The topology has zero Lennard-Jones and zero charge, so the potential
GROMACS prints is the bias.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


TOPOL = """
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
Ar 39.948 0.0 A 0.10 0.0

[ moleculetype ]
MOL 1

[ atoms ]
1 Ar 1 MOL Ar 1 0.0

[ system ]
two atoms

[ molecules ]
MOL 2
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


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_gromacs.py rgpot-lj libRgpotPlumed.so work")
    tool, plugin, work = sys.argv[1:]
    work_path = Path(work)
    work_path.mkdir(parents=True, exist_ok=True)
    factor = float(subprocess.check_output([tool, "--kJ-per-eV"], text=True).strip())
    # 1 eV well, 8 Angstrom cutoff, 1 Angstrom zero crossing.
    (work_path / "plumed.dat").write_text(
        "\n".join(
            [
                f"LOAD FILE={Path(plugin).resolve()}",
                "ene: RGPOT ATOMS=1,2 POTENTIAL=LJ OPTIONS={u0 1.0 psi 1.0 cutoff 8.0} TYPES=Ar",
                "b: BIASVALUE ARG=ene",
                "PRINT ARG=ene FILE=COLVAR FMT=%16.10f",
                "",
            ]
        )
    )
    (work_path / "topol.top").write_text(TOPOL)
    (work_path / "conf.gro").write_text(GRO)
    (work_path / "md.mdp").write_text(MDP)
    env = os.environ.copy()
    prefix = env.get("CONDA_PREFIX")
    if prefix:
        kernel = Path(prefix) / "lib" / "libplumedKernel.so"
        if kernel.is_file():
            env["PLUMED_KERNEL"] = str(kernel)
    subprocess.run(
        ["gmx", "grompp", "-f", "md.mdp", "-c", "conf.gro", "-p", "topol.top", "-o", "run.tpr",
         "-maxwarn", "2"],
        cwd=work_path, check=True, env=env,
    )
    subprocess.run(
        ["gmx", "mdrun", "-s", "run.tpr", "-deffnm", "run", "-plumed", "plumed.dat", "-nsteps", "0",
         "-ntmpi", "1"],
        cwd=work_path, check=True, env=env,
    )
    colvar = (work_path / "COLVAR").read_text().splitlines()
    values = [line for line in colvar if line and not line.startswith("#")]
    if not values:
        raise SystemExit("PLUMED wrote no energy")
    kj = float(values[-1].split()[-1])
    ev = kj / factor
    direct = subprocess.check_output(
        [tool], input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True,
    )
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(ev - ref) > 1e-6:
        raise SystemExit(f"gromacs bias {ev} eV, rgpot {ref} eV")
    print(f"gromacs bias {ev:.10f} eV")


if __name__ == "__main__":
    main()
