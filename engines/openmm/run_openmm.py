#!/usr/bin/env python3
"""Compare OpenMM's shifted 12-6 force with rgpot on one pair.

OpenMM has no callback force. The expression below is the same shifted
Lennard-Jones potential rgpot evaluates, in OpenMM's nm and kJ/mol.
A difference here is a difference in the potential, not in the host.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import openmm as mm
from openmm import unit


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_openmm.py rgpot-lj")
    tool = sys.argv[1]
    factor = float(subprocess.check_output([tool, "--kJ-per-eV"], text=True).strip())
    eps = factor  # 1 eV in kJ/mol
    sig = 0.1  # 1 Angstrom in nm
    rc = 0.8
    expression = (
        "4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/r)^12-(sqrt(sig1*sig2)/r)^6)"
        "-4*sqrt(eps1*eps2)*((sqrt(sig1*sig2)/rc)^12-(sqrt(sig1*sig2)/rc)^6)"
    )
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = mm.CustomNonbondedForce(expression)
    force.addGlobalParameter("rc", rc)
    force.addPerParticleParameter("eps")
    force.addPerParticleParameter("sig")
    force.addParticle([eps, sig])
    force.addParticle([eps, sig])
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(rc * unit.nanometer)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(
        (2, 0, 0) * unit.nanometer, (0, 2, 0) * unit.nanometer, (0, 0, 2) * unit.nanometer
    )
    context = mm.Context(system, mm.VerletIntegrator(0.001))
    context.setPositions([[0, 0, 0], [0.15, 0, 0]] * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    ev = kj / factor
    direct = subprocess.check_output(
        [tool],
        input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True,
    )
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(ev - ref) > 1e-6:
        raise SystemExit(f"openmm {ev} eV, rgpot {ref} eV")
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    # eV/Angstrom = (kJ/mol/nm) / factor / 10
    fx = forces[1][0] / factor / 10.0
    ref_force = float(
        next(line.split()[2] for line in direct.splitlines() if line.startswith("force 1"))
    )
    if abs(fx - ref_force) > 1e-5:
        raise SystemExit(f"openmm force {fx}, rgpot {ref_force}")
    print(f"openmm energy {ev:.10f} eV")


if __name__ == "__main__":
    main()
