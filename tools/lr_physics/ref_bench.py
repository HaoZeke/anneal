#!/usr/bin/env python3
"""Same-host timings for s-dftd3, dftd4 and OpenMM PME.

The geometry, the real-space cutoffs and the repeat count match
lr_phys_bench. Each line records the hostname, the date and the CPU.
"""

import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
from dftd3.interface import DispersionModel, RationalDampingParam
from dftd4.interface import DampingParam, DispersionModel as D4Model
from openmm import Context, NonbondedForce, System, Vec3, VerletIntegrator
from openmm.unit import angstrom, nanometer, picosecond

BOHR = 0.529177210903
NREP = 6


def quantile7(x, p):
    x = np.sort(np.asarray(x, dtype=float))
    h = (len(x) - 1) * p
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return float(x[lo])
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def system(n, seed):
    side = n ** (1.0 / 3.0) * 3.2
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.4, side - 0.4, size=(n, 3))
    z = np.where(np.arange(n) % 2 == 0, 8, 1)
    q = np.where(np.arange(n) % 2 == 0, 0.4, -0.4)
    q = q.copy()
    q[0] -= q.sum()
    box = np.diag([side, side, side])
    return pos, z, q, box


def samples(fn):
    fn()
    out = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def report(name, n, values):
    med = quantile7(values, 0.5)
    iqr = quantile7(values, 0.75) - quantile7(values, 0.25)
    host = socket.gethostname()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cpu = platform.processor() or "unknown"
    print(
        f"run hostname={host} date={date} cpu={cpu}",
        flush=True,
    )
    print(
        f"ref {name} n={n} repeats={len(values)} median_us={med:.6g} iqr7_us={iqr:.6g}",
        flush=True,
    )


def time_dftd3(pos, z, box):
    model = DispersionModel(z, pos / BOHR, lattice=box / BOHR, periodic=np.array([True, True, True]))
    model.set_realspace_cutoff(15.0, 10.0, 10.0)
    param = RationalDampingParam(method="pbe", atm=True)

    def once():
        model.get_dispersion(param, grad=True)

    report("s-dftd3", len(z), samples(once))


def time_dftd4(pos, z, box):
    model = D4Model(z, pos / BOHR, charge=0.0, lattice=box / BOHR, periodic=np.array([True, True, True]))
    model.set_realspace_cutoff(15.0, 10.0, 10.0)
    param = DampingParam(method="pbe", atm=True)

    def once():
        model.get_dispersion(param, grad=True)

    report("dftd4", len(z), samples(once))


def time_openmm(pos, q, box):
    system = System()
    side = float(box[0, 0])
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(0.45 * nanometer)
    nb.setEwaldErrorTolerance(1.0e-5)
    for qi in q:
        system.addParticle(1.0)
        nb.addParticle(float(qi), 1.0, 0.0)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        Vec3(side, 0, 0) * angstrom,
        Vec3(0, side, 0) * angstrom,
        Vec3(0, 0, side) * angstrom,
    )
    ctx = Context(system, VerletIntegrator(0.001 * picosecond))
    ctx.setPositions(pos * angstrom)

    def once():
        ctx.getState(getEnergy=True, getForces=True)

    report("openmm-pme", len(q), samples(once))


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [128, 512, 2048, 8192]
    print(f"cpu {platform.processor()}", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    print("disp_cutoff_bohr disp2=15 disp3=10 cn=10", flush=True)
    for n in sizes:
        pos, z, q, box = system(n, 1000 + n)
        time_openmm(pos, q, box)
        time_dftd3(pos, z, box)
        time_dftd4(pos, z, box)


if __name__ == "__main__":
    main()
