#!/usr/bin/env python3
"""Same-host comparator for OpenMM PME, s-dftd3 and dftd4.

One geometry file per size is shared with the rgpot ladder. Each arm
runs in its own process. A logged warmup is followed by six steady
repeats. dftd4 is timed alone. A kill at the limit is reported as the
limit and the exit code, with no speedup.
"""

import hashlib
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BOHR = 0.529177210903
NREP = 6
CUTOFF_A = 8.0
DISP_CUT = (15.0, 10.0, 10.0)


def cpu_model():
    with open("/proc/cpuinfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile7(values, p):
    x = np.sort(np.asarray(values, dtype=float))
    h = (len(x) - 1) * p
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return float(x[lo])
    return float(x[lo] + (h - lo) * (x[hi] - x[lo]))


def load_geometry(path):
    text = Path(path).read_text(encoding="utf-8").split()
    n = int(text[0])
    box = np.array([float(v) for v in text[1:10]], dtype=float).reshape(3, 3)
    rest = text[10:]
    z = np.empty(n, dtype=int)
    pos = np.empty((n, 3), dtype=float)
    q = np.empty(n, dtype=float)
    for i in range(n):
        base = i * 5
        z[i] = int(rest[base])
        pos[i, 0] = float(rest[base + 1])
        pos[i, 1] = float(rest[base + 2])
        pos[i, 2] = float(rest[base + 3])
        q[i] = float(rest[base + 4])
    return pos, z, q, box


def header(name, n, threads, geom_sha):
    host = socket.gethostname()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"run hostname={host} date={date} cpu={cpu_model()}", flush=True)
    print(f"arm {name} n={n} threads={threads} geometry_sha256={geom_sha}", flush=True)


def time_calls(fn):
    t0 = time.perf_counter()
    fn()
    warmup = (time.perf_counter() - t0) * 1e6
    samples = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return warmup, samples


def report_times(name, n, threads, warmup, samples):
    med = quantile7(samples, 0.5)
    iqr = quantile7(samples, 0.75) - quantile7(samples, 0.25)
    print(f"warmup_us={warmup:.17g}", flush=True)
    print(
        f"ref {name} n={n} threads={threads} repeats={len(samples)} "
        f"median_us={med:.17g} iqr7_us={iqr:.17g}",
        flush=True,
    )
    print("samples_us " + " ".join(f"{v:.17g}" for v in samples), flush=True)


def library_hashes():
    import dftd3
    import dftd4
    import openmm

    roots = {
        "dftd3": Path(dftd3.__file__).resolve().parent,
        "dftd4": Path(dftd4.__file__).resolve().parent,
        "openmm": Path(openmm.__file__).resolve().parent,
    }
    lines = []
    for name, root in roots.items():
        so = sorted(root.rglob("*.so"))
        if not so:
            lines.append(f"{name}_sha256=missing")
            continue
        digest = sha256_file(so[0])
        lines.append(f"{name}_sha256={digest} path={so[0]}")
    return lines


def versions():
    import dftd3
    import dftd4
    import openmm
    from dftd3.library import get_api_version as d3_api
    from dftd4.library import get_api_version as d4_api

    return (
        f"openmm_version={openmm.Platform.getOpenMMVersion()} "
        f"s-dftd3_version={d3_api()} dftd4_version={d4_api()}"
    )


def openmm_context(pos, q, box, threads, tolerance):
    from openmm import Context, NonbondedForce, Platform, System, Vec3, VerletIntegrator
    from openmm.unit import angstrom, nanometer, picosecond

    system = System()
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(CUTOFF_A * 0.1 * nanometer)
    nb.setEwaldErrorTolerance(tolerance)
    for qi in q:
        system.addParticle(1.0)
        nb.addParticle(float(qi), 1.0, 0.0)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(
        Vec3(*box[0]) * angstrom, Vec3(*box[1]) * angstrom, Vec3(*box[2]) * angstrom
    )
    platform = Platform.getPlatformByName("CPU")
    ctx = Context(system, VerletIntegrator(0.001 * picosecond), platform, {"Threads": str(threads)})
    ctx.setPositions(pos * angstrom)
    actual = ctx.getPlatform().getPropertyValue(ctx, "Threads")
    alpha, nx, ny, nz = nb.getPMEParametersInContext(ctx)
    return ctx, nb, actual, (alpha, nx, ny, nz)


def forces_from_state(ctx):
    from openmm.unit import kilojoule_per_mole, nanometer

    state = ctx.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    # kJ/mol/nm to eV/A. 1 eV = 96.485 kJ/mol, 1 nm = 10 A.
    raw = state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole / nanometer)
    forces = np.asarray(raw, dtype=float) / 96.48533212331002 / 10.0
    return energy / 96.48533212331002, forces


def arm_openmm(path, threads):
    pos, _z, q, box = load_geometry(path)
    geom_sha = sha256_file(path)
    header("openmm-pme", len(q), threads, geom_sha)
    print(versions(), flush=True)
    for line in library_hashes():
        print(line, flush=True)
    ctx, _nb, actual, pme = openmm_context(pos, q, box, threads, 1.0e-4)
    print(
        f"openmm_platform={ctx.getPlatform().getName()} threads_property={actual} "
        f"cutoff_A={CUTOFF_A} ewald_tolerance=1e-4 "
        f"pme_alpha={pme[0]:.8g} grid={pme[1]} {pme[2]} {pme[3]}",
        flush=True,
    )

    def once():
        ctx.getState(getEnergy=True, getForces=True)

    warmup, samples = time_calls(once)
    report_times("openmm-pme", len(q), threads, warmup, samples)
    _e, prod = forces_from_state(ctx)
    ref_ctx, _nb2, _actual2, ref_pme = openmm_context(pos, q, box, threads, 1.0e-8)
    _er, ref = forces_from_state(ref_ctx)
    err = float(np.max(np.abs(prod - ref)))
    print(
        f"force_error_eV_A={err:.17g} reference=openmm-pme tolerance=1e-8 "
        f"ref_grid={ref_pme[1]} {ref_pme[2]} {ref_pme[3]}",
        flush=True,
    )


def arm_dftd3(path, threads):
    from dftd3.interface import DispersionModel, RationalDampingParam

    os.environ["OMP_NUM_THREADS"] = str(threads)
    pos, z, _q, box = load_geometry(path)
    geom_sha = sha256_file(path)
    header("s-dftd3", len(z), threads, geom_sha)
    print(versions(), flush=True)
    for line in library_hashes():
        print(line, flush=True)
    model = DispersionModel(z, pos / BOHR, lattice=box / BOHR, periodic=np.array([True, True, True]))
    model.set_realspace_cutoff(*DISP_CUT)
    param = RationalDampingParam(method="pbe", atm=True)

    def once():
        model.get_dispersion(param, grad=True)

    warmup, samples = time_calls(once)
    report_times("s-dftd3", len(z), threads, warmup, samples)


def arm_dftd4(path, threads):
    from dftd4.interface import DampingParam, DispersionModel

    os.environ["OMP_NUM_THREADS"] = str(threads)
    pos, z, _q, box = load_geometry(path)
    geom_sha = sha256_file(path)
    header("dftd4", len(z), threads, geom_sha)
    print(versions(), flush=True)
    for line in library_hashes():
        print(line, flush=True)
    model = DispersionModel(
        z, pos / BOHR, charge=0.0, lattice=box / BOHR, periodic=np.array([True, True, True])
    )
    model.set_realspace_cutoff(*DISP_CUT)
    param = DampingParam(method="pbe", atm=True)

    def once():
        model.get_dispersion(param, grad=True)

    warmup, samples = time_calls(once)
    report_times("dftd4", len(z), threads, warmup, samples)


def rgpot_forces(binary, kind, path):
    env = os.environ.copy()
    env["LR_DUMP"] = kind
    env["LR_GEOM"] = path
    env["OMP_NUM_THREADS"] = "1"
    proc = subprocess.run([binary, "6"], check=True, capture_output=True, text=True, env=env)
    energy = None
    forces = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts[0] == "energy_eV":
            energy = float(parts[1])
        elif parts[0] == "force_eV_A":
            forces.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return energy, np.array(forces, dtype=float)


def certificate(binary, path, threads):
    pos, _z, q, box = load_geometry(path)
    ref_ctx, _nb, _actual, ref_pme = openmm_context(pos, q, box, threads, 1.0e-8)
    _er, ref = forces_from_state(ref_ctx)
    _energy, ours = rgpot_forces(binary, "pme", path)
    err = float(np.max(np.abs(ours - ref)))
    print(
        f"rgpot_pme_force_error_eV_A={err:.17g} reference=openmm-pme tolerance=1e-8 "
        f"ref_grid={ref_pme[1]} {ref_pme[2]} {ref_pme[3]} n={len(q)}",
        flush=True,
    )


def limit_for(arm, n):
    if arm == "dftd4" and n >= 4096:
        return 1800
    if arm == "dftd4":
        return 600
    if n >= 4096:
        return 900
    return 300


def drive(geom_dir, binary, threads_list):
    print(f"cpu {cpu_model()}", flush=True)
    print(f"hostname {socket.gethostname()}", flush=True)
    print(versions(), flush=True)
    for line in library_hashes():
        print(line, flush=True)
    print("disp_cutoff_bohr disp2=15 disp3=10 cn=10", flush=True)
    print(f"openmm_cutoff_A {CUTOFF_A} openmm_ewald_tolerance 1e-4", flush=True)
    files = sorted(Path(geom_dir).glob("n*.txt"), key=lambda p: int(p.stem[1:]))
    script = str(Path(__file__).resolve())
    for path in files:
        n = int(path.stem[1:])
        digest = sha256_file(path)
        print(f"geometry n={n} sha256={digest} path={path}", flush=True)
        for threads in threads_list:
            for arm in ("openmm", "dftd3", "dftd4"):
                limit = limit_for(arm, n)
                cmd = [
                    "timeout",
                    "--signal=KILL",
                    str(limit),
                    sys.executable,
                    script,
                    "arm",
                    arm,
                    str(path),
                    str(threads),
                ]
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    print(
                        f"{arm} n={n} threads={threads} limit_s={limit} exit_code={proc.returncode}",
                        flush=True,
                    )
        if binary:
            certificate(binary, str(path), threads_list[0])


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "arm":
        arm, path, threads = sys.argv[2], sys.argv[3], int(sys.argv[4])
        if arm == "openmm":
            arm_openmm(path, threads)
        elif arm == "dftd3":
            arm_dftd3(path, threads)
        elif arm == "dftd4":
            arm_dftd4(path, threads)
        else:
            raise SystemExit(f"unknown arm {arm}")
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "drive":
        geom_dir = sys.argv[2]
        binary = sys.argv[3] if len(sys.argv) > 3 else ""
        threads = [int(x) for x in sys.argv[4:]] or [1, 2, 4, 8]
        drive(geom_dir, binary, threads)
        return
    raise SystemExit("usage: ref_bench.py drive GEOM_DIR [BINARY] [threads...] | arm NAME GEOM THREADS")


if __name__ == "__main__":
    main()
