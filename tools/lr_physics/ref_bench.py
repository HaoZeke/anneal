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


def load_fields():
    one, five, fifteen = Path("/proc/loadavg").read_text().split()[:3]
    return f"load1={float(one):.17g} load5={float(five):.17g} load15={float(fifteen):.17g}"


def report_times(name, n, threads, warmup, samples):
    med = quantile7(samples, 0.5)
    iqr = quantile7(samples, 0.75) - quantile7(samples, 0.25)
    print(f"warmup_us={warmup:.17g}", flush=True)
    print(load_fields(), flush=True)
    print(
        f"ref {name} n={n} threads={threads} repeats={len(samples)} "
        f"median_us={med:.17g} iqr7_us={iqr:.17g} {load_fields()}",
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


def openmm_context(pos, q, box, threads, tolerance, platform_name="CPU"):
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
    platform = Platform.getPlatformByName(platform_name)
    props = {"Threads": str(threads)} if platform_name == "CPU" else {}
    ctx = Context(system, VerletIntegrator(0.001 * picosecond), platform, props)
    ctx.setPositions(pos * angstrom)
    # The parameter query casts the PME implementation. One energy
    # evaluation creates it; querying first throws bad_cast.
    ctx.getState(getEnergy=True)
    if platform_name == "CPU":
        actual = ctx.getPlatform().getPropertyValue(ctx, "Threads")
    else:
        actual = platform_name
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
    del ctx


def native_team():
    """Observed team size inside a live parallel region, not the configured maximum."""
    import ctypes

    lib = ctypes.CDLL("libgomp.so.1")
    lib.omp_get_num_threads.restype = ctypes.c_int
    lib.GOMP_parallel.argtypes = [
        ctypes.CFUNCTYPE(None, ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    slot = ctypes.c_int(1)

    def worker(arg):
        ctypes.cast(arg, ctypes.POINTER(ctypes.c_int))[0] = lib.omp_get_num_threads()

    lib.GOMP_parallel(ctypes.CFUNCTYPE(None, ctypes.c_void_p)(worker), ctypes.byref(slot), 0, 0)
    return int(slot.value)


def arm_dftd3(path, threads):
    from dftd3.interface import DispersionModel, RationalDampingParam

    os.environ["OMP_NUM_THREADS"] = str(threads)
    print(f"omp_num_threads={os.environ['OMP_NUM_THREADS']} native_team={native_team()}", flush=True)
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
    print(f"omp_num_threads={os.environ['OMP_NUM_THREADS']} native_team={native_team()}", flush=True)
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


# The mesh rule targets 1e-4 eV/A. The tight arm is OpenMM's Reference
# platform at that tolerance. The CPU platform is recorded too: its
# force table drifts on close pairs and is not the acceptance target.
FORCE_LIMIT_EV_A = 1.0e-4


def write_force_table(path, ours, cpu, ref):
    force_dir = Path(os.environ.get("LR_FORCE_DIR", str(Path(path).resolve().parent / "forces")))
    force_dir.mkdir(parents=True, exist_ok=True)
    out_path = force_dir / f"{Path(path).stem}.txt"
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("rgpot_x rgpot_y rgpot_z cpu_x cpu_y cpu_z ref_x ref_y ref_z\n")
        for a, b, c in zip(ours, cpu, ref):
            handle.write(
                " ".join(f"{v:.17g}" for v in (a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]))
                + "\n"
            )
    return out_path


def certificate(binary, path, threads):
    pos, _z, q, box = load_geometry(path)
    ref_ctx, _nb, _actual, ref_pme = openmm_context(pos, q, box, threads, 1.0e-4, "Reference")
    cpu_ctx, _nb2, _actual2, cpu_pme = openmm_context(pos, q, box, threads, 1.0e-4, "CPU")
    _er, ref = forces_from_state(ref_ctx)
    _ec, cpu = forces_from_state(cpu_ctx)
    _energy, ours = rgpot_forces(binary, "pme", path)
    err = float(np.max(np.abs(ours - ref)))
    own = float(np.max(np.abs(cpu - ref)))
    table = write_force_table(path, ours, cpu, ref)
    print(
        f"rgpot_pme_force_error_eV_A={err:.17g} reference=openmm-reference tolerance=1e-4 "
        f"ref_grid={ref_pme[1]} {ref_pme[2]} {ref_pme[3]} "
        f"cpu_grid={cpu_pme[1]} {cpu_pme[2]} {cpu_pme[3]} "
        f"openmm_cpu_error_eV_A={own:.17g} n={len(q)} forces={table}",
        flush=True,
    )
    del ref_ctx, cpu_ctx
    if not np.isfinite(err) or err > FORCE_LIMIT_EV_A:
        raise SystemExit(f"pme force error {err:.6g} exceeds {FORCE_LIMIT_EV_A}")
    print("pme_force_check pass", flush=True)


def limit_for(arm, n):
    if arm == "dftd4" and n >= 4096:
        return 1800
    if arm == "dftd4":
        return 600
    if n >= 4096:
        return 900
    return 300


def time_rgpot_pme(binary, path, threads, tol):
    """Time rgpot on the accuracy-rule mesh for one force error."""
    env = os.environ.copy()
    env["LR_PME_TOL"] = f"{tol:.17g}"
    env["LR_LADDERS"] = "strong"
    env["LR_METHODS"] = "pme"
    n = int(Path(path).stem[1:])
    env["LR_NMIN"] = str(n)
    env["LR_NMAX"] = str(n)
    env["LR_TMIN"] = str(threads)
    env["LR_TMAX"] = str(threads)
    env["LR_GEOM_DIR"] = str(Path(path).resolve().parent)
    env.pop("OMP_NUM_THREADS", None)
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    proc = subprocess.run([binary, "6"], check=True, capture_output=True, text=True, env=env)
    median = None
    grid = None
    samples = None
    for line in proc.stdout.splitlines():
        if line.startswith("strong pme "):
            for part in line.split():
                if part.startswith("median_us="):
                    median = float(part.split("=", 1)[1])
        if " grid=" in line and line.startswith("point "):
            grid = line.split(" grid=", 1)[1].split(" spread_us=", 1)[0]
        if line.startswith("samples_us "):
            samples = line
    return median, grid, samples, proc.stdout


def reference_and_match(path, binary=None):
    """Converged OpenMM Reference forces, then a CPU tolerance sweep.

    The head-to-head column times OpenMM at the tolerance that reached
    its force-error floor, and rgpot on the accuracy-rule mesh for that
    same error. The 1e-4 rgpot column is a separate run.
    """
    import time

    pos, _z, q, box = load_geometry(path)
    n = len(q)
    print(f"converged_reference n={n}", flush=True)
    tight = None
    tight_tol = None
    prev = None
    for tol in (1.0e-5, 1.0e-6, 1.0e-7):
        try:
            ctx, _nb, _actual, pme = openmm_context(pos, q, box, 1, tol, "Reference")
            _e, forces = forces_from_state(ctx)
            del ctx
        except Exception as exc:
            print(f"reference_tol={tol:.0e} failed {exc}", flush=True)
            break
        if prev is not None:
            delta = float(np.max(np.abs(forces - prev[0])))
            print(
                f"reference_convergence n={n} tol={tol:.0e} "
                f"versus={prev[1]:.0e} max_abs_eV_A={delta:.17g} grid={pme[1]} {pme[2]} {pme[3]}",
                flush=True,
            )
        prev = (forces, tol)
        tight = forces
        tight_tol = tol
        # A 1e-7 Reference mesh at large N can take many minutes.
        # Stop once two neighboring tolerances agree past 1e-7.
        if prev is not None and tol <= 1.0e-6:
            # evaluated above when tol is the second or third value
            pass
        if tol <= 1.0e-6 and n >= 4096:
            break
    if tight is None:
        print(f"converged_reference n={n} unavailable", flush=True)
        return
    print(f"reference_used n={n} tol={tight_tol:.0e}", flush=True)
    matched = None
    sweep = []
    for tol in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7):
        ctx, _nb, _actual, pme = openmm_context(pos, q, box, 1, tol, "CPU")
        ctx.getState(getForces=True)
        samples = []
        for _ in range(NREP):
            t0 = time.perf_counter()
            _e, forces = forces_from_state(ctx)
            samples.append((time.perf_counter() - t0) * 1e6)
        err = float(np.max(np.abs(forces - tight)))
        med = quantile7(samples, 0.5)
        print(
            f"cpu_sweep n={n} ewald_tolerance={tol:.0e} max_abs_eV_A={err:.17g} "
            f"median_us={med:.17g} grid={pme[1]} {pme[2]} {pme[3]}",
            flush=True,
        )
        print("samples_us " + " ".join(f"{v:.17g}" for v in samples), flush=True)
        del ctx
        sweep.append((tol, err, med, pme[1], pme[2], pme[3]))
        if err <= FORCE_LIMIT_EV_A and matched is None:
            matched = (tol, med, err)
    if matched is None:
        print(f"matched_cpu n={n} none", flush=True)
    else:
        print(
            f"matched_cpu n={n} ewald_tolerance={matched[0]:.0e} "
            f"median_us={matched[1]:.17g} max_abs_eV_A={matched[2]:.17g}",
            flush=True,
        )
    if not sweep:
        return
    floor_err = min(row[1] for row in sweep)
    # The loosest tolerance that sits on the floor. Tighter knobs that
    # do not move the error are not a more accurate comparator.
    floor_row = next(row for row in sweep if row[1] <= floor_err * (1.0 + 1.0e-9) + 1.0e-18)
    reaches = floor_err <= FORCE_LIMIT_EV_A
    print(
        f"openmm_cpu_floor n={n} max_abs_eV_A={floor_err:.17g} "
        f"reaches_1e-4={'yes' if reaches else 'no'} "
        f"ewald_tolerance={floor_row[0]:.0e} median_us={floor_row[2]:.17g} "
        f"grid={floor_row[3]} {floor_row[4]} {floor_row[5]}",
        flush=True,
    )
    if not reaches:
        print(
            f"openmm_cpu_does_not_reach_1e-4 n={n} floor_max_abs_eV_A={floor_err:.17g}",
            flush=True,
        )
    if binary:
        for threads in (1, 2, 4, 8):
            rgpot_med, rgpot_grid, rgpot_samples, _out = time_rgpot_pme(
                binary, path, threads, floor_err
            )
            if rgpot_med is None:
                raise SystemExit(f"rgpot matched timing produced no median n={n} threads={threads}")
            print(
                f"matched_head_to_head n={n} threads={threads} "
                f"openmm_median_us={floor_row[2]:.17g} openmm_max_abs_eV_A={floor_err:.17g} "
                f"rgpot_tol={floor_err:.17g} rgpot_median_us={rgpot_med:.17g} "
                f"rgpot_grid={rgpot_grid} {load_fields()}",
                flush=True,
            )
            if rgpot_samples:
                print("rgpot_" + rgpot_samples, flush=True)


def drive(geom_dir, binary, threads_list):
    print("command ref_bench.py drive", flush=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    print(f"head {head}", flush=True)
    print(f"tree {tree}", flush=True)
    print(f"binary {binary}", flush=True)
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
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(threads)
                proc = subprocess.run(cmd, env=env)
                if proc.returncode != 0:
                    # A killed comparator did not finish six repeats. Do not
                    # invent a median or a speedup from the budget.
                    print(
                        f"{arm} n={n} threads={threads} timeout limit_s={limit} "
                        f"exit_code={proc.returncode}",
                        flush=True,
                    )
        if binary:
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            proc = subprocess.run(
                [sys.executable, script, "certificate", binary, str(path), str(threads_list[0])],
                env=env,
            )
            if proc.returncode != 0:
                raise SystemExit(f"certificate n={n} exit_code={proc.returncode}")
        reference_and_match(path, binary)
    print("exit_code=0", flush=True)


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
    if len(sys.argv) >= 2 and sys.argv[1] == "certificate":
        certificate(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "match":
        geom = sys.argv[2]
        binary = sys.argv[3]
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
        print("command ref_bench.py match", flush=True)
        print(f"head {head}", flush=True)
        print(f"tree {tree}", flush=True)
        print(f"binary {binary}", flush=True)
        print(f"hostname {socket.gethostname()}", flush=True)
        print(f"cpu {cpu_model()}", flush=True)
        print(load_fields(), flush=True)
        reference_and_match(geom, binary)
        print("exit_code=0", flush=True)
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
