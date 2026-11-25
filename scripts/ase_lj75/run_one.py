#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One LJ75 seed of ASE basin hopping, ASE minima hopping, or anneal.

anneal is Config::for_cluster at the h2h_arm3 Lennard-Jones pin.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults.
The potential is the untruncated pair in scripts/ase_gofee_h2h/potential.py.
A hit is a quenched energy within 0.001 of -397.492331. A miss is censored
at the budget. One calculator call is one energy and one gradient.

Each ASE seed stops at a hit, the paper call budget, or an optional process
CPU cap. The CPU cap is independent of the older wall-cap protocol.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ase_gofee_h2h"))
from potential import (  # noqa: E402
    BudgetReached,
    CapReached,
    HitReached,
    LjCalculator,
    lj_value_and_gradient,
    warmup,
)

N = 75
PAPER_BUDGETS = {75: 4_000_000}
BUDGET = PAPER_BUDGETS[75]
REFERENCE = -397.492331
HIT_TOL = 0.001
MIN_SEP = 0.85
DENSITY = 0.7
SEEDS = tuple(range(2000, 2012))
FMAX = {"ase_bh": 0.1, "ase_mh": 0.05}
ARMS = ("anneal", "ase_bh", "ase_mh")
ANNEAL_REVISION = "98dcfb3106f191baf7381fecd5624426831094fe"
SMOKE_BUDGET = 2000
SMOKE_CAP_S = 30.0


def hostname() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().splitlines()[0].strip()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().splitlines()[0].strip()


def place(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = (3.0 * N / (4.0 * np.pi * DENSITY)) ** (1.0 / 3.0)
    points: list[np.ndarray] = []
    draws = 0
    while len(points) < N:
        draws += 1
        if draws > 1_000_000:
            raise RuntimeError("could not place the cluster")
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        point = direction * (radius * (float(rng.random()) ** (1.0 / 3.0)) / norm)
        if all(float(np.linalg.norm(point - other)) >= MIN_SEP for other in points):
            points.append(point)
    return np.vstack(points)


def run_ase(
    arm: str,
    seed: int,
    work: Path,
    budget: int,
    cap_s: float | None,
    cpu_cap_s: float | None = None,
) -> dict:
    import ase
    from ase import Atoms
    from ase.optimize.basin import BasinHopping
    from ase.optimize.minimahopping import MinimaHopping

    if ase.__version__ != "3.29.0":
        raise RuntimeError(f"ASE 3.29.0 required, found {ase.__version__}")
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    warmup()
    atoms = Atoms(f"Ar{N}", positions=place(seed), pbc=False)
    started = time.perf_counter()
    calc = LjCalculator(
        budget,
        REFERENCE,
        HIT_TOL,
        FMAX[arm],
        deadline=None if cap_s is None else started + cap_s,
    )
    atoms.calc = calc
    np.random.seed(seed)
    stop = "budget"
    cpu_started = time.process_time()
    old_sigprof = None
    if cpu_cap_s is not None:
        old_sigprof = signal.getsignal(signal.SIGPROF)

        def cpu_cap_reached(signum, frame):
            raise CapReached()

        signal.signal(signal.SIGPROF, cpu_cap_reached)
        signal.setitimer(signal.ITIMER_PROF, cpu_cap_s)
    try:
        if arm == "ase_bh":
            BasinHopping(
                atoms,
                logfile=None,
                optimizer_logfile=None,
                trajectory=None,
                local_minima_trajectory=None,
            ).run(10**9)
        else:
            MinimaHopping(atoms)(totalsteps=10**9)
    except CapReached:
        stop = "cap reached (12 h CPU)" if cpu_cap_s is not None else "wall"
    except (BudgetReached, HitReached):
        stop = "budget" if calc.hit_calls is None else "hit"
    finally:
        if cpu_cap_s is not None:
            signal.setitimer(signal.ITIMER_PROF, 0.0)
            signal.signal(signal.SIGPROF, old_sigprof)
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    best = calc.best_quenched if calc.best_quenched < 1.0e299 else calc.best
    hit = calc.hit_calls is not None
    return {
        "hit": int(hit),
        "evals": calc.hit_calls if hit else "",
        "charged": calc.calls,
        "best": float(best),
        "wall": wall,
        "cpu": cpu,
        "inside": calc.inside_s,
        "outside": max(0.0, wall - calc.inside_s),
        "stop": "hit" if hit else stop,
        "cap_s": (
            cpu_cap_s if cpu_cap_s is not None
            else ("" if cap_s is None else cap_s)
        ),
    }


def run_anneal(seed: int, budget: int) -> dict:
    binary = os.environ.get("ANNEAL_LJ75", "/work/target-gate/release/examples/leave_gate")
    proc = subprocess.run(
        [binary, "lj75", "base", str(budget), str(seed)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    fields = {}
    for line in proc.stdout.splitlines():
        if line.startswith("system="):
            fields = dict(part.split("=", 1) for part in line.split())
    if fields.get("arm") != "base" or fields.get("seed") != str(seed):
        raise RuntimeError("anneal stdout has no row")
    hit = fields["hit"] == "1"
    wall = float(fields["wall_s"])
    inside = float(fields["inside_s"])
    return {
        "hit": int(hit),
        "evals": "" if fields["first_calls"] == "-" else fields["first_calls"],
        "charged": int(fields["charged"]),
        "best": float(fields["best"]),
        "wall": wall,
        "inside": inside,
        "outside": float(fields["outside_s"]),
        "stop": "hit" if hit else "budget",
        "cap_s": "",
    }


def check_pair() -> None:
    r = 2.0 ** (1.0 / 6.0)
    x = np.zeros(6)
    x[3] = r
    energy, grad = lj_value_and_gradient(x)
    if abs(float(energy) + 1.0) > 1.0e-8 or float(np.linalg.norm(grad)) > 1.0e-8:
        raise RuntimeError(f"LJ pair drifted {energy}")


def emit(arm: str, seed: int, budget: int, revision: str, row: dict) -> None:
    cap = "" if row["cap_s"] == "" else f"{float(row['cap_s']):.6f}"
    print(
        ",".join(
            [
                "lj75",
                arm,
                str(seed),
                str(budget),
                str(row["hit"]),
                "" if row["evals"] == "" else str(row["evals"]),
                str(row["charged"]),
                f"{row['best']:.8f}",
                f"{row['wall']:.6f}",
                f"{row['inside']:.6f}",
                f"{row['outside']:.6f}",
                revision,
                hostname(),
                row["stop"],
                cap,
            ]
        ),
        flush=True,
    )
    print(f"boot_id={boot_id()}", flush=True)


def main() -> int:
    if hostname() != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    smoke = len(sys.argv) == 2 and sys.argv[1] == "--smoke"
    if smoke:
        check_pair()
        row = run_ase("ase_bh", 2000, Path("/work/ase-lj75/smoke"), SMOKE_BUDGET, SMOKE_CAP_S)
        emit("ase_bh", 2000, SMOKE_BUDGET, "smoke", row)
        return 0
    if len(sys.argv) != 3:
        print("usage: run_one.py ARM SEED", file=sys.stderr)
        return 2
    arm, seed_text = sys.argv[1], sys.argv[2]
    if arm not in ARMS:
        print(f"unknown arm {arm}", file=sys.stderr)
        return 2
    seed = int(seed_text)
    if seed not in SEEDS:
        print("seed is outside 2000-2011", file=sys.stderr)
        return 2
    check_pair()
    revision = os.environ.get("ASE_LJ75_REVISION") or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    budget = int(os.environ.get("ASE_LJ75_BUDGET", BUDGET))
    if not 0 < budget <= BUDGET:
        print(f"ASE_LJ75_BUDGET must be in 1-{BUDGET}", file=sys.stderr)
        return 2
    if arm == "anneal":
        revision = ANNEAL_REVISION
        row = run_anneal(seed, budget)
    else:
        cap_text = os.environ.get("ASE_LJ75_WALL_CAP", "")
        cap_s = float(cap_text) if cap_text else None
        row = run_ase(
            arm,
            seed,
            Path(os.environ.get("ASE_LJ75_WORK", "/work/ase-lj75/runs")) / f"{arm}-{seed}",
            budget,
            cap_s,
            float(os.environ["ASE_LJ75_CPU_CAP"])
            if os.environ.get("ASE_LJ75_CPU_CAP")
            else None,
        )
    emit(arm, seed, budget, revision, row)
    if arm in ("ase_bh", "ase_mh"):
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        mem_kib = next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemTotal:")
        )
        ratio = row["cpu"] / row["wall"] if row["wall"] else 0.0
        shape = {
            "system": "lj75",
            "arm": arm,
            "seed": seed,
            "code_revision": revision,
            "nproc": os.cpu_count(),
            "model_name": model,
            "memory_bytes": mem_kib * 1024,
            "core_affinity": ",".join(
                str(core) for core in sorted(os.sched_getaffinity(0))
            ),
            "cpu_s": row["cpu"],
            "wall_s": row["wall"],
            "cpu_wall_ratio": ratio,
            "vm_stayed_live": ratio >= 0.95,
        }
        print("shape_json=" + json.dumps(shape, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
