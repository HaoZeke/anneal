#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seed of the non-LJ (Morse) ASE head-to-head panel.

Usage: run_one.py SYSTEM ARM SEED
       run_one.py verify SYSTEM SEED

`for_cluster` is Config::with_scales + for_cluster Bayes screening.
`recommended` is Config::recommended_with_scales.
ASE basin hopping and minima hopping use the ASE 3.29 constructor defaults
with ASE's native MorsePotential (epsilon=1.0, r0=1.0, rho0=rho,
rcut1=1e6, rcut2=2e6 so no pair is truncated).
Before any ASE run, `verify_agreement` checks that anneal and ASE agree
on the exact same geometry to within 1e-8 on energy and forces.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL, energy_hits  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "ase_gofee_h2h"))
from potential import BudgetReached, CapReached, HitReached  # noqa: E402

PAPER_BUDGETS = {
    "m38r6": 400_000,
    "m38r14": 400_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
}
SYSTEMS = {
    "m38r6": {"n": 38, "rho": 6.0, "budget": 400_000, "reference": -157.477108},
    "m38r14": {"n": 38, "rho": 14.0, "budget": 400_000, "reference": -144.321054},
    "m55r6": {"n": 55, "rho": 6.0, "budget": 1_000_000, "reference": -250.286609},
    "m55r14": {"n": 55, "rho": 14.0, "budget": 1_000_000, "reference": -220.646208},
}
HIT_TOL = DEFAULT_HIT_TOL
ASE_VERSION = "3.29.0"
MIN_SEP = 0.85
DENSITY = 0.7
MORSE_LENGTH_SCALE = 1.0 / (2.0 ** (1.0 / 6.0))
SEEDS = tuple(range(2000, 2012))
FMAX = {"ase_bh": 0.1, "ase_mh": 0.05}
ARMS = ("for_cluster", "recommended", "ase_bh", "ase_mh")
AGREEMENT_TOL = 1.0e-8


def hostname() -> str:
    return Path("/proc/sys/kernel/hostname").read_text().splitlines()[0].strip()


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().splitlines()[0].strip()


def place(n: int, seed: int, length: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = length * (3.0 * n / (4.0 * np.pi * DENSITY)) ** (1.0 / 3.0)
    gap = MIN_SEP * length
    points: list[np.ndarray] = []
    draws = 0
    while len(points) < n:
        draws += 1
        if draws > 1_000_000:
            raise RuntimeError("could not place the cluster")
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            continue
        point = direction * (radius * (float(rng.random()) ** (1.0 / 3.0)) / norm)
        if all(float(np.linalg.norm(point - other)) >= gap for other in points):
            points.append(point)
    return np.vstack(points)


def require_ase_version() -> None:
    import ase

    if ase.__version__ != ASE_VERSION:
        raise RuntimeError(f"ASE {ASE_VERSION} required, found {ase.__version__}")


def verify_agreement(system: str, seed: int) -> tuple[float, float, float, float]:
    from ase import Atoms
    from ase.calculators.morse import MorsePotential

    require_ase_version()

    binary = os.environ.get("ASE_NONLJ", "/work/target-panel/release/examples/ase_nonlj")
    spec = SYSTEMS[system]
    proc = subprocess.run(
        [binary, "eval", system, str(seed)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = proc.stdout.splitlines()
    e_anneal = float(next(line for line in lines if line.startswith("energy=")).split("=", 1)[1])
    coords = np.array(
        [[float(v) for v in line.split()[1:]] for line in lines if line.startswith("coord ")],
        dtype=np.float64,
    )
    grad_anneal = np.array(
        [[float(v) for v in line.split()[1:]] for line in lines if line.startswith("grad ")],
        dtype=np.float64,
    )
    atoms = Atoms(f"Ar{spec['n']}", positions=coords, pbc=False)
    atoms.calc = MorsePotential(
        epsilon=1.0,
        r0=1.0,
        rho0=spec["rho"],
        rcut1=1.0e6,
        rcut2=2.0e6,
    )
    e_ase = float(atoms.get_potential_energy())
    f_ase = np.asarray(atoms.get_forces(), dtype=np.float64)
    abs_de = abs(e_anneal - e_ase)
    max_df = float(np.max(np.abs(f_ase + grad_anneal)))
    if abs_de > AGREEMENT_TOL or max_df > AGREEMENT_TOL:
        raise RuntimeError(
            f"energy/force mismatch on {system} seed {seed}: |dE|={abs_de:.3e} max|dF|={max_df:.3e}"
        )
    return e_anneal, e_ase, abs_de, max_df


def make_morse_calculator(rho: float, budget: int, reference: float, fmax: float, deadline: float):
    from ase.calculators.morse import MorsePotential

    class MeteredMorse(MorsePotential):
        def __init__(self):
            super().__init__(
                epsilon=1.0,
                r0=1.0,
                rho0=rho,
                rcut1=1.0e6,
                rcut2=2.0e6,
            )
            self.budget = budget
            self.reference = reference
            self.tol = HIT_TOL
            self.fmax = fmax
            self.deadline = deadline
            self.calls = 0
            self.inside_s = 0.0
            self.hit_calls = None
            self.best = 1.0e300
            self.best_quenched = 1.0e300

        def calculate(self, atoms=None, properties=None, system_changes=None):
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                raise CapReached()
            if self.calls >= self.budget:
                raise BudgetReached()
            started = time.perf_counter()
            super().calculate(atoms, properties, system_changes)
            self.inside_s += time.perf_counter() - started
            self.calls += 1
            energy = float(self.results["energy"])
            forces = self.results["forces"]
            if energy < self.best:
                self.best = energy
            peak = float(np.max(np.abs(forces))) if forces.size else 0.0
            if peak <= self.fmax and energy < self.best_quenched:
                self.best_quenched = energy
            if (
                self.hit_calls is None
                and peak <= self.fmax
                and energy_hits(energy, self.reference, self.tol)
            ):
                self.hit_calls = self.calls
                raise HitReached()

    return MeteredMorse()


def run_ase(system: str, arm: str, seed: int, work: Path, cap_s: float) -> dict:
    from ase import Atoms
    from ase.optimize.basin import BasinHopping
    from ase.optimize.minimahopping import MinimaHopping

    require_ase_version()
    verify_agreement(system, seed)
    spec = SYSTEMS[system]
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    atoms = Atoms(
        f"Ar{spec['n']}",
        positions=place(spec["n"], seed, MORSE_LENGTH_SCALE),
        pbc=False,
    )
    started = time.perf_counter()
    calc = make_morse_calculator(
        spec["rho"],
        spec["budget"],
        spec["reference"],
        FMAX[arm],
        started + cap_s,
    )
    atoms.calc = calc
    np.random.seed(seed)
    stop = "budget"
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
        stop = "wall"
    except (BudgetReached, HitReached):
        stop = "budget"
    wall = time.perf_counter() - started
    best = calc.best_quenched if calc.best_quenched < 1.0e299 else calc.best
    hit = calc.hit_calls is not None
    return {
        "hit": int(hit),
        "evals": calc.hit_calls if hit else "",
        "charged": calc.calls,
        "best": float(best),
        "wall": wall,
        "inside": calc.inside_s,
        "outside": max(0.0, wall - calc.inside_s),
        "stop": "hit" if hit else stop,
        "cap_s": cap_s,
    }


def run_anneal(system: str, seed: int, preset: str, cap_s: str) -> dict:
    binary = os.environ.get("ASE_NONLJ", "/work/target-panel/release/examples/ase_nonlj")
    spec = SYSTEMS[system]
    command = [binary, "run", system, str(spec["budget"]), str(seed), preset]
    if cap_s:
        command.append(cap_s)
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    fields = {}
    for line in proc.stdout.splitlines():
        if line.startswith("system="):
            fields = dict(part.split("=", 1) for part in line.split())
    if fields.get("system") != system or fields.get("seed") != str(seed) or fields.get("arm") != preset:
        raise RuntimeError("anneal stdout has no row")
    hit = fields["hit"] == "1"
    stop = fields.get("stop") or ("hit" if hit else "budget")
    return {
        "hit": int(hit),
        "evals": "" if fields["first_calls"] == "-" else fields["first_calls"],
        "charged": int(fields["charged"]),
        "best": float(fields["best"]),
        "wall": float(fields["wall_s"]),
        "inside": float(fields["inside_s"]),
        "outside": float(fields["outside_s"]),
        "stop": stop,
        "cap_s": cap_s,
    }


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "verify":
        system, seed_text = sys.argv[2:]
        e_anneal, e_ase, abs_de, max_df = verify_agreement(system, int(seed_text))
        print(f"{system},{seed_text},{e_anneal:.16e},{e_ase:.16e},{abs_de:.6e},{max_df:.6e}")
        return 0
    if hostname() != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    if len(sys.argv) != 4:
        print("usage: run_one.py SYSTEM ARM SEED", file=sys.stderr)
        return 2
    system, arm, seed_text = sys.argv[1:]
    if system not in SYSTEMS or arm not in ARMS:
        print("unknown system or arm", file=sys.stderr)
        return 2
    seed = int(seed_text)
    if seed not in SEEDS:
        print("seed is outside 2000-2011", file=sys.stderr)
        return 2
    revision = os.environ.get("ASE_NONLJ_REVISION", "")
    if not revision:
        print("ASE_NONLJ_REVISION is unset", file=sys.stderr)
        return 2
    spec = SYSTEMS[system]
    if arm in ("for_cluster", "recommended"):
        cap_text = ""
        if arm == "recommended":
            cap_text = os.environ.get("ASE_NONLJ_WALL_CAP", "")
            if not cap_text:
                print("ASE_NONLJ_WALL_CAP is unset", file=sys.stderr)
                return 2
        row = run_anneal(system, seed, arm, cap_text)
    else:
        cap_text = os.environ.get("ASE_NONLJ_WALL_CAP", "")
        if not cap_text:
            print("ASE_NONLJ_WALL_CAP is unset", file=sys.stderr)
            return 2
        row = run_ase(
            system,
            arm,
            seed,
            Path(os.environ.get("ASE_NONLJ_WORK", "/work/ase-nonlj/runs")) / f"{system}-{arm}-{seed}",
            float(cap_text),
        )
    cap = "" if row["cap_s"] == "" else f"{float(row['cap_s']):.6f}"
    print(
        ",".join(
            [
                system,
                arm,
                str(seed),
                str(spec["budget"]),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
