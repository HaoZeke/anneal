#!/usr/bin/env python3
# Copyright (c) anneal developers
"""AGOX basin hopping for LJ38, as the 3.11.1 cluster example documents it.

BoxConstraint is the stock class from Environment.get_constraints.
Each Lennard-Jones call is timed, and the total is printed as inside_s.
Seeds are 6400 through 6405. One argument runs that seed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if os.uname().nodename != "bench":
    print("uts hostname is not bench", file=sys.stderr)
    sys.exit(2)

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "cores",
        "PYTHONUNBUFFERED": "1",
        "HOME": "/work/home",
        "TMPDIR": "/work/scratch",
        "MPLCONFIGDIR": "/work/mpl",
        "NUMBA_CACHE_DIR": "/work/numba",
        "XDG_CACHE_HOME": "/work/cache",
    }
)

import numpy as np
from numba import njit

BUDGET = 400_000
REFERENCE = -173.928427
HIT_TOL = 1.0e-4
FORCE_BAR = 0.0514221
SEEDS = (6400, 6401, 6402, 6403, 6404, 6405)


class BudgetReached(BaseException):
    pass


class HitReached(BaseException):
    pass


@njit(cache=True)
def lj_value_and_gradient(x):
    n = x.shape[0] // 3
    energy = 0.0
    grad = np.zeros(x.shape[0])
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3 * i] - x[3 * j]
            dy = x[3 * i + 1] - x[3 * j + 1]
            dz = x[3 * i + 2] - x[3 * j + 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < 1.0e-12:
                continue
            inv = 1.0 / r2
            inv6 = inv * inv * inv
            inv12 = inv6 * inv6
            energy += 4.0 * (inv12 - inv6)
            coef = 24.0 * inv * (2.0 * inv12 - inv6)
            grad[3 * i] -= coef * dx
            grad[3 * i + 1] -= coef * dy
            grad[3 * i + 2] -= coef * dz
            grad[3 * j] += coef * dx
            grad[3 * j + 1] += coef * dy
            grad[3 * j + 2] += coef * dz
    return energy, grad


class PotentialTimer:
    """Times every call that returns the Lennard-Jones energy and gradient."""

    def __init__(self):
        self.calls = 0
        self.inside_s = 0.0

    def value_and_gradient(self, x):
        started = time.perf_counter()
        energy, grad = lj_value_and_gradient(x)
        self.inside_s += time.perf_counter() - started
        self.calls += 1
        return energy, grad


def bound_calculator(timer: PotentialTimer):
    from ase.calculators.calculator import Calculator

    class _Bound(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            Calculator.__init__(self)
            self.use_cache = False
            self.calls = 0
            self.best = 1.0e300
            self.hit_calls = None
            self._pos = None

        def check_state(self, atoms, tol=1e-15):
            pos = atoms.positions
            cached = self._pos
            if (
                cached is not None
                and cached.shape == pos.shape
                and np.array_equal(cached, pos)
            ):
                return []
            self._pos = np.array(pos, copy=True)
            return ["positions"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            if self.calls >= BUDGET:
                raise BudgetReached()
            positions = np.ascontiguousarray(atoms.positions.ravel(), dtype=np.float64)
            energy, grad = timer.value_and_gradient(positions)
            self.calls += 1
            forces = (-grad).reshape(-1, 3)
            self.results["energy"] = float(energy)
            self.results["forces"] = forces
            if energy < self.best:
                self.best = float(energy)
            if (
                self.hit_calls is None
                and energy < REFERENCE + HIT_TOL
                and float(np.max(np.abs(forces))) <= FORCE_BAR
            ):
                self.hit_calls = self.calls
                raise HitReached()

    return _Bound()


def run_seed(seed: int, timer: PotentialTimer) -> int:
    inside_before = timer.inside_s
    calls_before = timer.calls
    import contextlib

    import ase.constraints as ase_constraints
    from ase import Atoms
    from ase.constraints.constraint import IndexedConstraint, slice2enlist
    from ase.optimize import BFGS
    from agox import AGOX
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RandomGenerator, RattleGenerator
    from agox.samplers import MetropolisSampler

    if not hasattr(ase_constraints, "IndexedConstraint"):
        ase_constraints.IndexedConstraint = IndexedConstraint
    if not hasattr(ase_constraints, "slice2enlist"):
        ase_constraints.slice2enlist = slice2enlist

    n = 38
    radius = (3.0 * n / (4.0 * np.pi * 0.7)) ** (1.0 / 3.0)
    confinement_edge = 2.0 * radius + 4.0
    cell_edge = 2.0 * confinement_edge
    corner = confinement_edge / 2.0
    template = Atoms(cell=np.eye(3) * cell_edge, pbc=False)
    environment = Environment(
        template=template,
        symbols=f"Ar{n}",
        confinement_cell=np.eye(3) * confinement_edge,
        confinement_corner=np.full(3, corner),
        print_report=False,
    )
    work = Path("results") / "lj38-agox" / f"seed{seed}"
    work.mkdir(parents=True, exist_ok=True)
    calc = bound_calculator(timer)
    db_path = work / "agox.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=4, initialize=True)
    sampler = MetropolisSampler(temperature=0.25, order=3)
    generator = RattleGenerator(
        **environment.get_confinement(),
        environment=environment,
        sampler=sampler,
        order=1,
    )
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "candidates"},
        store_trajectory=False,
        optimizer_run_kwargs={"fmax": 0.05, "steps": 100},
        optimizer_kwargs={"logfile": None},
        order=2,
        constraints=environment.get_constraints(),
    )
    starter = RandomGenerator(**environment.get_confinement(), environment=environment)
    seeded = None
    for _ in range(20):
        made = starter.get_candidates(sampler, environment)
        if made:
            seeded = made[0]
            break
    if seeded is None:
        print("RandomGenerator produced no start", file=sys.stderr)
        return 2
    seeded.calc = calc
    try:
        BFGS(seeded, logfile=None).run(fmax=0.05, steps=100)
    except (BudgetReached, HitReached):
        pass
    if calc.hit_calls is None and calc.calls < BUDGET:
        sampler.sample = [seeded]
    search = AGOX(generator, database, sampler, evaluator, seed=seed, report=False)

    def dispatch(*args, **kwargs):
        if calc.hit_calls is not None or calc.calls >= BUDGET:
            raise BudgetReached()
        before = calc.calls
        AGOX.dispatch_to_observers(search, *args, **kwargs)
        if calc.hit_calls is not None or calc.calls >= BUDGET:
            raise BudgetReached()
        if calc.calls == before:
            dispatch.stall += 1
            if dispatch.stall >= 3:
                raise RuntimeError("the search added no charged calls")
        else:
            dispatch.stall = 0

    dispatch.stall = 0
    search.dispatch_to_observers = dispatch
    started = time.perf_counter()
    try:
        with (work / "agox.stdio").open("w") as handle:
            with contextlib.redirect_stdout(handle):
                try:
                    search.run(N_iterations=BUDGET, verbose=False, hide_log=True)
                finally:
                    import atexit

                    atexit.unregister(search.on_failure)
    except (BudgetReached, HitReached, RuntimeError):
        pass
    wall_s = time.perf_counter() - started
    hit = int(calc.hit_calls is not None)
    stop = "hit" if hit else "ceiling" if calc.calls >= BUDGET else "stall"
    print(
        f"seed={seed} hit={hit} calls={calc.calls} best={calc.best:.6f} "
        f"stop={stop} wall_s={wall_s:.6f} "
        f"inside_s={timer.inside_s - inside_before:.6f} "
        f"potential_calls={timer.calls - calls_before}",
        flush=True,
    )
    return 0


def main() -> int:
    if len(sys.argv) == 1:
        seeds = SEEDS
    elif len(sys.argv) == 2 and int(sys.argv[1]) in SEEDS:
        seeds = (int(sys.argv[1]),)
    else:
        print("usage: agox_documented.py [6400-6405]", file=sys.stderr)
        return 2
    timer = PotentialTimer()
    for seed in seeds:
        rc = run_seed(seed, timer)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
