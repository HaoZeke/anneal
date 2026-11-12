#!/usr/bin/env python3
# Copyright (c) anneal developers
"""ASE basin hopping and AGOX on the paper Lennard-Jones sizes.

Shared starts are the coordinates ``h2h_gmin_proof coords`` writes.
Seeds are 2000-2005. A hit is a quenched energy at or below the
Cambridge minimum plus 0.001. The process refuses to run unless the
UTS hostname is bench. One thread, serial.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

REFERENCES = {
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
    98: -543.665361,
    104: -582.086642,
}
BUDGETS = {38: 400_000, 55: 1_000_000, 75: 4_000_000, 98: 4_000_000, 104: 4_000_000}
SEEDS = (2000, 2001, 2002, 2003, 2004, 2005)
HIT_TOL = 0.001


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


class Counter:
    def __init__(self, budget: int, reference: float):
        self.budget = budget
        self.reference = reference
        self.calls = 0
        self.best = 1.0e300
        self.hit_calls = None

    def evaluate(self, positions):
        if self.calls >= self.budget:
            raise BudgetReached()
        energy, grad = lj_value_and_gradient(np.ascontiguousarray(positions, dtype=np.float64))
        self.calls += 1
        value = float(energy)
        if value < self.best:
            self.best = value
        if self.hit_calls is None and value <= self.reference + HIT_TOL:
            self.hit_calls = self.calls
            raise HitReached()
        return value, -grad


def load_positions(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        for word in line.split():
            values.append(float(word))
    coords = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    return coords


def make_calculator(counter: Counter):
    from ase.calculators.calculator import Calculator

    class _Bound(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            energy, forces = counter.evaluate(atoms.positions.ravel())
            self.results = {"energy": energy, "forces": forces.reshape(-1, 3)}

    return _Bound()


def run_ase(n: int, seed: int, positions: np.ndarray, counter: Counter) -> None:
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.optimize.basin import BasinHopping

    atoms = Atoms(f"Ar{n}", positions=positions, cell=[40.0, 40.0, 40.0], pbc=False)
    atoms.calc = make_calculator(counter)
    scratch = Path("/work/stated-baselines") / f"lj{n}" / f"ase-{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    hopper = BasinHopping(
        atoms,
        temperature=0.8,
        optimizer=BFGS,
        fmax=0.05,
        dr=0.4,
        logfile=str(scratch / "hop.log"),
        trajectory=str(scratch / "hop.traj"),
        optimizer_logfile=str(scratch / "opt.log"),
        local_minima_trajectory=str(scratch / "minima.traj"),
    )
    while counter.hit_calls is None and counter.calls < counter.budget:
        hopper.run(1)


def run_agox(n: int, seed: int, positions: np.ndarray, counter: Counter) -> None:
    import contextlib

    import ase.constraints as ase_constraints
    from ase.constraints.constraint import IndexedConstraint, slice2enlist

    if not hasattr(ase_constraints, "IndexedConstraint"):
        ase_constraints.IndexedConstraint = IndexedConstraint
    if not hasattr(ase_constraints, "slice2enlist"):
        ase_constraints.slice2enlist = slice2enlist

    from ase import Atoms
    from agox import AGOX
    from agox.databases import Database
    from agox.environments import Environment
    from agox.evaluators import LocalOptimizationEvaluator
    from agox.generators import RattleGenerator
    from agox.samplers import MetropolisSampler

    radius = (3.0 * n / (4.0 * np.pi * 0.7)) ** (1.0 / 3.0)
    edge = 2.0 * radius + 4.0
    cell = 2.0 * edge
    template = Atoms(cell=np.eye(3) * cell, pbc=False)
    environment = Environment(
        template=template,
        symbols=f"Ar{n}",
        confinement_cell=np.eye(3) * edge,
        confinement_corner=np.full(3, edge / 2.0),
        print_report=False,
    )
    work = Path("/work/stated-baselines") / f"lj{n}" / f"agox-{seed}"
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "agox.db"
    if db_path.exists():
        db_path.unlink()
    database = Database(filename=str(db_path), order=4, initialize=True)
    sampler = MetropolisSampler(temperature=0.8, order=3)
    generator = RattleGenerator(
        **environment.get_confinement(),
        environment=environment,
        sampler=sampler,
        order=1,
    )
    calc = make_calculator(counter)
    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "candidates"},
        store_trajectory=False,
        optimizer_run_kwargs={"fmax": 0.05, "steps": 100},
        optimizer_kwargs={"logfile": None},
        order=2,
        constraints=environment.get_constraints(),
    )
    from agox.generators import RandomGenerator
    from ase.optimize import BFGS

    starter = RandomGenerator(**environment.get_confinement(), environment=environment)
    seeded = None
    for _ in range(20):
        made = starter.get_candidates(sampler, environment)
        if made:
            seeded = made[0]
            break
    if seeded is None:
        raise RuntimeError("AGOX produced no start")
    seeded.positions[:] = positions
    seeded.calc = calc
    try:
        BFGS(seeded, logfile=None).run(fmax=0.05, steps=100)
    except (BudgetReached, HitReached):
        return
    sampler.sample = [seeded]
    search = AGOX(generator, database, sampler, evaluator, seed=seed, report=False)

    def dispatch(*args, **kwargs):
        if counter.hit_calls is not None or counter.calls >= counter.budget:
            raise BudgetReached()
        AGOX.dispatch_to_observers(search, *args, **kwargs)
        if counter.hit_calls is not None or counter.calls >= counter.budget:
            raise BudgetReached()

    search.dispatch_to_observers = dispatch
    try:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            search.run(N_iterations=counter.budget, verbose=False, hide_log=True)
    except (BudgetReached, HitReached):
        pass


def one(binary: Path, n: int, seed: int, arm: str, starts: Path) -> dict:
    path = starts / f"lj{n}" / f"seed-{seed}.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        import subprocess

        subprocess.run([str(binary), "coords", f"lj{n}", str(seed), str(path)], check=True)
    positions = load_positions(path)
    lj_value_and_gradient(np.zeros(positions.size, dtype=np.float64))
    counter = Counter(BUDGETS[n], REFERENCES[n])
    started = time.perf_counter()
    try:
        if arm == "ase-bh":
            run_ase(n, seed, positions, counter)
        elif arm == "agox":
            run_agox(n, seed, positions, counter)
        else:
            raise SystemExit(f"unknown arm {arm}")
    except (BudgetReached, HitReached):
        pass
    wall = time.perf_counter() - started
    hit = counter.hit_calls is not None
    evals = counter.hit_calls if hit else counter.budget
    return {
        "system": f"lj{n}",
        "arm": arm,
        "seed": seed,
        "budget": BUDGETS[n],
        "hit": int(hit),
        "evals_to_hit": evals,
        "charged": counter.calls,
        "wall_s": f"{wall:.6f}",
        "best": f"{counter.best:.8f}",
    }


def main(argv: list[str]) -> int:
    if os.uname().nodename != "bench":
        print("uts hostname is not bench", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--size", type=int, action="append", required=True)
    parser.add_argument("--arm", action="append", required=True)
    args = parser.parse_args(argv)
    rows = []
    if args.summary.is_file():
        with args.summary.open() as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["system"], row["arm"], row["seed"]) for row in rows}
    for n in args.size:
        for seed in SEEDS:
            for arm in args.arm:
                key = (f"lj{n}", arm, str(seed))
                if key in done:
                    print(f"skip {key}", flush=True)
                    continue
                print(f"start n={n} arm={arm} seed={seed}", flush=True)
                row = one(args.binary, n, seed, arm, args.starts)
                rows.append(row)
                args.summary.parent.mkdir(parents=True, exist_ok=True)
                with args.summary.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                print(
                    f"done n={n} arm={arm} seed={seed} hit={row['hit']} evals={row['evals_to_hit']} wall_s={row['wall_s']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
