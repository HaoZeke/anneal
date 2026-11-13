#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One seeded ASE genetic algorithm on a Cambridge Gupta cluster.

The operators are the ASE cluster GA: a random start population, Deaven–Ho
cut and splice, rattle, mirror, and, for two species, permutation. Each
offspring is relaxed with BFGS. One calculator call is one energy-and-force
evaluation. The run stops at the shared call budget, or when a finished
relaxation lies within 0.001 of the Cambridge minimum.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.ga import set_raw_score
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import MirrorMutation, PermutationMutation, RattleMutation
from ase.ga.startgenerator import StartGenerator
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS

from gupta_calc import GuptaCalculator

HIT_TOL = 0.001
POPULATION = 8
FMAX = 1.0e-3
RELAX_STEPS = 200


class BudgetSpent(RuntimeError):
    """The shared call budget was reached inside a relaxation."""


class BudgetCalculator(GuptaCalculator):
    """Gupta calculator that counts calls and keeps the time inside ``evaluate``."""

    def __init__(self, budget: int, level: float):
        super().__init__()
        self.budget = budget
        self.level = level
        self.calls = 0
        self.pot_s = 0.0
        self.best = float("inf")
        self.calls_at_hit: int | None = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        if self.calls >= self.budget:
            raise BudgetSpent
        started = time.perf_counter()
        super().calculate(atoms, properties, system_changes)
        self.pot_s += time.perf_counter() - started
        self.calls += 1

    def note_quench(self, energy: float) -> None:
        if energy < self.best:
            self.best = energy
        if energy <= self.level and self.calls_at_hit is None:
            self.calls_at_hit = self.calls


def composition(system: str) -> tuple[list[int], float, dict[tuple[int, int], float]]:
    if system.startswith("zn"):
        count = int(system[2:])
        numbers = [30] * count
        scale = 1.0
        blmin = {(30, 30): 0.70}
        return numbers, scale, blmin
    if system == "cu19au19":
        numbers = [29] * 19 + [79] * 19
        blmin = {(29, 29): 1.9, (79, 79): 2.1, (29, 79): 2.0, (79, 29): 2.0}
        return numbers, 2.7, blmin
    raise SystemExit(f"unknown system {system}")


def reference_of(binary: Path, system: str) -> tuple[int, float]:
    proc_budget = __import__("subprocess").run(
        [str(binary), "budget", system],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in proc_budget.stdout.splitlines() if line.strip()]
    return int(lines[0]), float(lines[1])


def relax(atoms: Atoms, calc: BudgetCalculator) -> bool:
    atoms.calc = calc
    try:
        with BFGS(atoms, logfile=None, maxstep=0.2) as dyn:
            dyn.run(fmax=FMAX, steps=RELAX_STEPS)
    except BudgetSpent:
        return False
    energy = float(atoms.get_potential_energy())
    set_raw_score(atoms, -energy)
    calc.note_quench(energy)
    return True


def run(system: str, seed: int, budget: int, level: float, work: Path) -> dict[str, float | int | str]:
    numbers, scale, blmin = composition(system)
    count = len(numbers)
    side = 1.8 * scale * count ** (1.0 / 3.0)
    cell = 3.0 * side
    slab = Atoms(cell=[cell, cell, cell], pbc=False)
    origin = np.full(3, 0.5 * (cell - side))
    box = [origin, [np.eye(3)[axis] * side for axis in range(3)]]
    rng = np.random.RandomState(seed)
    starter = StartGenerator(
        slab=slab,
        blocks=numbers,
        blmin=blmin,
        box_to_place_in=box,
        rng=rng,
        test_dist_to_slab=False,
        test_too_far=False,
    )
    work.mkdir(parents=True, exist_ok=True)
    database = work / "ga.db"
    if database.exists():
        database.unlink()
    prepared = PrepareDB(db_file_name=str(database), simulation_cell=slab, stoichiometry=numbers)
    calc = BudgetCalculator(budget, level)
    started = time.perf_counter()
    for _ in range(POPULATION):
        candidate = starter.get_new_candidate(maxiter=5000)
        if candidate is None:
            continue
        prepared.add_unrelaxed_candidate(candidate)
    connection = DataConnection(str(database))
    comparator = InteratomicDistanceComparator(
        n_top=count,
        pair_cor_cum_diff=0.015,
        pair_cor_max=0.7,
        dE=0.02,
        mic=False,
    )
    pairing = CutAndSplicePairing(slab, count, blmin, rng=rng)
    operators = [
        MirrorMutation(blmin, count, rng=rng),
        RattleMutation(blmin, count, rng=rng),
    ]
    if len(set(numbers)) > 1:
        operators.append(PermutationMutation(count, rng=rng))
    mutations = OperationSelector([1.0] * len(operators), operators, rng=rng)
    while connection.get_number_of_unrelaxed_candidates() > 0 and calc.calls < budget:
        candidate = connection.get_an_unrelaxed_candidate()
        if relax(candidate, calc):
            connection.add_relaxed_step(candidate)
        if calc.calls_at_hit is not None:
            break
    population = Population(
        data_connection=connection,
        population_size=POPULATION,
        comparator=comparator,
        rng=rng,
    )
    while calc.calls < budget and calc.calls_at_hit is None:
        try:
            parents = population.get_two_candidates()
        except Exception:
            break
        if not parents or any(parent is None for parent in parents):
            break
        child, description = pairing.get_new_individual(parents)
        if child is None:
            if calc.calls >= budget:
                break
            continue
        connection.add_unrelaxed_candidate(child, description=description)
        if rng.random() < 0.3:
            mutated, mutated_description = mutations.get_new_individual([child])
            if mutated is not None:
                connection.add_unrelaxed_step(mutated, mutated_description)
                child = mutated
        if relax(child, calc):
            connection.add_relaxed_step(child)
            population.update()
    wall = time.perf_counter() - started
    hit = calc.calls_at_hit is not None
    evals = calc.calls_at_hit if hit else budget
    outside = max(0.0, wall - calc.pot_s)
    calls = max(calc.calls, 1)
    best = calc.best if np.isfinite(calc.best) else 0.0
    return {
        "system": system,
        "seed": seed,
        "budget": budget,
        "hit": int(hit),
        "evals": int(evals),
        "charged": calc.calls,
        "wall": wall,
        "best": best,
        "pot_s": calc.pot_s,
        "outside": outside,
        "pot_per": calc.pot_s / calls,
        "outside_per": outside / calls,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv[1:])
    _budget, reference = reference_of(args.binary, args.system)
    if args.budget <= 0:
        raise SystemExit("budget must be positive")
    row = run(args.system, args.seed, args.budget, reference + HIT_TOL, args.work)
    print(
        "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,pot_per_eval_s,outside_per_eval_s"
    )
    print(
        "{system},ase,{seed},{budget},{hit},{evals},{charged},{wall:.6f},{best:.8f},{pot_s:.6f},{outside:.6f},{pot_per:.9f},{outside_per:.9f}".format(
            **row
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
