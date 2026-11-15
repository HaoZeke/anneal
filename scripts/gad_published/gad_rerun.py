#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One Guided Adaptive Diffusion run on a Lennard-Jones cluster.

The search is CondEvo's CHARLES at ``CONDEVO_COMMIT`` with the settings
the Zenodo 10.5281/zenodo.19001618 deposit records for its LJ31 and LJ38
runs: population 64, initial step 1.5, a DDIM over 5000 steps on an MLP
of 8 hidden layers of 96 LeakyReLU units, the genetic-algorithm
selection at pressure 20 with adaptive pressure, elite ratio 0.15,
crossover ratio 0.125, mutation rate 0.05, a buffer of 1000, Adam at
learning rate 0.01 and weight decay 1e-5, batch 256, 1000 training
epochs, and two conditions: the origin sphere and k-nearest-neighbour
novelty with k 10 and beta 10. Every sample is relaxed by ASE FIRE to
fmax 0.001 within 1000 steps, and the relaxed coordinates are what the
search is told, as in the paper's memetic step.

The origin condition marks a sample whose atoms all lie within the
sphere about the origin. Its radius is the founder's largest distance
from the origin times ``ORIGIN_SCALE``, the ratio of the recorded
threshold to that distance in both deposited runs.

The founder is the seeded start the anneal example writes for the same
seed, so all three codes start from one cluster. The potential is the
untruncated Lennard-Jones pair sum in reduced units. One call is one
energy and force evaluation. The run stops at the first call within
0.001 of the Cambridge energy or when the budget is spent; a miss is
reported as the budget.

  gad_rerun.py lj38 2000 --founder FILE [--budget 400000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import FIRE

from condevo.diffusion import DDIM
from condevo.es import CHARLES
from condevo.es.guidance import Condition, KNNNoveltyCondition
from condevo.nn import MLP

CONDEVO_COMMIT = "ae76bf0462de63fa6bf3bde97df7cc00237b8d1d"
REFERENCES = {"lj31": -133.586422, "lj38": -173.928427}
HIT_TOL = 0.001
BUDGET = 400_000
ORIGIN_SCALE = 3.0512013542537524 / 3.046205997143528
POPSIZE = 64
SIGMA_INIT = 1.5
FMAX = 0.001
FIRE_STEPS = 1000
GENERATIONS = 1000


class Spent(Exception):
    pass


class LennardJones(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, tally: "Tally"):
        super().__init__()
        self.tally = tally

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.tally.calls >= self.tally.budget:
            raise Spent
        mark = time.perf_counter()
        x = self.atoms.get_positions()
        d = x[:, None, :] - x[None, :, :]
        r2 = (d * d).sum(-1)
        np.fill_diagonal(r2, np.inf)
        inv6 = r2**-3
        energy = 2.0 * float((inv6 * inv6 - inv6).sum())
        scale = (48.0 * inv6 * inv6 - 24.0 * inv6) / r2
        forces = (scale[:, :, None] * d).sum(1)
        self.tally.inside += time.perf_counter() - mark
        self.tally.calls += 1
        self.tally.observe(energy)
        self.results = {"energy": energy, "forces": forces}


class Tally:
    def __init__(self, budget: int, target: float):
        self.budget = budget
        self.target = target
        self.calls = 0
        self.inside = 0.0
        self.best = np.inf
        self.hit_at: int | None = None

    def observe(self, energy: float) -> None:
        self.best = min(self.best, energy)
        if self.hit_at is None and energy <= self.target:
            self.hit_at = self.calls
            raise Spent


class OriginCondition(Condition):
    def __init__(self, radius: float, target: float = 1.0):
        super().__init__()
        self.radius = radius
        self.target = target

    @torch.no_grad()
    def evaluate(self, charles_instance, x, f):
        norms = x.reshape(len(x), -1, 3).norm(dim=-1)
        return (norms.max(dim=-1).values <= self.radius).float()

    def sample(self, charles_instance, num_samples):
        return torch.full((num_samples,), self.target)

    def to_dict(self):
        return {"radius": self.radius, "target": self.target}


def read_founder(path: Path) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=float)


def relax(flat: np.ndarray, calc: LennardJones) -> tuple[np.ndarray, float]:
    atoms = Atoms(f"Ar{len(flat) // 3}", positions=flat.reshape(-1, 3), pbc=False)
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=FMAX, steps=FIRE_STEPS)
    return atoms.get_positions().reshape(-1), float(atoms.get_potential_energy())


def search(system: str, seed: int, founder: np.ndarray, budget: int) -> dict[str, float]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    n = len(founder)
    dim = 3 * n
    tally = Tally(budget, REFERENCES[system] + HIT_TOL)
    calc = LennardJones(tally)
    radius = float(np.linalg.norm(founder, axis=1).max()) * ORIGIN_SCALE
    conditions = (
        OriginCondition(radius),
        KNNNoveltyCondition(k=10, metric=2.0, beta=10.0, weight_by_fitness=True),
    )
    mlp = MLP(
        num_params=dim,
        num_hidden=96,
        num_layers=8,
        activation="LeakyReLU",
        num_conditions=len(conditions),
    )
    es = CHARLES(
        num_params=dim,
        conditions=conditions,
        model=DDIM(nn=mlp, num_steps=5000),
        popsize=POPSIZE,
        sigma_init=SIGMA_INIT,
        x0=torch.tensor(founder.reshape(-1), dtype=torch.float32),
        is_genetic_algorithm=True,
        selection_pressure=20.0,
        adaptive_selection_pressure=True,
        elite_ratio=0.15,
        crossover_ratio=0.125,
        mutation_rate=0.05,
        readaptation=False,
        forget_best=False,
        weight_decay=0.0,
        diff_optim="Adam",
        diff_lr=0.01,
        diff_weight_decay=1e-5,
        diff_batch_size=256,
        diff_max_epoch=1000,
        diff_continuous_training=False,
        buffer_size=1000,
        training_interval=1,
    )
    started = time.perf_counter()
    generations = 0
    try:
        for _ in range(GENERATIONS):
            samples = es.ask().detach().cpu().numpy().astype(float)
            relaxed = np.empty_like(samples)
            fitness = np.empty(len(samples))
            for i, sample in enumerate(samples):
                relaxed[i], energy = relax(sample, calc)
                fitness[i] = -energy
            es.tell(
                torch.tensor(fitness, dtype=torch.float32),
                parameters=torch.tensor(relaxed, dtype=torch.float32),
            )
            generations += 1
    except Spent:
        pass
    wall = time.perf_counter() - started
    hit = tally.hit_at is not None
    return {
        "hit": int(hit),
        "evals_to_hit": tally.hit_at if hit else budget,
        "charged": tally.calls,
        "wall_s": wall,
        "best": tally.best,
        "pot_s": tally.inside,
        "outside_s": max(wall - tally.inside, 0.0),
        "generations": generations,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=sorted(REFERENCES))
    parser.add_argument("seed", type=int)
    parser.add_argument("--founder", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args(argv[1:])
    founder = read_founder(args.founder)
    if founder.shape != (int(args.system[2:]), 3):
        raise SystemExit(f"founder shape {founder.shape} is not {args.system}")
    row = search(args.system, args.seed, founder, args.budget)
    print("system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s,generations")
    print(
        f"{args.system},gad,{args.seed},{args.budget},{row['hit']},{row['evals_to_hit']},"
        f"{row['charged']},{row['wall_s']:.6f},{row['best']:.8f},{row['pot_s']:.6f},"
        f"{row['outside_s']:.6f},{row['generations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
