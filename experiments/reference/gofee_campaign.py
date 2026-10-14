"""GOFEE baseline through AGOX, configured from the current documentation.

The search block below reproduces the documented GOFEE example
(agox.gitlab.io/agox/algorithms/gofee.html): LowerConfidenceBoundAcquisitor
with kappa=2, KMeansSampler with sample_size=10, ParallelCollector with the
staged num_candidates schedule, ParallelRelaxPostprocess on the acquisition
surface, and LocalOptimizationEvaluator taking one true step per iteration.
Only the system (template, symbols, calculator) and the stopping rule differ:
a counting wrapper around the true calculator logs every real force call and
raises once the matched budget is spent, so the comparison counts true calls
and never surrogate evaluations.

Usage: gofee_campaign.py --system {w6,lj38,lj75,lj98} --budget N --seed S
Writes calls_<system>_s<seed>.dat lines: call_index energy.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from agox import AGOX
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.collectors import ParallelCollector
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.models.descriptors.fingerprint import Fingerprint
from agox.models.GPR import GPR
from agox.models.GPR.kernels import RBF, Noise
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.priors import Repulsive
from agox.postprocessors import ParallelRelaxPostprocess
from agox.samplers import KMeansSampler
import sys


class KeepZeroStepRelax(ParallelRelaxPostprocess):
    """A surrogate relaxation of zero steps is still a proposal.

    The pool relaxer drops a candidate whose optimizer took no step.
    Once every proposal is already under the surrogate force threshold,
    the true calculator is never called and the iteration counter runs
    on alone.
    """

    def _set_results(self, candidates, results):
        kept = []
        for candidate, result in zip(candidates, results):
            self.remove_constraints(candidate)
            if result.relaxation_steps > 0:
                self._set_result_to_candidate(candidate, result)
            kept.append(candidate)
        return kept


class BudgetExhausted(Exception):
    pass


class CountingCalc(Calculator):
    """Delegates to the true calculator, logging and budgeting every call."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base, budget, logpath):
        super().__init__()
        self.base = base
        self.budget = budget
        self.count = 0
        self.log = open(logpath, "a", buffering=1)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        if self.count >= self.budget:
            raise BudgetExhausted()
        self.count += 1
        work = atoms.copy()
        work.calc = self.base
        energy = work.get_potential_energy()
        forces = work.get_forces()
        self.log.write(f"{self.count} {energy:.8f}\n")
        self.results = {"energy": energy, "forces": forces}


def make_system(name):
    """Calculator, symbols, and box geometry in AGOX's native length scale.

    Everything AGOX defaults to (fingerprint grids, rattle amplitudes, the
    repulsive prior) is parameterized in Angstrom for real elements, so the
    Lennard-Jones system runs in argon units (sigma 3.4 A, epsilon 10.4 meV)
    rather than reduced units; the harvest divides by epsilon to compare
    against the reduced-unit global minimum.
    """
    if name == "w6":
        from xtb.ase.calculator import XTB

        return XTB(method="GFN2-xTB"), "O6H12", 16.0, 10.0, {}
    if name in {"lj38", "lj75", "lj98"}:
        from ase.calculators.lj import LennardJones

        n = int(name[2:])
        calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=10.2, smooth=False)
        # The LJ38 box is cell 24 A and confinement 16 A. Larger clusters
        # use the same box scaled by (N/38)^(1/3). Generators window bond
        # lengths as [c1, c2] x summed covalent radii (2.12 A for an Ar
        # pair); the defaults land inside the LJ repulsive wall, so the
        # window brackets the LJ minimum at 3.82 A.
        scale = (n / 38.0) ** (1.0 / 3.0)
        return calc, f"Ar{n}", 24.0 * scale, 16.0 * scale, {"c1": 1.6, "c2": 2.2}
    raise SystemExit(f"unknown system {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["w6", "lj38", "lj75", "lj98"])
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    true_calc, symbols, cell_size, conf_size, genkw = make_system(args.system)
    calc = CountingCalc(
        true_calc, args.budget, f"calls_{args.system}_s{args.seed}.dat"
    )

    template = Atoms("", cell=np.eye(3) * cell_size)
    confinement_cell = np.eye(3) * conf_size
    corner = (cell_size - conf_size) / 2.0
    confinement_corner = np.array([corner, corner, corner])
    environment = Environment(
        template=template,
        symbols=symbols,
        confinement_cell=confinement_cell,
        confinement_corner=confinement_corner,
    )

    database = Database(
        filename=f"db_{args.system}_s{args.seed}.db", order=5
    )

    descriptor = Fingerprint(environment=environment)
    beta = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(0.01, (0.01, 0.01))
    model = GPR(
        descriptor=descriptor, kernel=kernel, database=database, prior=Repulsive()
    )

    sample_size = 10
    sampler = KMeansSampler(
        descriptor=descriptor, database=database, sample_size=sample_size
    )
    rattle_generator = RattleGenerator(**environment.get_confinement(), **genkw)
    random_generator = RandomGenerator(**environment.get_confinement(), **genkw)
    generators = [random_generator, rattle_generator]
    num_candidates = {0: [10, 0], 5: [3, 7]}

    collector = ParallelCollector(
        generators=generators,
        sampler=sampler,
        environment=environment,
        num_candidates=num_candidates,
        order=1,
    )

    acquisitor = LowerConfidenceBoundAcquisitor(model=model, kappa=2, order=3)

    relaxer = KeepZeroStepRelax(
        model=acquisitor.get_acquisition_calculator(),
        constraints=environment.get_constraints(),
        optimizer_run_kwargs={"steps": 5},
        start_relax=8,
        order=2,
    )

    evaluator = LocalOptimizationEvaluator(
        calc,
        gets={"get_key": "prioritized_candidates"},
        optimizer_kwargs={"logfile": None},
        optimizer_run_kwargs={"fmax": 0.05, "steps": 1},
        constraints=environment.get_constraints(),
        store_trajectory=False,
        order=4,
    )

    agox = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=args.seed)
    # The budget, not the iteration count, is the stopping rule: iterations
    # are set high enough that the counter always trips first.
    try:
        agox.run(N_iterations=args.budget)
    except BudgetExhausted:
        pass
    print(
        f"gofee {args.system} seed {args.seed}: true calls {calc.count}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
