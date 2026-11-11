#!/usr/bin/env python3
"""One seeded run of the published bonobo optimiser.

The search is ``bonobo_optimizer`` from
https://github.com/bhriguchak/modified_bo_framework_codes
(commit 47c6e9069fcb4082ede44a83d8ac6e8d4c53ef50). The initial population is
the ``pyar-tabu --make-composite`` command that ``initial_pop.py`` already
shells out to, taken from PyAR commit 4c45c115d4a4812f85e59cc8802a1251bb9ac94d,
the tree that still installs that script.

Their committed ``seed.xyz`` and ``monomer.xyz`` label the atom ``X``. PyAR
gives ``X`` a covalent radius of 0, and the tabu placer then never declares
contact. ``bo.py`` already parses the cluster as carbon (``formula = 'C89'``
in the published driver), so this arm writes that same one-atom geometry
with the symbol C. The half-width of the box stays 3.5, the value in their
default ``bounds``. The length of the box is ``3N`` because the published
default is 267 numbers, which is 89 atoms and will not reshape at 38 or 75.

A local relaxation is one call of their ``optimize_structure`` (L-BFGS-B,
``maxiter`` 5000). Their initial loop hands that function an ``(N, 3)``
array, and ``len(array) // 3`` then under-counts the atoms; the generation
loop already passes a flat vector. This arm flattens both.

The preprint states a population of 20 and, for the LJ38 test without the
initialiser, a cap of 1500 iterations. ``pyar-tabu --make-composite`` writes
100 files, and the first head-to-head relaxed all of them before spending
the rest of a 2000-relaxation cap on the loop, which is about 95 generations.
``LJ_BONOBO_POPULATION`` keeps that many of the generated structures, and
``LJ_BONOBO_MAX_ITER`` is the generation cap. The run still stops at the
relaxation budget or at the first relaxation inside the Cambridge window.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path


HIT_TOL = 1.0e-4
BOX = 3.5
POPULATION = 20


class BudgetReached(Exception):
    pass


class HitReached(Exception):
    pass


def _write_carbon_atom(path: Path) -> None:
    path.write_text("1\ntrial\nC 0.0 0.0 0.0\n")


def run_seed(n: int, seed: int, budget: int, reference: float, work: Path, upstream: Path) -> dict:
    if os.uname().nodename != "bench":
        raise SystemExit("uts hostname is not bench")
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("result_*.xyz"):
        stale.unlink()
    for name in ("newcost.txt", "traj.xyz"):
        path = work / name
        if path.exists():
            path.unlink()
    _write_carbon_atom(work / "seed.xyz")
    _write_carbon_atom(work / "monomer.xyz")

    os.environ["LJ_BONOBO_SEED"] = str(seed)
    sys.path.insert(0, str(upstream))
    os.chdir(work)

    import numpy as np

    import bo
    import lj
    import lbfgs_opt

    population_size = int(os.environ.get("LJ_BONOBO_POPULATION", "0"))
    max_iter = int(os.environ.get("LJ_BONOBO_MAX_ITER", "0")) or budget
    loop_n = population_size or POPULATION
    state = {"n": 0, "hit": None, "best": float("inf")}
    original = lbfgs_opt.optimize_structure
    original_init = bo.initialize_population

    def initialize_population(num_atoms):
        population = original_init(num_atoms)
        if population_size and len(population) > population_size:
            population = population[:population_size]
        if population_size and len(population) < population_size:
            raise RuntimeError(
                f"pyar-tabu returned {len(population)} structures, need {population_size}"
            )
        return population

    bo.initialize_population = initialize_population

    def optimize_structure(positions, epsilon=1.0):
        # The initial loop passes an (N, 3) array. len() on that is N, and
        # their N_atoms = len // 3 then relaxes a third of the cluster. The
        # generation loop already flattens. Flatten here too.
        flat = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1)
        if state["hit"] is not None or state["n"] >= budget:
            raise BudgetReached()
        state["n"] += 1
        coordinates, energy = original(flat, epsilon=epsilon)
        energy = float(energy)
        if energy < state["best"]:
            state["best"] = energy
        if state["hit"] is None and energy < reference + HIT_TOL:
            state["hit"] = state["n"]
            raise HitReached()
        if state["n"] >= budget:
            raise BudgetReached()
        return coordinates, energy

    # bo.py bound the name at import. Patch that binding, not only the module.
    bo.optimize_structure = optimize_structure
    # Their numba kernel compiles on the first call. Do that before the clock
    # so the seed pays for the search, not for the compiler.
    original(np.zeros(6))

    started = time.perf_counter()
    try:
        np.random.seed(seed)
        bo.bonobo_optimizer(
            lj.lennard_jones,
            bounds=[(-BOX, BOX)] * (3 * n),
            formula=f"C{n}",
            gm_energy=reference + HIT_TOL,
            N=loop_n,
            max_iter=max_iter,
            p_xgm_initial=0.035,
            scab=1.3,
            scsb=1.25,
            rcpp=0.0035,
            tsgs_factor_max=0.05,
        )
    except (BudgetReached, HitReached):
        pass
    wall_s = time.perf_counter() - started
    hit = state["hit"] is not None
    return {
        "arm": "bonobo",
        "n": n,
        "seed": seed,
        "hit": int(hit),
        "relaxations": state["n"],
        "relaxations_to_hit": state["hit"] if hit else "",
        "screens": 0,
        "best": state["best"],
        "wall_s": wall_s,
        "charged": "",
    }


def format_row(row: dict) -> str:
    best = row["best"]
    best_text = f"{best:.6f}" if isinstance(best, float) and best < 1e150 else "inf"
    return (
        f"arm={row['arm']} n={row['n']} seed={row['seed']} hit={row['hit']} "
        f"relaxations={row['relaxations']} relaxations_to_hit={row['relaxations_to_hit']} "
        f"screens={row['screens']} best={best_text} wall_s={row['wall_s']:.6f} "
        f"charged={row['charged']}"
    )


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: bonobo_arm.py <n> <seed> <budget> <reference> <upstream>",
            file=sys.stderr,
        )
        return 2
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    budget = int(sys.argv[3])
    reference = float(sys.argv[4])
    upstream = Path(sys.argv[5])
    work = Path.cwd()
    try:
        print(format_row(run_seed(n, seed, budget, reference, work, upstream)), flush=True)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
