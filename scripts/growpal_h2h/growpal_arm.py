#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One GrowPAL search, using the documented LJ production loop.

The loop is the one printed in GrowPAL 0.1.6 as
``examples/LJ/x_run_growpal.py``: interstitial growth from the published
LJ10 seeds, a local L-BFGS-B relaxation, five reoptimisation rounds,
USR deduplication, and Ward selection of ``round(N * 1.8)`` structures
from each special-lineage cluster. The energy window and the 0.01 eV
early stop are theirs.

The published script has no evaluation ceiling. This driver counts every
energy-and-gradient call inside that L-BFGS-B, and the one energy call
their early stop makes, and stops when the count reaches the paper
budget. A hit is a finished quench at the target size whose energy is
within 0.001 of the Cambridge reference. Their update does not read the
seed; the seed is applied to the process RNGs so a library draw stays
tied to the replicate.
"""

from __future__ import annotations

import os
import time

# Thread caps must be in place before numpy, numba, or OpenMP initialise.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import gc
import random

GROWATOM = "Mo"
DTOL = 1.2
REOPT_ROUNDS = 5
DEDUP_TOLS = 0.99
DEDUP_TOLE = 0.1
EARLY_TOL = 0.01
HIT_TOL = 0.001
SELECTION_FACTOR = 1.8


class BudgetSpent(Exception):
    """The shared evaluation budget was reached inside a quench."""


class Found(Exception):
    """A quenched target structure met the Cambridge tolerance."""


class EarlyStop(Exception):
    """Their 0.01 eV check ended the growth before the target size."""


_ORIGINAL_OBJ = None


class Meter:
    def __init__(self, budget: int):
        self.budget = budget
        self.evals = 0
        self.pot_s = 0.0
        self.hit_at: int | None = None
        self.best: float | None = None
        self.reached = 10

    def charge(self, dt: float) -> None:
        self.pot_s += dt
        self.evals += 1
        if self.evals >= self.budget:
            raise BudgetSpent()


def _install_counter(meter: Meter):
    """Count pair evaluations. Reinstalling replaces the wrapper, not the kernel."""
    global _ORIGINAL_OBJ
    import aegon.optimization.potentials.lj as lj

    if _ORIGINAL_OBJ is None:
        _ORIGINAL_OBJ = lj._obj_and_grad

    def counted(x):
        if meter.evals >= meter.budget:
            raise BudgetSpent()
        mark = time.perf_counter()
        energy, gradient = _ORIGINAL_OBJ(x)
        meter.charge(time.perf_counter() - mark)
        return energy, gradient

    lj._obj_and_grad = counted
    return lj


def _energy(meter: Meter, positions) -> float:
    import aegon.optimization.potentials.lj as lj

    if meter.evals >= meter.budget:
        raise BudgetSpent()
    mark = time.perf_counter()
    value = float(lj.lj_energy(positions))
    meter.charge(time.perf_counter() - mark)
    return value


def _one_thread() -> None:
    """Their Ward step asks kneighbors_graph for every core. The graph is the
    same with one worker, and this session is pinned to one core."""
    import growpal.libclustering as clustering
    from sklearn.neighbors import kneighbors_graph

    def on_one_core(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return kneighbors_graph(*args, **kwargs)

    clustering.kneighbors_graph = on_one_core


def warmup() -> None:
    """Compile the pair kernel and the USR / MBTR paths before the clock."""
    import numba
    import numpy as np
    from ase import Atoms
    from aegon.discrimination.cluster_usr import deduplicate_by_usr
    from aegon.optimization.potentials.lj import lj_energy, lj_energy_and_gradient
    from growpal.libdescriptors import mbtr_comb_fast

    numba.set_num_threads(1)
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    lj_energy(positions)
    lj_energy_and_gradient(positions)
    left = Atoms("Mo4", positions=positions)
    left.info["e"] = -1.0
    left.info["i"] = "warmup"
    right = left.copy()
    right.info["e"] = -1.1
    deduplicate_by_usr([left, right], tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True)
    mbtr_comb_fast([left, right])


def assert_budget_stops_lbfgs() -> None:
    """The cap has to interrupt their optimiser, not merely wrap it."""
    import numpy as np
    import aegon.optimization.potentials.lj as lj

    meter = Meter(1)
    _install_counter(meter)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.2, 0.0], [0.1, 1.3, 0.2], [0.4, 0.2, 1.1]],
        dtype=np.float64,
    )
    try:
        lj._opt_positions(positions)
    except BudgetSpent:
        if meter.evals != 1:
            raise RuntimeError(f"budget of 1 charged {meter.evals}")
        return
    raise RuntimeError("L-BFGS-B ran past the evaluation cap")


def _optimise(mol_list, meter: Meter, target: int, level: float):
    """Serial stand-in for ``opt_LJ_parallel(..., n_jobs=1)``.

    ``_opt_positions`` is their worker: L-BFGS-B, gtol 1e-6, maxiter 1000,
    and the same maxcor rule. Stopping between structures is what lets the
    shared budget end the batch.
    """
    import numpy as np
    from ase import Atoms
    import aegon.optimization.potentials.lj as lj

    if not mol_list:
        return []
    symbols = mol_list[0].get_chemical_symbols()
    cell = mol_list[0].get_cell().copy()
    pbc = mol_list[0].get_pbc().copy()
    started = time.perf_counter()
    results = []
    for mol in mol_list:
        if meter.evals >= meter.budget:
            raise BudgetSpent()
        opt_pos, energy, converged = lj._opt_positions(np.asarray(mol.get_positions(), dtype=np.float64))
        info = dict(mol.info)
        info["e"] = float(energy)
        info["c"] = converged
        out = Atoms(symbols=symbols, positions=opt_pos, cell=cell, pbc=pbc, info=info)
        results.append(out)
        if len(out) == target:
            if meter.best is None or info["e"] < meter.best:
                meter.best = info["e"]
            if info["e"] <= level and meter.hit_at is None:
                meter.hit_at = meter.evals
                raise Found()
    elapsed = time.perf_counter() - started
    print(
        "Local OPT parallel at %5.2f s [%d -> %d]" % (elapsed, len(mol_list), len(results)),
        flush=True,
    )
    return results


def _consider(meter: Meter, energy: float, level: float) -> None:
    if meter.best is None or energy < meter.best:
        meter.best = energy
    if energy <= level and meter.hit_at is None:
        meter.hit_at = meter.evals
        raise Found()


def run_growpal(
    target: int, seed: int, budget: int, reference: float, seeds_path: str
) -> dict:
    """Grow from the published LJ10 seeds until ``target`` or the budget."""
    from aegon.data.lj import get_lj_cluster
    from aegon.discrimination.cluster_usr import deduplicate_by_usr, filter_against_reference_usr
    from aegon.io.xyz import readxyzs
    from aegon.population import cutter_energy, rename, sort_by_energy
    import numpy as np
    from growpal.libgrowpal import growpal_parallel
    from growpal.libselect_clustering import select_by_clustering

    _one_thread()
    stored = float(get_lj_cluster(target).info["e"])
    if abs(stored - reference) > 1e-6:
        raise RuntimeError(f"aegon LJ{target} energy {stored} != Cambridge {reference}")

    # The published growth loop does not draw random numbers. Seeding keeps
    # any library that does consult the process RNG tied to this replicate.
    random.seed(seed)
    np.random.seed(seed)
    meter = Meter(budget)
    level = reference + HIT_TOL
    _install_counter(meter)
    warmup()
    molecu0 = readxyzs(seeds_path)
    special_mols = []
    started = time.perf_counter()
    stop = "complete"
    try:
        for size in range(11, target + 1):
            meter.reached = size
            base_name = "LJ" + str(size).zfill(3)
            print("-------------------------------------------------------------------", flush=True)
            print(f"{base_name} START evals={meter.evals}", flush=True)
            special_mols = [mol for mol in special_mols if len(mol) == size - 1]
            sp_tags = []
            for index, special in enumerate(special_mols):
                special.info["i"] = f"_sp{index:03d}"
                sp_tags.append(f"_sp{index:03d}")
            grown_from = rename(molecu0, base_name, 5)
            molecu0 = growpal_parallel(
                grown_from + special_mols, GROWATOM, dtol=DTOL, n_cores=1
            )
            molecu1 = _optimise(molecu0, meter, target, level)
            special_candidates = []
            for tag in sp_tags:
                children = [
                    mol for mol in molecu1 if str(mol.info.get("i", "")).startswith(tag + "_")
                ]
                if children:
                    special_candidates.append(min(children, key=lambda mol: mol.info["e"]))
            special_mols = deduplicate_by_usr(
                special_candidates, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True
            )
            molecu0 = deduplicate_by_usr(molecu1, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True)
            del molecu1
            for _round in range(REOPT_ROUNDS):
                molecu1 = _optimise(molecu0, meter, target, level)
                molecu0 = deduplicate_by_usr(
                    molecu1, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True
                )
                del molecu1
                gc.collect()
            if not molecu0:
                raise RuntimeError(f"LJ{size} pool is empty after relaxation")
            global_min = molecu0[0]
            distinct = filter_against_reference_usr(
                [global_min], special_mols, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True, flag=1
            )
            if distinct:
                special_mols.append(global_min)
                special_mols = sort_by_energy(special_mols, abs_energy=True)
            if not special_mols:
                raise RuntimeError(f"LJ{size} has no special lineage")
            print(
                "Special cluster  : %d members (n=%d)" % (len(special_mols), size),
                flush=True,
            )
            e_best = _energy(meter, special_mols[0].get_positions())
            e_true = float(get_lj_cluster(size).info["e"])
            delta = e_best - e_true
            print(
                "Global min check: E_best=%.6f  E_true=%.6f  dE=%.4f" % (e_best, e_true, delta),
                flush=True,
            )
            if size == target:
                _consider(meter, e_best, level)
            elif delta > EARLY_TOL:
                print(
                    "STOP: global minimum not found for %s (dE=%.4f > 0.01 eV)"
                    % (base_name, delta),
                    flush=True,
                )
                raise EarlyStop()
            main_pool = filter_against_reference_usr(
                molecu0, special_mols, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True, flag=1
            )
            ecut = 5.0 * (size ** (2.0 / 3.0))
            selected_from = cutter_energy(main_pool, ecut)
            n_clusters = max(1, len(special_mols))
            selection_count = round(size * SELECTION_FACTOR)
            molecu0 = select_by_clustering(
                selected_from,
                selection_count=selection_count,
                n_clusters=n_clusters,
                n_jobs=1,
                use_connectivity=True,
                mono=True,
            )
            gc.collect()
    except Found:
        stop = "hit"
    except BudgetSpent:
        stop = "budget"
    except EarlyStop:
        stop = "early"
    wall = time.perf_counter() - started
    hit = meter.hit_at is not None
    evals_to_hit = meter.hit_at if hit else budget
    best = "" if meter.best is None else f"{meter.best:.8f}"
    return {
        "system": f"lj{target}",
        "arm": "growpal",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals_to_hit),
        "charged": str(meter.evals),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{meter.pot_s:.6f}",
        "best": best,
        "stop": stop,
        "reached": str(meter.reached),
    }


def per_eval(wall_s: float, pot_s: float, charged: int) -> tuple[float, float]:
    """Microseconds per charged evaluation inside and outside the potential."""
    if charged <= 0:
        raise ValueError("per-evaluation time needs a charged evaluation")
    inside = pot_s / charged * 1e6
    outside = (wall_s - pot_s) / charged * 1e6
    return inside, outside
