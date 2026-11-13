#!/usr/bin/env python3
# Copyright (c) anneal developers
"""One GrowPAL search on a Lennard-Jones or Morse cluster, under a call cap.

Lennard-Jones is the production loop in GrowPAL's ``examples/LJ``:
interstitial growth from the published LJ10 seeds (element Mo, ``dtol``
1.2), aegon's L-BFGS-B (``gtol`` 1e-6, ``maxiter`` 1000), five
reoptimisation rounds, USR deduplication at 0.99 and 0.1, the energy
window ``5 N^(2/3)``, and Ward selection of ``round(N * 1.8)`` structures
per cluster. After each size the loop compares the best structure with
aegon's Lennard-Jones table and stops when it is more than 0.01 above it.
A candidate hit is scored on the full pair sum.

Morse goes through aegon's generic ASE strategy with ASE's
``MorsePotential``: ``epsilon`` 1, ``rho0`` the Cambridge range, ``r0`` 3,
smooth cutoff from 1000 r0. Relaxation
is aegon's ASE worker with its defaults: ASE LBFGS, ``fmax`` 0.01,
500 steps. The LJ10 seeds are relaxed on the Morse surface first, and
those calls are charged. aegon has no Morse table, so there is no
per-size early stop.

Every energy-and-gradient call is charged. On Lennard-Jones that is each
call of aegon's ``_obj_and_grad`` and the one energy the early stop reads.
On Morse it is each ``MorsePotential.calculate``. The search ends at the
budget, at the first quench of the target size within 0.001 of the
Cambridge minimum, or at the early stop. Their update draws no random
numbers; the seed is applied to the process RNGs and stored.

    growpal_arm.py run <system> <seed> <budget> <reference> <seeds.xyz>
    growpal_arm.py identity <system> <coords> <scale>
    growpal_arm.py check
"""

from __future__ import annotations

import os
import sys
import time

# Thread caps must be in place before numpy, numba, or OpenMP initialise.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import gc
import json
import random

GROWATOM = "Mo"
DTOL = 1.2
REOPT_ROUNDS = 5
DEDUP_TOLS = 0.99
DEDUP_TOLE = 0.1
EARLY_TOL = 0.01
HIT_TOL = 0.001
SELECTION_FACTOR = 1.8
LJ_PAIR_DISTANCE = 3.0
MORSE_R0 = 3.0
MORSE_CUTOFF = (1000.0, 1001.0)
ASE_FMAX = 0.01
ASE_STEPS = 500
ASE_OPTIMIZER = "LBFGS"

SYSTEMS = {
    "lj38": ("lj", 38, None),
    "lj55": ("lj", 55, None),
    "lj75": ("lj", 75, None),
    "lj98": ("lj", 98, None),
    "m38r6": ("morse", 38, 6.0),
}


class BudgetSpent(Exception):
    """The shared evaluation budget was reached."""


class Found(Exception):
    """A quenched target structure met the Cambridge tolerance."""


class EarlyStop(Exception):
    """Their 0.01 check ended the growth before the target size."""


class Meter:
    def __init__(self, budget: int, target: int, level: float):
        self.budget = budget
        self.target = target
        self.level = level
        self.evals = 0
        self.pot_s = 0.0
        self.hit_at: int | None = None
        self.best: float | None = None
        self.reached = 10

    def open(self) -> None:
        if self.evals >= self.budget:
            raise BudgetSpent()

    def charge(self, dt: float) -> None:
        self.pot_s += dt
        self.evals += 1

    def consider(self, size: int, energy: float) -> None:
        if size != self.target:
            return
        if self.best is None or energy < self.best:
            self.best = energy
        if energy <= self.level and self.hit_at is None:
            self.hit_at = self.evals
            raise Found()


_ORIGINALS: dict[str, object] = {}


def install_lj_counter(meter: Meter) -> None:
    """Count aegon's pair calls. Reinstalling replaces the wrapper, not the kernel."""
    import aegon.optimization.potentials.lj as lj

    original = _ORIGINALS.setdefault("lj", lj._obj_and_grad)

    def counted(x):
        meter.open()
        mark = time.perf_counter()
        out = original(x)
        meter.charge(time.perf_counter() - mark)
        return out

    lj._obj_and_grad = counted


def install_morse_counter(meter: Meter) -> None:
    from ase.calculators.morse import MorsePotential

    original = _ORIGINALS.setdefault("morse", MorsePotential.calculate)

    def counted(self, *args, **kwargs):
        meter.open()
        mark = time.perf_counter()
        original(self, *args, **kwargs)
        meter.charge(time.perf_counter() - mark)

    MorsePotential.calculate = counted


def uninstall_counters() -> None:
    if "lj" in _ORIGINALS:
        import aegon.optimization.potentials.lj as lj

        lj._obj_and_grad = _ORIGINALS["lj"]
    if "morse" in _ORIGINALS:
        from ase.calculators.morse import MorsePotential

        MorsePotential.calculate = _ORIGINALS["morse"]


def morse_kwargs(rho: float) -> dict[str, float]:
    return {
        "epsilon": 1.0,
        "rho0": rho,
        "r0": MORSE_R0,
        "rcut1": MORSE_CUTOFF[0],
        "rcut2": MORSE_CUTOFF[1],
    }


def one_thread() -> None:
    """Ward clustering builds its k-NN graph with one worker."""
    import growpal.libclustering as clustering
    from sklearn.neighbors import kneighbors_graph

    def on_one_core(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return kneighbors_graph(*args, **kwargs)

    clustering.kneighbors_graph = on_one_core


def warmup(kind: str, rho: float | None, warm_xyz: str | None = None) -> None:
    """Compile the numba paths before the clock, growing the LJ9 seeds once."""
    import numba
    import numpy as np
    from ase import Atoms
    from aegon.discrimination.cluster_usr import deduplicate_by_usr
    from growpal.libdescriptors import mbtr_comb_fast

    numba.set_num_threads(1)
    if warm_xyz is not None:
        from aegon.io.xyz import readxyzs
        from growpal.libgrowpal import process_single_molecule

        for index, mol in enumerate(readxyzs(warm_xyz)):
            mol.info["i"] = f"warm{index:03d}"
            process_single_molecule((mol, GROWATOM, DTOL))
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    if kind == "lj":
        from aegon.optimization.potentials.lj import lj_energy, lj_energy_and_gradient

        lj_energy(positions)
        lj_energy_and_gradient(positions)
    else:
        from ase.calculators.morse import MorsePotential

        probe = Atoms("Mo4", positions=positions)
        probe.calc = MorsePotential(**morse_kwargs(rho))
        probe.get_potential_energy()
    left = Atoms("Mo4", positions=positions)
    left.info["e"] = -1.0
    left.info["i"] = "warmup"
    right = left.copy()
    right.info["e"] = -1.1
    deduplicate_by_usr([left, right], tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True)
    mbtr_comb_fast([left, right])


class LjRelax:
    """Serial ``opt_LJ_parallel``: their worker, one structure at a time."""

    def __init__(self, meter: Meter):
        self.meter = meter

    def __call__(self, mol_list):
        import numpy as np
        from ase import Atoms
        import aegon.optimization.potentials.lj as lj

        out = []
        for mol in mol_list:
            self.meter.open()
            pos, energy, converged = lj._opt_positions(
                np.asarray(mol.get_positions(), dtype=np.float64)
            )
            info = dict(mol.info)
            info["e"] = float(energy)
            info["c"] = converged
            relaxed = Atoms(
                symbols=mol.get_chemical_symbols(),
                positions=pos,
                cell=mol.get_cell().copy(),
                pbc=mol.get_pbc().copy(),
                info=info,
            )
            out.append(relaxed)
            self.score(relaxed, info["e"])
        return out

    def score(self, mol, energy: float) -> None:
        """A candidate hit is scored on the full pair sum."""
        if len(mol) == self.meter.target and energy <= self.meter.level + EARLY_TOL:
            energy = lj_exact(mol.get_positions())
        self.meter.consider(len(mol), energy)

    def energy(self, mol) -> float:
        import aegon.optimization.potentials.lj as lj

        self.meter.open()
        mark = time.perf_counter()
        value = float(lj.lj_energy(mol.get_positions()))
        self.meter.charge(time.perf_counter() - mark)
        return value


class MorseRelax:
    """Serial ``opt_ASE_parallel``: their ASE worker, one structure at a time."""

    def __init__(self, meter: Meter, rho: float):
        self.meter = meter
        self.kwargs = morse_kwargs(rho)

    def __call__(self, mol_list):
        from aegon.optimization.potentials.ase_generic import _opt_ASE_single

        out = []
        for mol in mol_list:
            self.meter.open()
            relaxed = _opt_ASE_single(
                mol,
                "ase.calculators.morse",
                "MorsePotential",
                self.kwargs,
                True,
                "FrechetCellFilter",
                ASE_FMAX,
                ASE_STEPS,
                ASE_OPTIMIZER,
            )
            relaxed.calc = None
            out.append(relaxed)
            self.meter.consider(len(relaxed), float(relaxed.info["e"]))
        return out


def assert_cap_interrupts(kind: str, rho: float | None) -> None:
    """The cap has to stop their optimiser mid-quench, not only between quenches."""
    import numpy as np
    from ase import Atoms

    positions = np.array(
        [[0.0, 0.0, 0.0], [3.6, 0.6, 0.0], [0.3, 3.9, 0.6], [1.2, 0.6, 3.3]],
        dtype=np.float64,
    )
    meter = Meter(1, 99, -1e9)
    if kind == "lj":
        install_lj_counter(meter)
        relax = LjRelax(meter)
    else:
        install_morse_counter(meter)
        relax = MorseRelax(meter, rho)
    try:
        relax([Atoms("Mo4", positions=positions)])
    except BudgetSpent:
        if meter.evals != 1:
            raise RuntimeError(f"{kind} budget of 1 charged {meter.evals}")
        return
    finally:
        uninstall_counters()
    raise RuntimeError(f"{kind} relaxation ran past the evaluation cap")


def _pair_distances(positions):
    import numpy as np

    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff**2).sum(-1))[np.triu_indices(len(positions), 1)]


def morse_exact(positions, rho: float, r0: float) -> float:
    import numpy as np

    e = np.exp(rho * (1.0 - _pair_distances(positions) / r0))
    return float((e * (e - 2.0)).sum())


def lj_exact(positions, cutoff: float | None = None) -> float:
    """Lennard-Jones sum at aegon's length, optionally cut at ``cutoff``."""
    r = _pair_distances(positions)
    if cutoff is not None:
        r = r[r < cutoff]
    sigma = LJ_PAIR_DISTANCE / 2.0 ** (1.0 / 6.0)
    sr6 = (sigma / r) ** 6
    return float((4.0 * sr6 * (sr6 - 1.0)).sum())


def check() -> None:
    """Cap and potential checks, run before the timed session."""
    import numpy as np
    from ase import Atoms
    from ase.calculators.morse import MorsePotential

    assert_cap_interrupts("lj", None)
    assert_cap_interrupts("morse", 6.0)
    rng = np.random.default_rng(7)
    unit = rng.uniform(-1.6, 1.6, size=(38, 3))
    probe = Atoms("Mo38", positions=unit * MORSE_R0)
    probe.calc = MorsePotential(**morse_kwargs(6.0))
    ase_energy = float(probe.get_potential_energy())
    exact = morse_exact(unit, 6.0, 1.0)
    if abs(ase_energy - exact) > 1e-9 * max(1.0, abs(exact)):
        raise RuntimeError(f"ASE Morse {ase_energy} != Cambridge form {exact}")
    print(json.dumps({"check": "ok", "ase_morse": ase_energy, "exact_morse": exact}))


def identity(system: str, coords_path: str, scale: float) -> None:
    """GrowPAL's energy of an anneal geometry, scaled to GrowPAL lengths."""
    import numpy as np
    from ase import Atoms

    kind, _n, rho = SYSTEMS[system]
    unit = np.loadtxt(coords_path, dtype=np.float64).reshape(-1, 3)
    positions = unit * scale
    if kind == "lj":
        import aegon.optimization.potentials.lj as lj

        energy = float(lj.lj_energy(positions))
        exact = lj_exact(positions)
        same_form = lj_exact(positions, cutoff=float(lj.cutoff))
        cutoff = float(lj.cutoff) / scale
    else:
        from ase.calculators.morse import MorsePotential

        atoms = Atoms(f"Mo{len(positions)}", positions=positions)
        atoms.calc = MorsePotential(**morse_kwargs(rho))
        energy = float(atoms.get_potential_energy())
        exact = morse_exact(unit, rho, 1.0)
        same_form = exact
        cutoff = MORSE_CUTOFF[0] * MORSE_R0 / scale
    print(
        json.dumps(
            {
                "system": system,
                "growpal": energy,
                "exact": exact,
                "same_form": same_form,
                "cutoff": cutoff,
                "rmax": float(_pair_distances(unit).max()),
            }
        )
    )


def run(system: str, seed: int, budget: int, reference: float, seeds_path: str) -> dict:
    """Grow from the LJ10 seeds until the target size or the budget."""
    import numpy as np
    from aegon.discrimination.cluster_usr import (
        deduplicate_by_usr,
        filter_against_reference_usr,
    )
    from aegon.io.xyz import readxyzs
    from aegon.population import cutter_energy, rename, sort_by_energy
    from growpal.libgrowpal import growpal_parallel
    from growpal.libselect_clustering import select_by_clustering

    kind, target, rho = SYSTEMS[system]
    table = None
    if kind == "lj":
        from aegon.data.lj import get_lj_cluster

        table = get_lj_cluster
        stored = float(get_lj_cluster(target).info["e"])
        if abs(stored - reference) > 1e-6:
            raise RuntimeError(f"aegon LJ{target} {stored} != Cambridge {reference}")
    one_thread()
    random.seed(seed)
    np.random.seed(seed)
    warmup(kind, rho, os.path.join(os.path.dirname(seeds_path), "LJ009.xyz"))
    meter = Meter(budget, target, reference + HIT_TOL)
    if kind == "lj":
        install_lj_counter(meter)
        relax = LjRelax(meter)
    else:
        install_morse_counter(meter)
        relax = MorseRelax(meter, rho)
    molecu0 = readxyzs(seeds_path)
    special_mols = []
    started = time.perf_counter()
    stop = "complete"
    try:
        if kind == "morse":
            molecu0 = sort_by_energy(relax(molecu0), True)
        for size in range(11, target + 1):
            meter.reached = size
            base_name = "LJ" + str(size).zfill(3)
            print(f"{base_name} START evals={meter.evals}", flush=True)
            special_mols = [mol for mol in special_mols if len(mol) == size - 1]
            sp_tags = []
            for index, special in enumerate(special_mols):
                special.info["i"] = f"_sp{index:03d}"
                sp_tags.append(f"_sp{index:03d}")
            grown_from = rename(molecu0, base_name, 5)
            molecu0 = growpal_parallel(grown_from + special_mols, GROWATOM, dtol=DTOL, n_cores=1)
            molecu1 = relax(molecu0)
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
                molecu1 = relax(molecu0)
                molecu0 = deduplicate_by_usr(
                    molecu1, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True
                )
                del molecu1
                gc.collect()
            if not molecu0:
                raise RuntimeError(f"{base_name} pool is empty after relaxation")
            global_min = molecu0[0]
            distinct = filter_against_reference_usr(
                [global_min], special_mols, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True, flag=1
            )
            if distinct:
                special_mols.append(global_min)
                special_mols = sort_by_energy(special_mols, True)
            if not special_mols:
                raise RuntimeError(f"{base_name} has no special lineage")
            if table is not None:
                e_best = relax.energy(special_mols[0])
                delta = e_best - float(table(size).info["e"])
                print(f"{base_name} E_best={e_best:.6f} dE={delta:.4f}", flush=True)
                relax.score(special_mols[0], e_best)
                if size != target and delta > EARLY_TOL:
                    raise EarlyStop()
            else:
                print(f"{base_name} E_best={special_mols[0].info['e']:.6f}", flush=True)
            if size == target:
                break
            main_pool = filter_against_reference_usr(
                molecu0, special_mols, tols=DEDUP_TOLS, tole=DEDUP_TOLE, mono=True, flag=1
            )
            selected_from = cutter_energy(main_pool, 5.0 * (size ** (2.0 / 3.0)))
            molecu0 = select_by_clustering(
                selected_from,
                selection_count=round(size * SELECTION_FACTOR),
                n_clusters=max(1, len(special_mols)),
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
    return {
        "system": system,
        "arm": "growpal",
        "seed": str(seed),
        "budget": str(budget),
        "hit": "1" if hit else "0",
        "evals_to_hit": str(meter.hit_at if hit else budget),
        "charged": str(meter.evals),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{meter.pot_s:.6f}",
        "best": "" if meter.best is None else f"{meter.best:.8f}",
        "stop": stop,
        "reached": str(meter.reached),
    }


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "check":
        check()
        return 0
    if len(argv) == 5 and argv[1] == "identity":
        identity(argv[2], argv[3], float(argv[4]))
        return 0
    if len(argv) == 7 and argv[1] == "run":
        system, seed, budget = argv[2], int(argv[3]), int(argv[4])
        reference, seeds_path = float(argv[5]), argv[6]
        if system not in SYSTEMS:
            raise SystemExit(f"GrowPAL has no {system}")
        row = run(system, seed, budget, reference, seeds_path)
        if int(row["charged"]) > budget:
            raise SystemExit(f"charged {row['charged']} past {budget}")
        print("ROW " + json.dumps(row), flush=True)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
