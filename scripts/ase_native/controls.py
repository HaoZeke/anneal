# Copyright (c) anneal developers
"""ASE optimizers and searches with the mapped controls.

Basin hopping is the ASE 3.29 algorithm: a uniform Cartesian kick of
half-width ``dr``, an optional centre-of-mass restoration, a local
minimisation, and Metropolis acceptance at ``kT``. The local minimiser
is L-BFGS rather than the FIRE constructor default.

Minima hopping is the ASE 3.29 algorithm with trajectory and log files
omitted. The acceptance, temperature feedback and escape loop are the
ones in ``MinimaHopping``. The local minimiser is the same L-BFGS.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.optimize.lbfgs import LBFGS
from ase.optimize.minimahopping import MinimaHopping
from ase.utils.linesearch import LineSearch

from mapping import (  # noqa: E402
    ARMIJO,
    ASE_RELAX_STEPS,
    LBFGS_MEMORY,
    MD_MINIMA,
    WOLFE_CURVATURE,
    basin_dr,
    initial_hessian_alpha,
    local_maxstep,
    metropolis_kT,
    minima_ediff,
    minima_temperature_kelvin,
    minima_threshold,
    minima_timestep_fs,
)


def greedy_dmax(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Same pairing as ``ComparePositions._indistinguishable_compare``.

    Atoms are walked in order. Each takes the closest still-unmatched
    partner, ties keeping the earlier index, and the result is the largest
    of those distances. Partner distances that can win are recomputed with
    ``np.linalg.norm`` on the 3-vector, which is what ASE calls, so the
    selected index and the recorded distance match that walk.
    """
    remaining = np.asarray(pos2, dtype=np.float64)
    alive = np.ones(remaining.shape[0], dtype=bool)
    furthest = 0.0
    for point in np.asarray(pos1, dtype=np.float64):
        delta = remaining - point
        approx = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        approx = np.where(alive, approx, np.inf)
        nearest = float(np.min(approx))
        candidates = np.flatnonzero(approx <= nearest + 1.0e-9)
        best_index = int(candidates[0])
        best_distance = float("inf")
        for index in candidates:
            distance = float(np.linalg.norm(point - remaining[int(index)]))
            if distance < best_distance:
                best_distance = distance
                best_index = int(index)
        if best_distance > furthest:
            furthest = best_distance
        alive[best_index] = False
    return furthest


def install_fast_compare() -> None:
    """Replace the pure-Python atom walk with the same distances in NumPy."""
    from ase.optimize.minimahopping import ComparePositions

    if getattr(ComparePositions._indistinguishable_compare, "_native_fast", False):
        return
    original = ComparePositions._indistinguishable_compare

    def _fast(self, atoms1, atoms2):
        symbols1 = atoms1.get_chemical_symbols()
        symbols2 = atoms2.get_chemical_symbols()
        if symbols1 == symbols2 and len(set(symbols1)) == 1:
            return greedy_dmax(atoms1.positions, atoms2.positions)
        return original(self, atoms1, atoms2)

    _fast._native_fast = True
    ComparePositions._indistinguishable_compare = _fast


def make_local_optimizer(length_scale: float):
    """L-BFGS class closed over the mapped step cap and Wolfe constants."""
    maxstep = local_maxstep(length_scale)
    alpha = initial_hessian_alpha()

    class MatchedLBFGS(LBFGS):
        def __init__(self, atoms, logfile=None, trajectory=None, **kwargs):
            super().__init__(
                atoms,
                logfile=None,
                trajectory=None,
                maxstep=maxstep,
                memory=LBFGS_MEMORY,
                alpha=alpha,
                use_line_search=True,
                **kwargs,
            )

        def run(self, fmax=0.05, steps=None):
            cap = ASE_RELAX_STEPS if steps is None else min(int(steps), ASE_RELAX_STEPS)
            try:
                return super().run(fmax=fmax, steps=cap)
            except RuntimeError as err:
                if "LineSearch failed" not in str(err):
                    raise
                return False

        def line_search(self, r, g, e):
            p_size = float(np.sqrt((self.p**2).sum()))
            floor = float(np.sqrt(self.optimizable.ndofs() / 3 * 1e-10))
            if p_size <= floor:
                self.p /= p_size / floor
            search = LineSearch(get_gradient_norm=self.optimizable.gradient_norm)
            self.alpha_k, _energy, self.e0, self.no_update = search._line_search(
                self.func,
                self.fprime,
                r,
                self.p,
                g,
                e,
                self.e0,
                maxstep=self.maxstep,
                c1=ARMIJO,
                c2=WOLFE_CURVATURE,
                stpmax=50.0,
            )
            if self.alpha_k is None:
                raise RuntimeError("LineSearch failed!")

    MatchedLBFGS.__name__ = "MatchedLBFGS"
    return MatchedLBFGS


class QuietMinimaHopping(MinimaHopping):
    """ASE minima hopping that keeps minima in memory and writes no files."""

    def _log(self, cat="msg", message=None):
        return None

    def _read_minima(self):
        if getattr(self, "_minima", None) is None:
            self._minima = []
        return True

    def _record_minimum(self):
        self._minima.append(self._atoms.copy())

    def _startup(self):
        self._minima = []
        self._counter = 0
        self._optimize()
        self._check_results()
        self._counter += 1

    def _optimize(self):
        self._atoms.set_momenta(np.zeros(self._atoms.get_momenta().shape))
        with self._optimizer(self._atoms, logfile=None) as opt:
            opt.run(fmax=self._fmax)

    def _molecular_dynamics(self, resume=None):
        if resume is not None:
            raise RuntimeError("this run does not resume a trajectory")
        mincount = 0
        energies: list[float] = []
        oldpositions: list[np.ndarray] = []
        from ase.md.velocitydistribution import thermalize_momenta

        thermalize_momenta(
            self._atoms,
            self._temperature,
            exact_temperature=True,
            rng=self._rng,
        )
        dyn = VelocityVerlet(self._atoms, timestep=self._timestep * units.fs)
        with dyn:
            while mincount < self._mdmin:
                dyn.run(1)
                energies.append(float(self._atoms.get_potential_energy()))
                passed = self._passedminimum(energies)
                if passed:
                    mincount += 1
                oldpositions.append(self._atoms.positions.copy())
        self._atoms.positions = oldpositions[passed[0]]


def run_basin_hopping(atoms, seed: int, length_scale: float, energy_scale: float, fmax: float) -> None:
    """One basin-hopping search. Stops when the calculator raises.

    The kick is ASE's: every Cartesian component is Uniform(-dr, dr), then
    the centre of mass is put back. Anneal's all-points move uses the same
    box and does not restore the centre of mass. ``dr`` is the mapped
    half-width. No trajectory file is opened.
    """
    from ase.optimize.basin import BasinHopping
    from ase.optimize.optimize import Dynamics

    class QuietBasinHopping(BasinHopping):
        def __init__(self, atoms, temperature, optimizer, fmax, dr):
            self.kT = temperature
            self.optimizer = optimizer
            self.fmax = fmax
            self.dr = dr
            self.cm = atoms.get_center_of_mass()
            self.optimizer_logfile = None
            self.lm_trajectory = None
            Dynamics.__init__(self, atoms, logfile=None, trajectory=None)
            self.initialize()

    # BasinHopping.move draws from the global numpy stream.
    np.random.seed(seed)
    search = QuietBasinHopping(
        atoms,
        temperature=metropolis_kT(energy_scale),
        optimizer=make_local_optimizer(length_scale),
        fmax=fmax,
        dr=basin_dr(length_scale),
    )
    search.run(10**9)


def run_minima_hopping(atoms, seed: int, length_scale: float, energy_scale: float, fmax: float) -> None:
    """One minima-hopping search. Stops when the calculator raises."""
    atoms.set_masses(np.ones(len(atoms)))
    install_fast_compare()
    search = QuietMinimaHopping(
        atoms,
        T0=minima_temperature_kelvin(energy_scale, units.kB),
        Ediff0=minima_ediff(energy_scale),
        mdmin=MD_MINIMA,
        timestep=minima_timestep_fs(units.fs),
        optimizer=make_local_optimizer(length_scale),
        fmax=fmax,
        minima_threshold=minima_threshold(length_scale),
        minima_traj="minima.traj",
        logfile="hop.log",
        rng=np.random.RandomState(seed),
    )
    search(totalsteps=10**9)
