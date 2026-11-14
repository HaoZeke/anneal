"""Threaded energy and force calls at the minimum-image sizes.

``threads`` is applied with ``omp_set_num_threads`` before the geometry
is built, so the same file runs on a build that ignores the count and
on one that folds pairs across that many threads. asv does not time or
track ``setup``.
"""

from __future__ import annotations

import ctypes
import math
import os

import numpy as np

REPEAT = 6
ATOMS = (256, 1024, 4096)
THREADS = (1, 2, 4, 8)

_DIAMOND = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.25),
    (0.75, 0.25, 0.75),
    (0.25, 0.75, 0.75),
)
_FCC = (
    (0.00, 0.00, 0.00),
    (0.50, 0.50, 0.00),
    (0.50, 0.00, 0.50),
    (0.00, 0.50, 0.50),
)


def _set_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "false"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    lib = ctypes.CDLL("libgomp.so.1")
    lib.omp_set_dynamic(0)
    lib.omp_set_num_threads(int(threads))


def _lattice(atoms: int, lattice: float, basis, atomic_number: int):
    per_cell = len(basis)
    ncell = 1
    while per_cell * ncell * ncell * ncell < atoms:
        ncell += 1
    coords = []
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                for site in basis:
                    if len(coords) >= atoms:
                        break
                    coords.append(
                        (
                            (ix + site[0]) * lattice,
                            (iy + site[1]) * lattice,
                            (iz + site[2]) * lattice,
                        )
                    )
                if len(coords) >= atoms:
                    break
            if len(coords) >= atoms:
                break
        if len(coords) >= atoms:
            break
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.full(atoms, atomic_number, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * (ncell * lattice)
    return positions, atom_types, box


def _diamond(atoms: int):
    return _lattice(atoms, 5.431, _DIAMOND, 14)


def _aluminium(atoms: int):
    return _lattice(atoms, 4.05, _FCC, 13)


def _lj_cluster(atoms: int):
    side = 1
    while side * side * side < atoms:
        side += 1
    spacing = 2.0 ** (1.0 / 6.0)
    coords = []
    for index in range(side * side * side):
        if len(coords) == atoms:
            break
        x = index % side
        y = (index // side) % side
        z = index // (side * side)
        coords.append((x * spacing, y * spacing, z * spacing))
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.ones(atoms, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 80.0
    return positions, atom_types, box


def _energy_force(pot, positions, atom_types, box) -> None:
    energy, forces, _variance = pot(positions, atom_types, box)
    if getattr(forces, "shape", None) != positions.shape:
        raise AssertionError(getattr(forces, "shape", None))
    if not math.isfinite(float(energy)):
        raise AssertionError("energy is not finite")


class _Timed:
    params = (list(ATOMS), list(THREADS))
    param_names = ["atoms", "threads"]
    repeat = REPEAT
    sample_time = 0.01
    min_run_count = 1
    timeout = 600

    def time_energy_force(self, atoms: int, threads: int) -> None:
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class _Tracked:
    params = (list(ATOMS), list(THREADS))
    param_names = ["atoms", "threads"]
    number = 1
    repeat = REPEAT
    timeout = 600

    def ray_energy_force(self, atoms: int, threads: int) -> None:
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


def _prepare(self, atoms: int, threads: int, geometry, pot) -> None:
    _set_threads(threads)
    self.positions, self.atom_types, self.box = geometry(atoms)
    self.pot = pot
    _energy_force(self.pot, self.positions, self.atom_types, self.box)


class LJCluster(_Timed):
    """Lennard-Jones cluster at 256, 1024 and 4096 atoms."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _lj_cluster, rgpot.LJClusterPot(cutoff=15.0))


class RayLJCluster(_Tracked):
    """Memray peak of one threaded Lennard-Jones cluster call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _lj_cluster, rgpot.LJClusterPot(cutoff=15.0))


class StillingerWeber(_Timed):
    """Stillinger-Weber silicon at 256, 1024 and 4096 atoms."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.StillingerWeberSi())


class RayStillingerWeber(_Tracked):
    """Memray peak of one threaded Stillinger-Weber call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.StillingerWeberSi())


class Tersoff(_Timed):
    """Tersoff silicon kernel at 256, 1024 and 4096 atoms."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.TersoffSiC())


class RayTersoff(_Tracked):
    """Memray peak of one threaded Tersoff call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        _prepare(self, atoms, threads, _diamond, rgpot.TersoffSiC())


class StoredEAM(_Timed):
    """Stored aluminium EAM at 256, 1024 and 4096 atoms."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        if rgpot.EAMAlPot is None:
            raise RuntimeError("this build has no stored EAM potential")
        _prepare(self, atoms, threads, _aluminium, rgpot.EAMAlPot())


class RayStoredEAM(_Tracked):
    """Memray peak of one threaded stored-EAM call."""

    def setup(self, atoms: int, threads: int) -> None:
        import rgpot

        if rgpot.EAMAlPot is None:
            raise RuntimeError("this build has no stored EAM potential")
        _prepare(self, atoms, threads, _aluminium, rgpot.EAMAlPot())
