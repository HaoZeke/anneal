"""Energy and force calls for the classical potentials.

Each ``time_`` sample is one call that returns the energy and the forces.
Each ``ray_`` sample is the memray peak of that same call. The geometry
is built in ``setup``, which asv does not time and does not track.
"""

from __future__ import annotations

import math

import numpy as np

REPEAT = 6
SIZES_CLUSTER = (32, 64, 128)
SIZES_DIAMOND = (64, 216)
SIZES_FCC = (32, 108)

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


def _crystal(n_side: int, lattice: float, basis, atomic_number: int):
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                origin = (i, j, k)
                for site in basis:
                    coords.append(
                        (
                            (origin[0] + site[0]) * lattice,
                            (origin[1] + site[1]) * lattice,
                            (origin[2] + site[2]) * lattice,
                        )
                    )
    positions = np.ascontiguousarray(coords, dtype=np.float64)
    atom_types = np.full(positions.shape[0], atomic_number, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * (n_side * lattice)
    return positions, atom_types, box


def _side(atoms: int, per_cell: int) -> int:
    side = 1
    while per_cell * side * side * side < atoms:
        side += 1
    if per_cell * side * side * side != atoms:
        raise AssertionError(atoms)
    return side


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
    box = np.eye(3, dtype=np.float64) * 40.0
    return positions, atom_types, box


def _diamond(atoms: int):
    return _crystal(_side(atoms, 8), 5.431, _DIAMOND, 14)


def _aluminium(atoms: int):
    return _crystal(_side(atoms, 4), 4.05, _FCC, 13)


def _energy_force(pot, positions, atom_types, box) -> None:
    energy, forces, _variance = pot(positions, atom_types, box)
    if getattr(forces, "shape", None) != positions.shape:
        raise AssertionError(getattr(forces, "shape", None))
    if not math.isfinite(float(energy)):
        raise AssertionError("energy is not finite")


class _Timed:
    repeat = REPEAT
    sample_time = 0.01
    min_run_count = 1
    timeout = 300

    def time_energy_force(self, atoms: int) -> None:
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class _Tracked:
    number = 1
    repeat = REPEAT
    timeout = 300

    def ray_energy_force(self, atoms: int) -> None:
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class LJCluster(_Timed):
    """Lennard-Jones cluster, free boundary. One energy and force call."""

    params = list(SIZES_CLUSTER)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _lj_cluster(atoms)
        self.pot = rgpot.LJClusterPot(cutoff=15.0)
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class RayLJCluster(_Tracked):
    """Memray peak of one Lennard-Jones cluster energy and force call."""

    params = list(SIZES_CLUSTER)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _lj_cluster(atoms)
        self.pot = rgpot.LJClusterPot(cutoff=15.0)


class StillingerWeber(_Timed):
    """Stillinger-Weber silicon on diamond cells of 64 and 216 atoms."""

    params = list(SIZES_DIAMOND)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _diamond(atoms)
        self.pot = rgpot.StillingerWeberSi()
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class RayStillingerWeber(_Tracked):
    """Memray peak of one Stillinger-Weber energy and force call."""

    params = list(SIZES_DIAMOND)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _diamond(atoms)
        self.pot = rgpot.StillingerWeberSi()


class Tersoff(_Timed):
    """Tersoff silicon-carbon kernel, all silicon, on the same cells."""

    params = list(SIZES_DIAMOND)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _diamond(atoms)
        self.pot = rgpot.TersoffSiC()
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class RayTersoff(_Tracked):
    """Memray peak of one Tersoff energy and force call."""

    params = list(SIZES_DIAMOND)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        self.positions, self.atom_types, self.box = _diamond(atoms)
        self.pot = rgpot.TersoffSiC()


class StoredEAM(_Timed):
    """Aluminium EAM with the parameters stored in the library."""

    params = list(SIZES_FCC)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        if rgpot.EAMAlPot is None:
            raise RuntimeError("this build has no stored EAM potential")
        self.positions, self.atom_types, self.box = _aluminium(atoms)
        self.pot = rgpot.EAMAlPot()
        _energy_force(self.pot, self.positions, self.atom_types, self.box)


class RayStoredEAM(_Tracked):
    """Memray peak of one stored-EAM energy and force call."""

    params = list(SIZES_FCC)
    param_names = ["atoms"]

    def setup(self, atoms: int) -> None:
        import rgpot

        if rgpot.EAMAlPot is None:
            raise RuntimeError("this build has no stored EAM potential")
        self.positions, self.atom_types, self.box = _aluminium(atoms)
        self.pot = rgpot.EAMAlPot()
