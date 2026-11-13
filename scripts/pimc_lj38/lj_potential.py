# Copyright (c) anneal developers
"""Lennard-Jones potential used by the i-PI driver in the RPQA LJ example.

The pair formula, the sharp cutoff, and the volume tail are the ones in
``drivers/f90/LJ.f90`` at afiorent/i-pi ``4e86f5fd``. Positions and ``sigma``
are in bohr, energies in hartree. The tail is the Martyna-Hughes correction
that ``LJ_getall`` adds after the pair sum.
"""

from __future__ import annotations

import math

import numpy as np

# run.sh of examples/features/quantum_annealing/RPQA/LJ_cluster
SIGMA = 5.270446
EPSILON = 0.00367493
CUTOFF = 7.0 * SIGMA
# 8 * pi / 3, the literal in LJ.f90
FOUR_TAU_BY_3 = 8.3775804095727811
# i-PI UnitMap length factor: angstrom times this is bohr
ANGSTROM_TO_BOHR = 1.8897261
SIGMA_ANGSTROM = SIGMA / ANGSTROM_TO_BOHR


def tail(n_atoms: int, volume: float, sigma: float = SIGMA, eps: float = EPSILON, rc: float = CUTOFF) -> tuple[float, float]:
    """Return ``(pot_lr, vir_lr)`` from ``LJ_longrange``."""
    sbyr = sigma / rc
    s3 = sbyr * sbyr * sbyr
    s6 = s3 * s3
    prefactor = FOUR_TAU_BY_3 * n_atoms * n_atoms * eps / volume
    prefactor *= s3 * sigma * sigma * sigma
    pot_lr = prefactor * (s6 / 3.0 - 1.0)
    vir_lr = prefactor * (s6 - 1.0) + pot_lr
    return pot_lr, vir_lr


def evaluate(cell: np.ndarray, pos: np.ndarray, sigma: float = SIGMA, eps: float = EPSILON, rc: float = CUTOFF) -> tuple[float, np.ndarray, np.ndarray]:
    """Energy, forces, and virial for one geometry.

    ``cell`` is ``(3, 3)`` as the Python socket client receives it. ``pos`` is
    ``(natoms, 3)``. The pair term matches ``LJ_functions`` / ``LJ_fij`` with
    ``rij = ri - rj`` under the minimum image.
    """
    cell = np.asarray(cell, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    n_atoms = pos.shape[0]
    inv = np.linalg.inv(cell)
    delta = pos[:, None, :] - pos[None, :, :]
    sij = np.einsum("ab,ijb->ija", inv, delta)
    sij -= np.rint(sij)
    rij = np.einsum("ab,ijb->ija", cell, sij)
    iu, ju = np.triu_indices(n_atoms, k=1)
    vec = rij[iu, ju]
    r2 = np.einsum("ij,ij->i", vec, vec)
    keep = r2 < rc * rc
    vec = vec[keep]
    r2 = r2[keep]
    iu = iu[keep]
    ju = ju[keep]
    sr2 = (sigma * sigma) / r2
    sr6 = sr2 * sr2 * sr2
    pot_pairs = 4.0 * eps * (sr6 * (sr6 - 1.0))
    # force magnitude is -dV/dr; fij = that times rij/r, then /r for the broadcast
    f_over_r = 24.0 * eps * (sr6 * (2.0 * sr6 - 1.0)) / r2
    fij = vec * f_over_r[:, None]
    forces = np.zeros((n_atoms, 3), dtype=np.float64)
    np.add.at(forces, iu, fij)
    np.add.at(forces, ju, -fij)
    virial = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(a, 3):
            virial[a, b] = np.dot(fij[:, a], vec[:, b])
    volume = abs(float(np.linalg.det(cell)))
    pot_lr, vir_lr = tail(n_atoms, volume, sigma, eps, rc)
    for a in range(3):
        virial[a, a] += vir_lr
    return float(pot_pairs.sum() + pot_lr), forces, virial


def reduced_pair_energy(pos_sigma: np.ndarray) -> float:
    """Full pair sum in units of epsilon. No cutoff and no tail."""
    pos = np.asarray(pos_sigma, dtype=np.float64)
    delta = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    iu, ju = np.triu_indices(pos.shape[0], k=1)
    r2 = r2[iu, ju]
    sr2 = 1.0 / r2
    sr6 = sr2 * sr2 * sr2
    return float(np.sum(4.0 * (sr6 * (sr6 - 1.0))))


def dimer_well() -> float:
    """Pair energy at ``r = 2**(1/6) sigma`` is ``-epsilon`` before the tail."""
    r = SIGMA * (2.0 ** (1.0 / 6.0))
    cell = np.diag([300.0 * ANGSTROM_TO_BOHR] * 3)
    pos = np.zeros((2, 3))
    pos[1, 0] = r
    pot, forces, _virial = evaluate(cell, pos)
    sr6 = 0.5  # (sigma/r)^6 = 1/2 at the minimum
    pair = 4.0 * EPSILON * (sr6 * (sr6 - 1.0))
    volume = abs(float(np.linalg.det(cell)))
    pot_lr, _vir_lr = tail(2, volume)
    if abs(pair + EPSILON) > 1e-15:
        raise RuntimeError("dimer pair is not -epsilon")
    if abs(pot - (pair + pot_lr)) > 1e-12:
        raise RuntimeError("dimer evaluator missed the tail")
    # Repulsive inside the well: atom 1 is pushed to +x.
    if forces[1, 0] >= 0.0:
        raise RuntimeError("dimer force sign")
    return pot


def _reference_loop(cell: np.ndarray, pos: np.ndarray) -> float:
    """Scalar pair sum plus tail, for the vectorised evaluator."""
    inv = np.linalg.inv(cell)
    n_atoms = pos.shape[0]
    pot = 0.0
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            sij = inv @ (pos[i] - pos[j])
            sij -= np.rint(sij)
            rij = cell @ sij
            r2 = float(rij @ rij)
            if r2 >= CUTOFF * CUTOFF:
                continue
            sr2 = (SIGMA * SIGMA) / r2
            sr6 = sr2 * sr2 * sr2
            pot += 4.0 * EPSILON * (sr6 * (sr6 - 1.0))
    volume = abs(float(np.linalg.det(cell)))
    pot_lr, _vir_lr = tail(n_atoms, volume)
    return pot + pot_lr


def self_test() -> None:
    dimer_well()
    rng = np.random.default_rng(20261113)
    pos = rng.normal(size=(6, 3)) * SIGMA
    cell = np.diag([300.0 * ANGSTROM_TO_BOHR] * 3)
    pot, forces, _virial = evaluate(cell, pos)
    if abs(pot - _reference_loop(cell, pos)) > 1e-10:
        raise RuntimeError("vectorised LJ drifted from the pair loop")
    if forces.shape != pos.shape:
        raise RuntimeError("force shape")
    if not math.isfinite(pot):
        raise RuntimeError("non-finite LJ")
    reduced = reduced_pair_energy(pos / SIGMA)
    # Every pair sits inside the cutoff, so the pair sum is the reduced energy.
    volume = abs(float(np.linalg.det(cell)))
    pot_lr, _vir_lr = tail(pos.shape[0], volume)
    if abs((pot - pot_lr) / EPSILON - reduced) > 1e-8:
        raise RuntimeError("reduced LJ does not match the driver pair sum")
