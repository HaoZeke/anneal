"""ASE's own EAM calculator driven by the closed form of rgpot's EAM-Al.

``CppCore/rgpot/fortran/rgpot_eam_al.f90`` writes the kernel as

    E = sum_{i<j} phi(r_ij) + sum_i F(rho_i),   rho_i = sum_{j /= i} rho(r_ij)

with double-exponential pair and density terms shifted to vanish at
``rcut`` and dropped from ``sqrt(0.9999) * rcut`` outward, the pair term
carrying the ``-2 g rho(r)`` gauge term, and
``F(rho) = g rho + sum_k c_k rho^k``. The parameters are the
single-precision roundings the kernel stores. ASE's ``EAM`` accepts
callables for F, rho and phi and their derivatives, so the reference has
no spline: it differs from the kernel only in summation order and in the
neighbour search.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.eam import EAM


def _f32(value: float) -> float:
    return float(np.float32(value))


G = _f32(-0.60647886749203)
PAIR_A, PAIR_ALPHA = _f32(2294.3609145535), _f32(3.0205380362464)
PAIR_B, PAIR_BETA = _f32(-192.06894637533), _f32(1.5102690181232)
DENSITY_SCALE = _f32(0.87928364657088)
DENSITY_A, DENSITY_B = _f32(3.3068828537934), _f32(6.6137657075868)
DENSITY_WEIGHT = _f32(512.0)
DENSITY_POWER = 6
EMBED = [
    _f32(c)
    for c in (
        4.4572157051836,
        193.21775368064,
        -1173.6502684704,
        4203.3500116196,
        -8785.1680280827,
        10632.994102532,
        -6921.0410455328,
        1875.2698365752,
    )
]
RCUT = _f32(7.1)
TRUNCATION = float(np.sqrt(0.9999) * RCUT)


def _pair_shape(r: np.ndarray) -> np.ndarray:
    return PAIR_A * np.exp(-PAIR_ALPHA * r) + PAIR_B * np.exp(-PAIR_BETA * r)


def _d_pair_shape(r: np.ndarray) -> np.ndarray:
    return -PAIR_ALPHA * PAIR_A * np.exp(-PAIR_ALPHA * r) - PAIR_BETA * PAIR_B * np.exp(
        -PAIR_BETA * r
    )


def _density_shape(r: np.ndarray) -> np.ndarray:
    return r**DENSITY_POWER * (
        np.exp(-DENSITY_A * r) + DENSITY_WEIGHT * np.exp(-DENSITY_B * r)
    )


def _d_density_shape(r: np.ndarray) -> np.ndarray:
    ea = np.exp(-DENSITY_A * r)
    eb = DENSITY_WEIGHT * np.exp(-DENSITY_B * r)
    return DENSITY_POWER * r ** (DENSITY_POWER - 1) * (ea + eb) - r**DENSITY_POWER * (
        DENSITY_A * ea + DENSITY_B * eb
    )


def density(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    value = DENSITY_SCALE * (_density_shape(r) - _density_shape(np.float64(RCUT)))
    return np.where(r < TRUNCATION, value, 0.0)


def d_density(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    return np.where(r < TRUNCATION, DENSITY_SCALE * _d_density_shape(r), 0.0)


def pair(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    value = _pair_shape(r) - _pair_shape(np.float64(RCUT)) - 2.0 * G * density(r)
    return np.where(r < TRUNCATION, value, 0.0)


def d_pair(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    value = _d_pair_shape(r) - 2.0 * G * d_density(r)
    return np.where(r < TRUNCATION, value, 0.0)


def embedding(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    return G * rho + sum(c * rho ** (k + 1) for k, c in enumerate(EMBED))


def d_embedding(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    return G + sum((k + 1) * c * rho**k for k, c in enumerate(EMBED))


def ase_eam_al() -> EAM:
    """ASE ``EAM`` with the analytic EAM-Al functions; aluminium only."""
    return EAM(
        elements=["Al"],
        embedded_energy=np.array([embedding]),
        electron_density=np.array([density]),
        phi=np.array([[pair]]),
        d_embedded_energy=np.array([d_embedding]),
        d_electron_density=np.array([d_density]),
        d_phi=np.array([[d_pair]]),
        cutoff=TRUNCATION,
        form="alloy",
        skin=0.0,
    )
