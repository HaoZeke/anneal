"""Public LAMMPS, GROMACS and OpenMM constructors.

The dimer energy is -0.3203213355 eV. GROMACS and OpenMM store
nanometres and kJ/mol once. The LAMMPS constructor needs the engine
plugin; without it the test says why it skipped.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import rgpot


DIMER = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
BOX = np.diag([20.0, 20.0, 20.0])
KJ_PER_EV = 96.48533212331002


def test_gromacs_and_openmm_store_units_once():
    for factory in (rgpot.rgpot_gromacs_potential, rgpot.rgpot_openmm_potential):
        pot = factory(1.0, 8.0, 1.0)
        energy, forces, variance = pot(DIMER, [1, 1], BOX)
        assert f"{energy:.10f}" == "-0.3203213355"
        assert variance == 0.0
        assert forces.shape == (2, 3)
        assert pot.native_u0() == pytest.approx(KJ_PER_EV, rel=1e-12)
        assert pot.native_cutoff() == pytest.approx(0.8)
        assert pot.native_psi() == pytest.approx(0.1)


def test_lammps_missing_plugin_names_the_library():
    with pytest.raises(RuntimeError, match="liblj_engine_missing"):
        rgpot.rgpot_lammps_potential(
            1.0, 8.0, 1.0, engine_path="./liblj_engine_missing.so"
        )


def test_lammps_plugin_dimer():
    path = os.environ.get("RGPOT_LJ_ENGINE", "")
    if not path:
        pytest.skip(
            "liblj_engine.so was not provided (RGPOT_LJ_ENGINE); "
            "the LAMMPS constructor keeps that plugin as its backend"
        )
    pot = rgpot.rgpot_lammps_potential(1.0, 8.0, 1.0, engine_path=path)
    energy, forces, _variance = pot(DIMER, [1, 1], BOX)
    assert f"{energy:.10f}" == "-0.3203213355"
    assert forces.shape == (2, 3)


def test_openmm_engine_dimer():
    path = os.environ.get("RGPOT_OPENMM_ENGINE", "")
    if not path:
        pytest.skip(
            "OpenMM was not found at configure time, so "
            "libopenmm_engine.so was not built"
        )
    pot = rgpot.rgpot_openmm_potential(1.0, 8.0, 1.0, engine_path=path)
    energy, forces, _variance = pot(DIMER, [1, 1], BOX)
    ref_e, ref_f, _ = rgpot.rgpot_gromacs_potential(1.0, 8.0, 1.0)(
        DIMER, [1, 1], BOX
    )
    # OpenMM's expression matches the in-process value to 1e-6 eV and
    # the force to 1e-5 eV/Angstrom, the same window as openmm_lj.
    assert f"{ref_e:.10f}" == "-0.3203213355"
    assert energy == pytest.approx(ref_e, abs=1e-6)
    assert forces.shape == (2, 3)
    assert forces[1, 0] == pytest.approx(ref_f[1, 0], abs=1e-5)
