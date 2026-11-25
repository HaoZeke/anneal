# Copyright (c) anneal developers
"""The metered Lennard-Jones calculator charges each geometry once."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

ROOT = Path(__file__).resolve().parents[1]
POTENTIAL_PATH = ROOT / "scripts" / "ase_gofee_h2h" / "potential.py"
SPEC = importlib.util.spec_from_file_location("ase_lj_potential", POTENTIAL_PATH)
POTENTIAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POTENTIAL
SPEC.loader.exec_module(POTENTIAL)
POTENTIAL.lj_value_and_gradient = POTENTIAL.lj_value_and_gradient.py_func


def atoms():
    return ase.Atoms(
        "Ar3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.3, 0.1, 0.0],
            [0.2, 1.6, 0.3],
        ],
    )


def calculator():
    return POTENTIAL.LjCalculator(
        budget=100,
        reference=-1.0e100,
        tol=0.0,
        fmax=1.0e-8,
    )


def test_energy_then_forces_charge_one_geometry():
    cluster = atoms()
    cluster.calc = calculator()
    cluster.get_potential_energy()
    cluster.get_forces()
    assert cluster.calc.calls == 1


def test_quasi_newton_charges_distinct_geometries_once(monkeypatch):
    from ase.optimize import QuasiNewton

    evaluated = []
    original = POTENTIAL.lj_value_and_gradient

    def record(positions):
        evaluated.append(positions.copy())
        return original(positions)

    monkeypatch.setattr(POTENTIAL, "lj_value_and_gradient", record)
    cluster = atoms()
    cluster.calc = calculator()
    QuasiNewton(cluster, logfile=None).run(fmax=0.01, steps=4)

    distinct = {
        np.ascontiguousarray(positions).tobytes()
        for positions in evaluated
    }
    assert cluster.calc.calls == len(distinct)


def test_cached_results_match_direct_potential():
    cluster = atoms()
    cluster.calc = calculator()
    energy = cluster.get_potential_energy()
    forces = cluster.get_forces()

    expected_energy, gradient = POTENTIAL.lj_value_and_gradient(
        np.ascontiguousarray(cluster.positions.ravel(), dtype=np.float64)
    )
    assert energy == pytest.approx(expected_energy, abs=1.0e-12)
    assert forces == pytest.approx(-gradient.reshape(-1, 3), abs=1.0e-12)
