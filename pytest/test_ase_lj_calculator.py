# Copyright (c) anneal developers
"""The metered Lennard-Jones calculator charges each geometry once."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
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


class GeometryTracking:
    def __init__(self, *args, **kwargs):
        self.geometry_hashes = []
        super().__init__(*args, **kwargs)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        source = atoms if atoms is not None else self.atoms
        positions = np.ascontiguousarray(source.positions, dtype=np.float64)
        self.geometry_hashes.append(hashlib.sha256(positions).digest())
        return super().calculate(atoms, properties, system_changes)


class TrackingLj(GeometryTracking, POTENTIAL.LjCalculator):
    pass


class StubServer:
    def __init__(self):
        self.round_trips = 0

    def eval(self, positions):
        self.round_trips += 1
        energy, gradient = POTENTIAL.lj_value_and_gradient(positions)
        return energy, 0.0, gradient


class TrackingAnneal(GeometryTracking, POTENTIAL.AnnealCalculator):
    pass


def calculator(kind, *, budget=100, deadline=None, hit=False):
    options = {
        "budget": budget,
        "reference": 1.0e100 if hit else -1.0e100,
        "tol": 0.0,
        "fmax": 1.0e100 if hit else 1.0e-8,
        "deadline": deadline,
    }
    if kind == "lj":
        return TrackingLj(**options)
    return TrackingAnneal(StubServer(), **options)


@pytest.mark.parametrize("kind", ["lj", "anneal"])
def test_energy_then_forces_charge_one_geometry(kind):
    cluster = atoms()
    cluster.calc = calculator(kind)
    cluster.get_potential_energy()
    cluster.get_forces()
    assert cluster.calc.calls == 1
    assert len(set(cluster.calc.geometry_hashes)) == 1
    if kind == "anneal":
        assert cluster.calc.server.round_trips == 1


@pytest.mark.parametrize("kind", ["lj", "anneal"])
@pytest.mark.parametrize("optimizer_name", ["QuasiNewton", "BFGS"])
def test_optimizers_charge_distinct_geometries_once(kind, optimizer_name):
    optimizer = getattr(pytest.importorskip("ase.optimize"), optimizer_name)
    cluster = atoms()
    cluster.calc = calculator(kind)
    optimizer(cluster, logfile=None).run(fmax=0.01, steps=4)

    distinct = set(cluster.calc.geometry_hashes)
    assert cluster.calc.calls == len(distinct)
    if kind == "anneal":
        assert cluster.calc.server.round_trips == len(distinct)


@pytest.mark.parametrize("kind", ["lj", "anneal"])
def test_position_change_recomputes(kind):
    cluster = atoms()
    cluster.calc = calculator(kind)
    cluster.get_potential_energy()
    cluster.get_forces()
    cluster.positions[0, 0] += 1.0e-12
    cluster.get_potential_energy()
    assert cluster.calc.calls == 2
    if kind == "anneal":
        assert cluster.calc.server.round_trips == 2


@pytest.mark.parametrize("kind", ["lj", "anneal"])
def test_budget_is_checked_on_next_distinct_geometry(kind):
    cluster = atoms()
    cluster.calc = calculator(kind, budget=1)
    cluster.get_potential_energy()
    cluster.get_forces()
    cluster.positions[0, 0] += 1.0e-12
    with pytest.raises(POTENTIAL.BudgetReached):
        cluster.get_potential_energy()
    assert cluster.calc.calls == 1
    assert len(set(cluster.calc.geometry_hashes)) == 2
    if kind == "anneal":
        assert cluster.calc.server.round_trips == 1


@pytest.mark.parametrize("kind", ["lj", "anneal"])
def test_deadline_is_checked_on_next_distinct_geometry(kind):
    cluster = atoms()
    cluster.calc = calculator(kind)
    cluster.get_potential_energy()
    cluster.calc.deadline = time.perf_counter() - 1.0
    cluster.get_forces()
    cluster.positions[0, 0] += 1.0e-12
    with pytest.raises(POTENTIAL.CapReached):
        cluster.get_potential_energy()
    assert cluster.calc.calls == 1
    assert len(set(cluster.calc.geometry_hashes)) == 2
    if kind == "anneal":
        assert cluster.calc.server.round_trips == 1


@pytest.mark.parametrize("kind", ["lj", "anneal"])
def test_hit_is_raised_on_first_distinct_geometry(kind):
    cluster = atoms()
    cluster.calc = calculator(kind, hit=True)
    with pytest.raises(POTENTIAL.HitReached):
        cluster.get_potential_energy()
    assert cluster.calc.calls == 1
    assert cluster.calc.hit_calls == 1
    assert len(set(cluster.calc.geometry_hashes)) == 1
    if kind == "anneal":
        assert cluster.calc.server.round_trips == 1


def test_cached_results_match_direct_potential():
    cluster = atoms()
    cluster.calc = calculator("lj")
    energy = cluster.get_potential_energy()
    forces = cluster.get_forces()

    expected_energy, gradient = POTENTIAL.lj_value_and_gradient(
        np.ascontiguousarray(cluster.positions.ravel(), dtype=np.float64)
    )
    assert energy == pytest.approx(expected_energy, abs=1.0e-12)
    assert forces == pytest.approx(-gradient.reshape(-1, 3), abs=1.0e-12)
