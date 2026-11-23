# MIT License
# Copyright 2023--present rgpot developers
"""ASE contract checks shared by every potential kind in this build."""

from __future__ import annotations

import copy
import gc
import math
import os
import pickle
import threading
from dataclasses import dataclass
from typing import Callable

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.socketio import SocketClient
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

import rgpot
from rgpot.ase import RgpotCalculator
from rgpot.ase.calculator import _potential_factories


@dataclass(frozen=True)
class PotentialCase:
    kind: str
    atoms: Callable[[], Atoms]
    parameters: dict[str, object] | None = None

    def calculator(self, *, numerical_stress: bool = False) -> RgpotCalculator:
        return RgpotCalculator(
            self.kind,
            numerical_stress=numerical_stress,
            **(self.parameters or {}),
        )


def pair_cluster() -> Atoms:
    return Atoms(
        "Ar3",
        positions=[[5.0, 5.0, 5.0], [8.7, 5.1, 5.0], [6.8, 8.4, 5.2]],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


def silicon_cluster() -> Atoms:
    return Atoms(
        "Si3",
        positions=[[5.0, 5.0, 5.0], [7.35, 5.0, 5.0], [6.18, 7.04, 5.15]],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


def aluminium_cluster() -> Atoms:
    return Atoms(
        "Al4",
        positions=[
            [5.0, 5.0, 5.0],
            [7.75, 5.0, 5.0],
            [6.38, 7.38, 5.0],
            [6.38, 5.8, 7.24],
        ],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


def iron_cluster() -> Atoms:
    return Atoms(
        "Fe2",
        positions=[[5.0, 5.0, 5.0], [7.48, 5.05, 5.0]],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


def copper_hydrogen_cluster() -> Atoms:
    return Atoms(
        numbers=[29, 29, 1, 1],
        positions=[
            [0.6394, 0.9048, 6.9752],
            [3.1965, 0.9042, 6.9755],
            [8.9836, 9.9470, 7.8356],
            [7.6408, 9.9470, 7.8356],
        ],
        cell=[15.3456, 21.702, 100.0],
        pbc=False,
    )


def copper_cluster() -> Atoms:
    return Atoms(
        "Cu3",
        positions=[[5.0, 5.0, 5.0], [7.55, 5.0, 5.0], [6.28, 7.2, 5.1]],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


def water_hydrogen_cluster() -> Atoms:
    return Atoms(
        numbers=[1, 1, 8, 1],
        positions=[
            [3.75, 3.5, 3.0],
            [2.25, 3.5, 3.0],
            [3.0, 3.0, 3.0],
            [5.0, 5.0, 4.6],
        ],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )


CASES = [
    PotentialCase("lj", pair_cluster),
    PotentialCase("ljcluster", pair_cluster),
    PotentialCase("morse", pair_cluster),
    PotentialCase("stillingerwebersi", silicon_cluster),
    PotentialCase("tersoffsic", silicon_cluster),
    PotentialCase("emt", copper_cluster),
    PotentialCase("gromacs", pair_cluster),
    PotentialCase("openmm", pair_cluster),
]

CU_FUNCFL = (
    Path(__file__).resolve().parents[2]
    / "CppCore"
    / "tests"
    / "data"
    / "eam_table"
    / "Cu_u3.eam"
)

CASES += [
    PotentialCase("d2", water_hydrogen_cluster),
    PotentialCase("d2cpmd", water_hydrogen_cluster),
    PotentialCase(
        "sum",
        pair_cluster,
        {
            "terms": [
                {"kind": "lj", "cutoff": 8.0},
                {"kind": "morse", "weight": 0.5, "cutoff": 8.0},
            ]
        },
    ),
]
if rgpot.has_eam_table:
    CASES.append(
        PotentialCase(
            "eam_table",
            copper_cluster,
            {"file": str(CU_FUNCFL), "format": "funcfl"},
        )
    )

if rgpot.has_expr:
    CASES.append(
        PotentialCase(
            "expr",
            pair_cluster,
            {
                "expression": "lj",
                "terms": {"lj": {"kind": "lj", "cutoff": 8.0}},
            },
        )
    )
if rgpot.has_fortran_pots:
    CASES += [
        PotentialCase("sw", silicon_cluster),
        PotentialCase("edip", silicon_cluster),
        PotentialCase("lenosky", silicon_cluster),
        PotentialCase("tersoff", silicon_cluster),
        PotentialCase("eam_al", aluminium_cluster),
        PotentialCase("fehe", iron_cluster),
        PotentialCase("cuh2", copper_hydrogen_cluster),
        PotentialCase("waterh", water_hydrogen_cluster),
    ]
if rgpot.has_dftd3:
    CASES.append(PotentialCase("d3", water_hydrogen_cluster))
if rgpot.has_dftd4:
    CASES.append(PotentialCase("d4", water_hydrogen_cluster))


def case_id(case: PotentialCase) -> str:
    return case.kind


ALL_CASES = pytest.mark.parametrize("case", CASES, ids=case_id)


def test_suite_covers_every_advertised_potential_type() -> None:
    factory_types = {
        {
            "rgpot_gromacs_potential": "GromacsPot",
            "rgpot_openmm_potential": "OpenMMPot",
        }.get(factory.__name__, factory.__name__)
        for factory in _potential_factories().values()
    }
    covered_types = {type(case.calculator().potential).__name__ for case in CASES}
    if rgpot.has_ipi:
        # IPIPot needs a live driver and has its own test below.
        covered_types.add("IPIPot")
    if hasattr(rgpot, "MullerBrownPot"):
        # A non-periodic surface: the cell-relaxation cases do not apply.
        mb = RgpotCalculator("mullerbrown")
        assert mb._cutoff == 0.0
        assert mb._kernel_periodic is False
        covered_types.add(type(mb.potential).__name__)
    assert factory_types == covered_types


def test_emt_energy_and_forces_match_ase_emt() -> None:
    atoms = copper_cluster()
    atoms.calc = RgpotCalculator("emt")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy == pytest.approx(4.602126, abs=1e-6)
    assert np.max(np.linalg.norm(forces, axis=1)) == pytest.approx(
        2.296119, abs=1e-5
    )
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)

    reference = copper_cluster()
    reference.calc = EMT(asap_cutoff=True)
    assert energy == pytest.approx(reference.get_potential_energy(), abs=1e-6)
    np.testing.assert_allclose(
        forces, reference.get_forces(), atol=1e-6, rtol=0.0
    )


@pytest.mark.parametrize("kind", ["gromacs", "openmm"])
def test_engine_factory_defaults_tile_like_lj(kind: str) -> None:
    atoms = pair_cluster()
    atoms.set_cell([10.0, 11.0, 12.0])
    atoms.pbc = True
    external = atoms.copy()
    external.calc = RgpotCalculator(kind, numerical_stress=True)
    direct = atoms.copy()
    direct.calc = RgpotCalculator("lj", u0=1.0, cutoff=8.0, psi=1.0)
    assert external.get_potential_energy() == direct.get_potential_energy()
    np.testing.assert_array_equal(external.get_forces(), direct.get_forces())
    np.testing.assert_allclose(
        external.get_stress(), direct.get_stress(), atol=2e-9, rtol=0.0
    )
    factory = getattr(rgpot, f"rgpot_{kind}_potential")
    direct_object = RgpotCalculator(factory())
    for operation in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="cannot pickle"):
            operation(direct_object)


def test_muller_brown_matches_its_surface_and_refuses_a_periodic_cell() -> None:
    atoms = Atoms(
        "H2",
        positions=[[-0.558, 1.442, 9.0], [4.0, 5.0, 6.0]],
        cell=[18.0, 19.0, 20.0],
        pbc=False,
    )
    atoms.calc = RgpotCalculator("mullerbrown")
    assert atoms.get_potential_energy() == pytest.approx(-146.70, abs=0.01)
    np.testing.assert_allclose(
        atoms.get_forces(),
        calculate_numerical_forces(atoms, eps=1e-5),
        atol=2e-6,
    )
    atoms.pbc = True
    with pytest.raises(ValueError, match="has no periodic boundaries"):
        atoms.get_potential_energy()


@ALL_CASES
def test_properties_forces_stress_and_free_energy(case: PotentialCase) -> None:
    atoms = case.atoms()
    analytic = case.calculator()
    assert {"energy", "free_energy", "forces"} <= set(
        analytic.implemented_properties
    )
    assert ("stress" in analytic.implemented_properties) is bool(
        analytic._caps.get("stress")
    )
    assert ("energies" in analytic.implemented_properties) is analytic._per_atom
    assert ("stresses" in analytic.implemented_properties) is (
        analytic._per_atom and analytic._analytic_stress
    )

    atoms.calc = case.calculator(numerical_stress=True)
    energy = atoms.get_potential_energy()
    assert atoms.get_potential_energy(force_consistent=True) == energy
    assert atoms.calc.results["free_energy"] == energy

    forces = atoms.get_forces()
    numerical_forces = calculate_numerical_forces(atoms, eps=1e-5)
    force_scale = max(1.0, float(np.max(np.abs(numerical_forces))))
    np.testing.assert_allclose(
        forces, numerical_forces, atol=2e-6 * force_scale, rtol=0.0
    )

    stress = atoms.get_stress()
    numerical_stress = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    stress_scale = max(1e-3, float(np.max(np.abs(numerical_stress))))
    np.testing.assert_allclose(
        stress, numerical_stress, atol=2e-6 * stress_scale, rtol=0.0
    )


@ALL_CASES
def test_state_reset_pickle_and_copy(case: PotentialCase) -> None:
    atoms = case.atoms()
    calc = case.calculator()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert calc.check_state(atoms) == []
    assert atoms.get_potential_energy() == energy

    changed = {
        "positions": atoms.copy(),
        "cell": atoms.copy(),
        "pbc": atoms.copy(),
        "numbers": atoms.copy(),
    }
    changed["positions"].positions[0, 0] += 0.01
    changed["cell"].cell[0, 0] += 0.01
    changed["pbc"].pbc[0] = not bool(changed["pbc"].pbc[0])
    changed["numbers"].numbers[0] += 1
    for name, altered in changed.items():
        assert name in calc.check_state(altered)
        calc.results["stale"] = True
        calc.calculate(atoms, properties=("energy",), system_changes=[name])
        assert "stale" not in calc.results
        assert math.isfinite(calc.results["energy"])

    calc.reset()
    assert calc.results == {}
    assert atoms.get_potential_energy() == pytest.approx(energy)
    for clone in (
        pickle.loads(pickle.dumps(calc)),
        copy.copy(calc),
        copy.deepcopy(calc),
    ):
        assert clone.potential is not calc.potential
        fresh = case.atoms()
        fresh.calc = clone
        assert fresh.get_potential_energy() == pytest.approx(energy)


@ALL_CASES
def test_constraints_and_short_cell_relaxation(case: PotentialCase) -> None:
    atoms = case.atoms()
    atoms.calc = case.calculator(numerical_stress=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    fixed = atoms.positions[0].copy()
    BFGS(atoms, logfile=None, maxstep=0.005).run(fmax=0.0, steps=2)
    np.testing.assert_array_equal(atoms.positions[0], fixed)
    assert math.isfinite(atoms.get_potential_energy())

    cell_atoms = case.atoms()
    cell_atoms.calc = case.calculator(numerical_stress=True)
    BFGS(FrechetCellFilter(cell_atoms), logfile=None, maxstep=0.002).run(
        fmax=0.0, steps=1
    )
    assert math.isfinite(cell_atoms.get_potential_energy())


@ALL_CASES
def test_partial_periodicity_works_or_raises_clearly(case: PotentialCase) -> None:
    atoms = case.atoms()
    atoms.pbc = [True, False, True]
    calc = case.calculator()
    atoms.calc = calc
    if calc._kernel_periodic:
        assert math.isfinite(atoms.get_potential_energy())
        assert np.all(np.isfinite(atoms.get_forces()))
    else:
        with pytest.raises(ValueError, match="has no periodic boundaries"):
            atoms.get_potential_energy()


@pytest.mark.skipif(
    not rgpot.has_ipi,
    reason="rgpot.has_ipi is False (built without -Dwith_ipi)",
)
def test_ipipot_forces_stress_cache_and_pickle_error(tmp_path) -> None:
    socket_name = f"rgpot_ase_{os.getpid()}_{tmp_path.name}"
    atoms = pair_cluster()
    atoms.pbc = True
    driver_atoms = atoms.copy()
    driver_atoms.calc = LennardJones(epsilon=0.2, sigma=3.0, rc=7.0)
    calc = RgpotCalculator(
        "ipi", address=f"unix:{socket_name}", timeout_s=5.0
    )
    errors: list[Exception] = []

    def drive() -> None:
        try:
            SocketClient(unixsocket=socket_name, timeout=5.0).run(
                driver_atoms, use_stress=True
            )
        except Exception as err:  # pragma: no cover - asserted below
            errors.append(err)

    driver = threading.Thread(target=drive)
    driver.start()
    atoms.calc = calc
    try:
        assert set(calc.implemented_properties) == {
            "energy",
            "free_energy",
            "forces",
            "stress",
        }
        energy = atoms.get_potential_energy()
        assert atoms.get_potential_energy(force_consistent=True) == energy
        np.testing.assert_allclose(
            atoms.get_forces(),
            calculate_numerical_forces(atoms, eps=1e-5),
            atol=2e-6,
        )
        np.testing.assert_allclose(
            atoms.get_stress(),
            calculate_numerical_stress(atoms, eps=1e-6, voigt=True),
            atol=2e-6,
        )
        atoms.get_potential_energy()
        assert calc.check_state(atoms) == []
        with pytest.raises(TypeError, match="live socket"):
            pickle.dumps(calc)
    finally:
        atoms.calc = None
        del calc
        gc.collect()
        driver.join(timeout=10.0)
    assert not driver.is_alive()
    assert errors == []
