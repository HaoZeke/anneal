# Copyright 2023--present rgpot developers

"""Comprehensive ASE Calculator / BaseCalculator contract tests."""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.calculators.test import gradient_test
from ase.filters import FrechetCellFilter
from ase.io import jsonio
from ase.optimize import BFGS

import rgpot
from rgpot.ase import RgpotCalculator

from support import lj_dimer

# Named tolerances matching standard double-precision central differences and
# ASE relaxation thresholds.
FORCE_FD_STEP_ANGSTROM = 1e-5
FORCE_FD_ATOL_EV_PER_ANG = 1e-8
GRADIENT_TEST_FINEST_ATOL = 1e-6
STRESS_FD_STEP = 1e-6
STRESS_FD_ATOL_EV_PER_ANG3 = 1e-9
BFGS_FMAX_EV_PER_ANG = 1e-4
CELL_RELAX_FMAX = 1e-5
STRESS_RELAX_ATOL_EV_PER_ANG3 = 1e-6
EXACT_MATCH_ATOL = 1e-12
SUM_RULE_ATOL = 1e-12

LJ_ARGON_PARAMS = {"u0": 0.0104, "psi": 3.40, "cutoff": 8.5}

needs_fortran = pytest.mark.skipif(
    not rgpot.has_fortran_pots, reason="this build has no Fortran potentials"
)


class _CountingPotential:
    """Wrapper that counts underlying kernel evaluations."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls = 0

    def caps(self):
        return self._inner.caps()

    def config(self):
        return self._inner.config()

    def evaluate(self, positions, numbers, box):
        self.calls += 1
        return self._inner.evaluate(positions, numbers, box)


def _argon_bulk(repeat: int = 2, seed: int = 11) -> Atoms:
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(repeat)
    atoms.rattle(0.05, seed=seed)
    return atoms


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _missing_source_files(repo_root: Path, paths: tuple[Path, ...]) -> list[str]:
    """Repo-relative names of paths that are not files in this tree."""
    missing: list[str] = []
    for path in paths:
        if path.is_file():
            continue
        try:
            missing.append(path.relative_to(repo_root).as_posix())
        except ValueError:
            missing.append(str(path))
    return missing


def test_calculator_and_basecalculator_inheritance() -> None:
    from rgpot.ase.rpc import RgpotRpcCalculator

    calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    assert isinstance(calc, Calculator)
    assert isinstance(calc, BaseCalculator)
    assert issubclass(RgpotRpcCalculator, Calculator)
    assert issubclass(RgpotRpcCalculator, BaseCalculator)


def test_implemented_properties_by_capabilities() -> None:
    lj = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    assert lj.implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "energies",
        "stresses",
    ]

    lj_cluster = RgpotCalculator("ljcluster", **LJ_ARGON_PARAMS)
    assert lj_cluster.implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "energies",
    ]

    morse = RgpotCalculator("morse")
    assert morse.implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "energies",
        "stresses",
    ]

    if rgpot.has_fortran_pots:
        eam = RgpotCalculator("eam_al")
        assert eam.implemented_properties == ["energy", "free_energy", "forces"]
        eam_num = RgpotCalculator("eam_al", numerical_stress=True)
        assert eam_num.implemented_properties == [
            "energy",
            "free_energy",
            "forces",
            "stress",
        ]
        sw = RgpotCalculator("sw")
        assert sw.implemented_properties == [
            "energy",
            "free_energy",
            "forces",
            "stress",
        ]
        tersoff_sic = RgpotCalculator("tersoffsic")
        assert tersoff_sic.implemented_properties == [
            "energy",
            "free_energy",
            "forces",
            "stress",
        ]


def test_calculate_and_calculation_required_caching() -> None:
    pot = _CountingPotential(rgpot.LJPot(**LJ_ARGON_PARAMS))
    calc = RgpotCalculator(pot)
    atoms = _argon_bulk()

    assert calc.calculation_required(atoms, ["energy"]) is True
    assert calc.calculation_required(atoms, ["dipole"]) is True

    calc.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
    assert pot.calls == 1
    assert calc.check_state(atoms) == []
    assert calc.calculation_required(atoms, ["energy", "free_energy", "forces", "stress"]) is False
    assert calc.calculation_required(atoms, ["dipole"]) is True

    # Calling through Atoms getters hits the cache.
    atoms.calc = calc
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
    _ = atoms.get_stress()
    assert pot.calls == 1

    # Each system_change key must invalidate cached results.
    moved = atoms.copy()
    moved.positions[0, 0] += 0.02
    assert "positions" in calc.check_state(moved)
    assert calc.calculation_required(moved, ["energy"]) is True

    strained = atoms.copy()
    strained.set_cell(strained.cell.array * 1.02, scale_atoms=True)
    assert "cell" in calc.check_state(strained)
    assert calc.calculation_required(strained, ["energy"]) is True

    nonperiodic = atoms.copy()
    nonperiodic.pbc = False
    assert "pbc" in calc.check_state(nonperiodic)
    assert calc.calculation_required(nonperiodic, ["energy"]) is True

    relabelled = atoms.copy()
    relabelled.numbers[0] = 36
    assert "numbers" in calc.check_state(relabelled)
    assert calc.calculation_required(relabelled, ["energy"]) is True


def test_shapes_units_and_ase_sign_convention() -> None:
    compressed = bulk("Ar", "fcc", a=4.7, cubic=True).repeat(2)
    compressed.rattle(0.03, seed=5)
    compressed.calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)

    energy = compressed.get_potential_energy()
    free_energy = compressed.get_potential_energy(force_consistent=True)
    forces = compressed.get_forces()
    stress_voigt = compressed.get_stress()
    stress_tensor = compressed.get_stress(voigt=False)
    energies = compressed.get_potential_energies()
    stresses = compressed.get_stresses()

    assert isinstance(energy, float)
    assert free_energy == pytest.approx(energy, abs=EXACT_MATCH_ATOL)
    assert compressed.calc.results["free_energy"] == pytest.approx(energy, abs=EXACT_MATCH_ATOL)

    assert forces.shape == (len(compressed), 3)
    assert forces.dtype == np.float64

    assert stress_voigt.shape == (6,)
    assert stress_voigt.dtype == np.float64
    # ASE convention: sigma = (1/V) dE/dstrain -> negative under compression.
    assert np.all(stress_voigt[:3] < 0.0)

    assert stress_tensor.shape == (3, 3)
    assert stress_tensor.dtype == np.float64
    np.testing.assert_allclose(stress_tensor, stress_tensor.T, atol=EXACT_MATCH_ATOL)
    np.testing.assert_allclose(
        stress_voigt,
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ],
        atol=EXACT_MATCH_ATOL,
    )

    assert energies.shape == (len(compressed),)
    assert energies.dtype == np.float64
    assert energies.sum() == pytest.approx(energy, abs=SUM_RULE_ATOL)

    assert stresses.shape == (len(compressed), 6)
    assert stresses.dtype == np.float64
    np.testing.assert_allclose(stresses.sum(axis=0), stress_voigt, atol=SUM_RULE_ATOL)


def test_basecalculator_calculate_properties_and_export_properties() -> None:
    atoms = _argon_bulk()
    calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)

    props = calc.calculate_properties(atoms, ["energy", "free_energy", "forces", "stress"])
    assert props["natoms"] == len(atoms)
    assert props["free_energy"] == pytest.approx(props["energy"], abs=EXACT_MATCH_ATOL)
    assert props["forces"].shape == (len(atoms), 3)
    assert props["stress"].shape == (6,)

    exported = calc.export_properties()
    assert exported["natoms"] == len(atoms)
    assert exported["energy"] == pytest.approx(props["energy"], abs=EXACT_MATCH_ATOL)
    np.testing.assert_allclose(exported["forces"], props["forces"], atol=EXACT_MATCH_ATOL)
    np.testing.assert_allclose(exported["stress"], props["stress"], atol=EXACT_MATCH_ATOL)


def test_reset_clears_atoms_and_results() -> None:
    atoms = _argon_bulk()
    calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)

    assert calc.atoms is None
    with pytest.raises(ValueError):
        calc.get_atoms()

    atoms.calc = calc
    _ = atoms.get_potential_energy()
    assert calc.atoms is not None
    assert "energy" in calc.results

    calc.reset()
    assert calc.atoms is None
    assert calc.results == {}
    assert calc.calculation_required(atoms, ["energy"]) is True
    with pytest.raises(ValueError):
        calc.get_atoms()


def test_atoms_and_results_copy_semantics() -> None:
    atoms = _argon_bulk()
    calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    atoms.calc = calc

    initial_positions = atoms.positions.copy()
    forces_ref = atoms.get_forces().copy()
    stress_ref = atoms.get_stress().copy()
    energies_ref = atoms.get_potential_energies().copy()
    stresses_ref = atoms.get_stresses().copy()

    # Mutating caller atoms after evaluation must not corrupt calc.atoms.
    assert calc.atoms is not atoms
    atoms.positions[0, 0] += 0.5
    np.testing.assert_allclose(calc.atoms.positions, initial_positions, atol=EXACT_MATCH_ATOL)

    # Atoms.copy() must drop the calculator attachment.
    copied_atoms = atoms.copy()
    assert copied_atoms.calc is None

    # calc.get_atoms() returns a fresh copy with calc attached.
    from_calc = calc.get_atoms()
    assert from_calc is not calc.atoms
    assert from_calc.calc is calc
    np.testing.assert_allclose(from_calc.positions, initial_positions, atol=EXACT_MATCH_ATOL)

    # Mutating arrays returned by ASE getters must not mutate cached results.
    atoms.positions = initial_positions
    f_view = atoms.get_forces()
    f_view[:] = 0.0
    np.testing.assert_allclose(atoms.get_forces(), forces_ref, atol=EXACT_MATCH_ATOL)

    s_view = atoms.get_stress()
    s_view[:] = 0.0
    np.testing.assert_allclose(atoms.get_stress(), stress_ref, atol=EXACT_MATCH_ATOL)

    e_view = atoms.get_potential_energies()
    e_view[:] = 0.0
    np.testing.assert_allclose(atoms.get_potential_energies(), energies_ref, atol=EXACT_MATCH_ATOL)

    st_view = atoms.get_stresses()
    st_view[:] = 0.0
    np.testing.assert_allclose(atoms.get_stresses(), stresses_ref, atol=EXACT_MATCH_ATOL)


def test_todict_json_roundtrip_and_custom_potential() -> None:
    atoms = _argon_bulk()
    calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    atoms.calc = calc
    ref_energy = atoms.get_potential_energy()
    ref_forces = atoms.get_forces()
    ref_stress = atoms.get_stress()

    params = calc.todict()
    assert params["potential"] == "LJPot"
    encoded = jsonio.encode(params)
    decoded = jsonio.decode(encoded)

    rebuilt = RgpotCalculator(**decoded)
    clone = atoms.copy()
    clone.calc = rebuilt
    assert clone.get_potential_energy() == pytest.approx(ref_energy, abs=EXACT_MATCH_ATOL)
    np.testing.assert_allclose(clone.get_forces(), ref_forces, atol=EXACT_MATCH_ATOL)
    np.testing.assert_allclose(clone.get_stress(), ref_stress, atol=EXACT_MATCH_ATOL)

    class _CallablePot:
        # A caller-supplied potential states its cutoff. 0 leaves the cell untiled.
        cutoff = 0.0

        def __call__(self, positions, numbers, box):
            del numbers, box
            return 0.0, np.zeros_like(positions), 0.0

    custom = RgpotCalculator(_CallablePot())
    assert custom.todict() == {}


def test_property_not_implemented_error() -> None:
    atoms = _argon_bulk()
    lj = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    atoms.calc = lj

    for prop in (
        "dipole",
        "charges",
        "magmom",
        "magmoms",
        "dielectric_tensor",
        "born_effective_charges",
        "polarization",
    ):
        with pytest.raises(PropertyNotImplementedError):
            lj.get_property(prop, atoms)

    with pytest.raises(PropertyNotImplementedError):
        atoms.get_dipole_moment()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_charges()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_magnetic_moment()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_magnetic_moments()

    cluster = lj_dimer()
    lj_cluster = RgpotCalculator("ljcluster", **LJ_ARGON_PARAMS)
    cluster.calc = lj_cluster
    for prop in ("stress", "stresses", "dipole", "charges"):
        with pytest.raises(PropertyNotImplementedError):
            lj_cluster.get_property(prop, cluster)

    if rgpot.has_fortran_pots:
        si = bulk("Si", "diamond", a=5.431, cubic=True)
        si.calc = RgpotCalculator("sw")
        for prop in ("energies", "stresses", "dipole"):
            with pytest.raises(PropertyNotImplementedError):
                si.calc.get_property(prop, si)

        al = bulk("Al", "fcc", a=4.05, cubic=True)
        al.calc = RgpotCalculator("eam_al")
        for prop in ("stress", "energies", "stresses"):
            with pytest.raises(PropertyNotImplementedError):
                al.calc.get_property(prop, al)


def test_ase_test_tooling_gradient_test_and_fd() -> None:
    atoms = _argon_bulk(repeat=1)
    atoms.calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)

    analytic_f, finest_fd_f = gradient_test(atoms)
    np.testing.assert_allclose(analytic_f, finest_fd_f, atol=GRADIENT_TEST_FINEST_ATOL, rtol=0.0)

    fd_forces = calculate_numerical_forces(atoms, eps=FORCE_FD_STEP_ANGSTROM)
    np.testing.assert_allclose(
        atoms.get_forces(), fd_forces, atol=FORCE_FD_ATOL_EV_PER_ANG, rtol=0.0
    )

    fd_stress = calculate_numerical_stress(atoms, eps=STRESS_FD_STEP, voigt=True)
    np.testing.assert_allclose(
        atoms.get_stress(), fd_stress, atol=STRESS_FD_ATOL_EV_PER_ANG3, rtol=0.0
    )


def test_ase_bfgs_and_frechet_cell_filter_step() -> None:
    cluster = lj_dimer(distance=4.2)
    cluster.calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    initial_e = cluster.get_potential_energy()
    assert BFGS(cluster, logfile=None).run(fmax=BFGS_FMAX_EV_PER_ANG, steps=200)
    assert cluster.get_potential_energy() < initial_e
    assert float(np.max(np.linalg.norm(cluster.get_forces(), axis=1))) < BFGS_FMAX_EV_PER_ANG

    strained = bulk("Ar", "fcc", a=5.0, cubic=True)
    strained.calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
    fcf = FrechetCellFilter(strained)
    opt = BFGS(fcf, logfile=None)
    e0 = strained.get_potential_energy()
    opt.step()
    assert strained.get_potential_energy() <= e0
    assert opt.run(fmax=CELL_RELAX_FMAX, steps=200)
    np.testing.assert_allclose(strained.get_stress(), 0.0, atol=STRESS_RELAX_ATOL_EV_PER_ANG3)


def test_rpc_shim_and_ase_helpers_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    helpers_path = repo_root / "CppCore" / "rgpot" / "rpc" / "ase_helpers.py"
    shim_path = repo_root / "CppCore" / "rgpot" / "rpc" / "ase_calculator.py"
    # Checkout scripts beside the RPC sources. meson does not install them
    # into the rgpot package, and the wheel test tree does not contain CppCore.
    missing = _missing_source_files(repo_root, (helpers_path, shim_path))
    if missing:
        pytest.skip("source file is not in this tree: " + ", ".join(missing))

    helpers = _load_module_from_path("_test_ase_helpers_contract", helpers_path)
    try:
        atoms = lj_dimer()
        calc = RgpotCalculator("lj", **LJ_ARGON_PARAMS)
        returned = helpers.atoms_to_calc(atoms, calc)
        assert returned is atoms
        assert atoms.calc is calc
        assert helpers.AnnealRunner.__name__ == "AnnealRunner"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            opt_cls = helpers.AnnealOptimizer
            assert opt_cls.__name__ == "AnnealOptimizer"
        assert not caught  # Warning fires on instantiation, not attribute access.
        with pytest.raises(AttributeError):
            _ = helpers.no_such_helper
    finally:
        sys.modules.pop("_test_ase_helpers_contract", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        shim = _load_module_from_path("_test_ase_calculator_shim", shim_path)
    try:
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        assert issubclass(shim.RgpotCalculator, Calculator)
        assert issubclass(shim.RgpotCalculator, BaseCalculator)
    finally:
        sys.modules.pop("_test_ase_calculator_shim", None)


@pytest.mark.rpc
def test_rpc_calculator_contract_when_potserv_exists(potserv_server) -> None:
    from rgpot.ase import RgpotRpcCalculator

    if potserv_server.potential == "CuH2":
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(3)
        atoms.rattle(0.05, seed=1)
        ref_pot = rgpot.CuH2Pot()
    else:
        atoms = lj_dimer()
        ref_pot = rgpot.LJPot()

    ref_atoms = atoms.copy()
    ref_atoms.calc = RgpotCalculator(ref_pot)
    ref_energy = ref_atoms.get_potential_energy()
    ref_forces = ref_atoms.get_forces()

    calc = RgpotRpcCalculator(host=potserv_server.host, port=potserv_server.port)
    try:
        assert isinstance(calc, Calculator)
        assert isinstance(calc, BaseCalculator)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert forces.shape == (len(atoms), 3)
        assert np.all(np.isfinite(forces))
        assert np.max(np.abs(forces)) > 0.0
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=SUM_RULE_ATOL)
        np.testing.assert_allclose(
            energy, ref_energy, rtol=1e-12, atol=EXACT_MATCH_ATOL
        )
        np.testing.assert_allclose(
            forces, ref_forces, rtol=1e-12, atol=EXACT_MATCH_ATOL
        )
        assert calc.check_state(atoms) == []

        moved = atoms.copy()
        moved.positions[0, 0] += 0.05
        assert "positions" in calc.check_state(moved)

        for prop in ("stress", "dipole", "charges", "magmom"):
            with pytest.raises(PropertyNotImplementedError):
                calc.get_property(prop, atoms)

        calc.reset()
        assert calc.atoms is None
        assert calc.results == {}
    finally:
        calc.close()
