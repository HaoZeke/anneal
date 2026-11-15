"""ASE calculator contract for RgpotCalculator."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.optimize import BFGS

import rgpot
from rgpot.ase import AnnealOptimizer, AnnealRunner, RgpotCalculator

from support import lj_crystal, lj_dimer, pt7, raw_bytes, si_crystal


def test_import_rgpot_does_not_import_ase() -> None:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    pythonpath = env.get("PYTHONPATH", "")
    extra = str(root / "python")
    env["PYTHONPATH"] = extra if not pythonpath else extra + os.pathsep + pythonpath
    script = (
        "import sys, rgpot; "
        "assert 'ase' not in sys.modules, sorted(sys.modules); "
        "assert rgpot.__version__"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_import_ase_subpackage_loads_ase() -> None:
    import rgpot.ase as ase_mod

    assert hasattr(ase_mod, "RgpotCalculator")


def test_implemented_properties_follow_caps() -> None:
    lj = RgpotCalculator("lj")
    assert lj.implemented_properties == [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "energies",
        "stresses",
    ]
    assert callable(lj.potential.evaluate_per_atom)
    if not rgpot.has_fortran_pots:
        pytest.skip("this build has no Fortran potentials")
    eam = RgpotCalculator("eam_al")
    assert eam.implemented_properties == ["energy", "free_energy", "forces"]
    assert eam.protocol_reentrant is True


def test_free_energy_equals_energy() -> None:
    atoms = lj_dimer()
    atoms.calc = RgpotCalculator("lj")
    energy = atoms.get_potential_energy()
    assert atoms.get_potential_energy(force_consistent=True) == energy
    assert atoms.calc.results["free_energy"] == energy


def test_lj_forces_match_finite_difference() -> None:
    atoms = lj_dimer()
    atoms.calc = RgpotCalculator("lj")
    analytic = atoms.get_forces()
    numeric = calculate_numerical_forces(atoms, eps=1e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build")
def test_eam_forces_match_finite_difference() -> None:
    atoms = pt7()
    atoms.calc = RgpotCalculator("eam_al")
    analytic = atoms.get_forces()
    numeric = calculate_numerical_forces(atoms, eps=1e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)


def test_atom_types_accept_int64_and_python_list() -> None:
    atoms = lj_dimer()
    positions = np.asarray(atoms.positions, dtype=np.float64)
    box = np.asarray(atoms.cell.array, dtype=np.float64)
    types32 = np.asarray(atoms.numbers, dtype=np.int32)
    pot = rgpot.LJPot()
    reference = pot.evaluate(positions, types32, box)

    types64 = np.asarray(atoms.numbers, dtype=np.int64)
    from_i64 = pot.evaluate(positions, types64, box)
    from_list = pot.evaluate(positions, atoms.numbers.tolist(), box)
    np.testing.assert_array_equal(from_i64["forces"], reference["forces"])
    assert from_i64["energy"] == reference["energy"]
    np.testing.assert_array_equal(from_list["forces"], reference["forces"])
    assert from_list["energy"] == reference["energy"]

    huge = np.array([2**40, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="internal integer dtype"):
        pot.evaluate(positions, huge, box)
    with pytest.raises(TypeError, match="atom_types"):
        pot.evaluate(positions, [1.5, 1], box)


def test_cluster_stress_matches_finite_difference() -> None:
    """A non-periodic cell is the volume ASE strains, not the padded cube."""
    atoms = lj_dimer()
    assert not np.any(atoms.pbc)
    atoms.calc = RgpotCalculator("lj", cutoff=8.0)
    analytic = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-5, voigt=True)
    np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=0.0)


def test_lj_stress_matches_finite_difference() -> None:
    atoms = lj_crystal()
    atoms.calc = RgpotCalculator("lj", cutoff=8.0)
    analytic = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)


def test_stillinger_weber_si_stress_matches_finite_difference() -> None:
    pot = rgpot.StillingerWeberSi()
    assert pot.caps()["stress"] is True
    atoms = si_crystal()
    atoms.calc = RgpotCalculator(pot)
    assert "stress" in atoms.calc.implemented_properties
    analytic = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no SWPot in this build")
def test_sw_stress_matches_finite_difference() -> None:
    atoms = si_crystal()
    atoms.calc = RgpotCalculator("sw")
    assert "stress" in atoms.calc.implemented_properties
    analytic = atoms.get_stress()
    numeric = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no SWPot in this build")
def test_sw_stress_is_the_1985_stress_scaled_by_epsilon() -> None:
    atoms = si_crystal()
    args = (atoms.positions, atoms.numbers, np.ascontiguousarray(atoms.cell.array))
    eon = rgpot.SWPot().evaluate(*args)
    published = rgpot.StillingerWeberSi().evaluate(*args)
    ratio = 2.16826 * 1.06767385 / (3.4723e-12 * 1.0e-7 / 1.602176634e-19)
    assert eon["energy"] == pytest.approx(ratio * published["energy"], rel=1e-13)
    np.testing.assert_allclose(
        eon["stress"], ratio * published["stress"], atol=1e-14, rtol=0.0
    )


def test_single_point_round_trip() -> None:
    atoms = lj_dimer()
    atoms.calc = RgpotCalculator("lj")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    bare = atoms.copy()
    bare.calc = SinglePointCalculator(bare, energy=energy, forces=forces.copy())
    assert bare.get_potential_energy() == energy
    np.testing.assert_array_equal(bare.get_forces(), forces)


def test_trajectory_round_trip(tmp_path: Path) -> None:
    atoms = lj_dimer()
    atoms.calc = RgpotCalculator("lj", directory=str(tmp_path / "calc"))
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    path = tmp_path / "lj.traj"
    write(path, atoms)
    images = read(path, index=":")
    assert len(images) == 1
    assert images[0].get_potential_energy() == pytest.approx(energy)
    np.testing.assert_allclose(images[0].get_forces(), forces)


def test_reset_and_check_state() -> None:
    atoms = lj_dimer()
    calc = RgpotCalculator("lj")
    atoms.calc = calc
    atoms.get_potential_energy()
    assert calc.check_state(atoms) == []
    assert "energy" in calc.results
    calc.reset()
    assert calc.results == {}
    moved = atoms.copy()
    moved.positions[1, 0] += 0.2
    assert "positions" in calc.check_state(moved)


def test_get_hessian_is_not_implemented() -> None:
    atoms = lj_dimer()
    calc = RgpotCalculator("lj")
    with pytest.raises(NotImplementedError):
        calc.get_hessian(atoms)


def test_unknown_kind_and_parameter() -> None:
    with pytest.raises(ValueError, match="unknown potential kind"):
        RgpotCalculator("not-a-potential")
    with pytest.raises(ValueError, match="unknown LJ"):
        RgpotCalculator("lj", nope=1.0)


def test_protocol_does_not_mutate_calculator() -> None:
    atoms = lj_dimer()
    calc = RgpotCalculator("lj")
    before = dict(calc.results)
    row = calc.__rg_evaluate__(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        np.asarray(atoms.cell.array),
        atoms.pbc,
    )
    assert calc.results == before
    assert calc.atoms is None
    assert row["free_energy"] == row["energy"]
    assert "stress" in row


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build")
def test_bfgs_two_calculators_agree(tmp_path: Path) -> None:
    def relax(folder: Path) -> float:
        atoms = pt7()
        atoms.calc = RgpotCalculator("eam_al", directory=str(folder))
        BFGS(atoms, logfile=None, trajectory=None).run(fmax=0.05, steps=200)
        return float(atoms.get_potential_energy())

    first = relax(tmp_path / "a")
    second = relax(tmp_path / "b")
    assert abs(first - second) < 1e-8


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build")
def test_nonperiodic_pt7_ignores_cell() -> None:
    """A 10 Angstrom cell with pbc false matches the same cluster with no cell."""
    from ase import Atoms

    cluster = pt7()
    cluster.set_cell([10.0, 10.0, 10.0])
    cluster.set_pbc(False)
    bare = Atoms(
        numbers=cluster.get_atomic_numbers(),
        positions=cluster.get_positions().copy(),
        pbc=False,
    )
    calc = RgpotCalculator("eam_al")
    cluster.calc = calc
    confined = raw_bytes(cluster.get_potential_energy(), cluster.get_forces())
    bare.calc = calc
    free = raw_bytes(bare.get_potential_energy(), bare.get_forces())
    assert confined == free
    # Periodic x and z 40 Angstrom apart keep every image out of the
    # 7.1 Angstrom cutoff, so the slab equals the cluster.
    mixed = cluster.copy()
    mixed.set_cell([40.0, 10.0, 40.0])
    mixed.set_pbc([True, False, True])
    mixed.calc = calc
    assert mixed.get_potential_energy() == pytest.approx(
        bare.get_potential_energy(), rel=1e-13
    )
    np.testing.assert_allclose(mixed.get_forces(), bare.get_forces(), atol=1e-12)


def test_rpc_bfgs_when_potserv_exists(tmp_path: Path) -> None:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    if not rgpot.has_fortran_pots:
        pytest.skip("this build has no Fortran potentials")
    from rgpot.ase import RgpotRpcCalculator

    atoms = pt7()
    atoms.calc = RgpotCalculator("eam_al", directory=str(tmp_path / "in"))
    BFGS(atoms, logfile=None, trajectory=None).run(fmax=0.05, steps=200)
    in_process = float(atoms.get_potential_energy())

    rpc_atoms = pt7()
    rpc = RgpotRpcCalculator.spawn(binary, potential="EAMAl")
    try:
        rpc_atoms.calc = rpc
        BFGS(rpc_atoms, logfile=None, trajectory=None).run(fmax=0.05, steps=200)
        remote = float(rpc_atoms.get_potential_energy())
    finally:
        rpc.close()
    assert abs(in_process - remote) < 1e-8


def test_anneal_runner_is_not_an_optimizer() -> None:
    from ase.optimize.optimize import Optimizer

    atoms = lj_dimer()
    atoms.calc = RgpotCalculator("lj")
    runner = AnnealRunner(atoms, budget=10, seed=1)
    assert not isinstance(runner, Optimizer)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        old = AnnealOptimizer(atoms, budget=10, seed=1)
    assert any(issubclass(item.category, DeprecationWarning) for item in caught)
    assert isinstance(old, AnnealRunner)


def test_deprecated_rpc_import_path() -> None:
    shim = (
        Path(__file__).resolve().parents[2]
        / "CppCore"
        / "rgpot"
        / "rpc"
        / "ase_calculator.py"
    )
    if not shim.is_file():
        pytest.skip("RPC shim is not in this tree")
    env = os.environ.copy()
    root = str(shim.parents[0])
    pythonpath = env.get("PYTHONPATH", "")
    src = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (root, src, pythonpath) if part
    )
    script = """
import warnings
warnings.simplefilter("always", DeprecationWarning)
with warnings.catch_warnings(record=True) as caught:
    from ase_calculator import RgpotCalculator
assert any(issubclass(item.category, DeprecationWarning) for item in caught)
assert RgpotCalculator.__name__ == "RgpotRpcCalculator"
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


def test_kind_string_matches_object() -> None:
    atoms = lj_dimer()
    by_name = RgpotCalculator("lennard-jones", u0=1.3, cutoff=7.0, psi=0.9)
    by_object = RgpotCalculator(rgpot.LJPot(u0=1.3, cutoff=7.0, psi=0.9))
    left = by_name.__rg_evaluate__(
        atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc
    )
    right = by_object.__rg_evaluate__(
        atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc
    )
    assert raw_bytes(left["energy"], left["forces"]) == raw_bytes(
        right["energy"], right["forces"]
    )
