# Copyright (c) anneal developers
"""Tests for built-in cluster potentials and PotentialCalculator."""

from __future__ import annotations

import math

import numpy as np
import pytest

import anneal


def _fd_grad(pot, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        grad[idx] = (pot.eval(xp) - pot.eval(xm)) / (2.0 * eps)
    return grad


def test_pair_potential_lennard_jones_and_morse_defaults():
    lj = anneal.PairPotential(4)
    lj_explicit = anneal.PairPotential.lennard_jones(4)
    assert lj.n_points == 4
    assert lj.dim == 12
    assert lj.kind == "lj"
    assert lj.rho is None
    assert math.isclose(lj.r_min, 2.0 ** (1.0 / 6.0), rel_tol=1e-12)
    assert math.isclose(lj.length_scale, 1.0, rel_tol=1e-12)
    assert lj.bounds.dims == lj_explicit.bounds.dims == 12
    assert lj.bounds.slack == lj_explicit.bounds.slack

    rng = np.random.default_rng(7)
    x = rng.uniform(-1.2, 1.2, size=12)
    val, grad = lj.value_and_gradient(x)
    assert math.isclose(val, lj(x), rel_tol=1e-12)
    assert math.isclose(val, lj_explicit.eval(x), rel_tol=1e-12)
    np.testing.assert_allclose(grad, lj.grad(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(grad, _fd_grad(lj, x), rtol=1e-5, atol=1e-6)

    morse = anneal.PairPotential.morse(6, 6.0)
    assert morse.kind == "morse"
    assert morse.rho == 6.0
    assert math.isclose(morse.r_min, 1.0, rel_tol=1e-12)
    assert anneal.PairPotential.ccd_morse_minimum(38, 6.0) is not None
    assert anneal.PairPotential.ccd_morse_minimum(2, 6.0) is None
    xm = rng.uniform(-1.0, 1.0, size=18)
    np.testing.assert_allclose(morse.grad(xm), _fd_grad(morse, xm), rtol=1e-5, atol=1e-6)


def test_binary_lennard_jones_defaults_match_rust():
    blj = anneal.BinaryLennardJones(2, 2)
    assert blj.n_a == 2
    assert blj.n_b == 2
    assert blj.n_points == 4
    assert blj.dim == 12
    assert blj.epsilon_aa == 1.0
    assert blj.epsilon_ab == 1.0
    assert blj.epsilon_bb == 1.0
    assert blj.sigma_aa == 1.0
    assert blj.sigma_ab == 1.0
    assert blj.sigma_bb == 1.0

    lj4 = anneal.PairPotential.lennard_jones(4)
    rng = np.random.default_rng(11)
    x = rng.uniform(-1.2, 1.2, size=12)
    assert math.isclose(blj.eval(x), lj4.eval(x), rel_tol=1e-12)
    np.testing.assert_allclose(blj.grad(x), lj4.grad(x), rtol=1e-12, atol=1e-12)

    dm = anneal.BinaryLennardJones.doye_meyer(2, 3, 1.3)
    assert dm.sigma_aa == 1.0
    assert math.isclose(dm.sigma_bb, 1.3, rel_tol=1e-12)
    assert math.isclose(dm.sigma_ab, 1.15, rel_tol=1e-12)
    xdm = rng.uniform(-1.5, 1.5, size=dm.dim)
    np.testing.assert_allclose(dm.grad(xdm), _fd_grad(dm, xdm), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("ctor", "expected_family", "expected_nm"),
    [
        (anneal.SuttonChen.sc_12_6, "12-6", (12, 6)),
        (anneal.SuttonChen.sc_9_6, "9-6", (9, 6)),
        (anneal.SuttonChen.sc_10_8, "10-8", (10, 8)),
    ],
)
def test_sutton_chen_families_and_gradient(ctor, expected_family, expected_nm):
    pot = ctor(4)
    by_name = anneal.SuttonChen(4, expected_family)
    assert pot.n_points == 4
    assert pot.dim == 12
    assert pot.family == expected_family
    assert (pot.n_exp, pot.m_exp) == expected_nm
    assert math.isclose(pot.c, by_name.c, rel_tol=1e-12)
    assert pot.basin_hopping_temperature > 0.0

    rng = np.random.default_rng(19)
    x = rng.uniform(-1.0, 1.0, size=12)
    val, grad = pot.value_and_gradient(x)
    assert math.isclose(val, by_name.eval(x), rel_tol=1e-12)
    np.testing.assert_allclose(grad, _fd_grad(pot, x), rtol=1e-5, atol=1e-6)


def test_gupta_constructors_constants_and_gradient():
    assert anneal.COPPER_Z == 29
    assert anneal.PLATINUM_Z == 78
    assert anneal.GOLD_Z == 79
    assert anneal.Gupta.copper_gold_species(2, 1) == [29, 29, 79]
    assert anneal.Gupta.platinum_copper_species(1, 2) == [78, 29, 29]
    assert anneal.Gupta.homonuclear_r0(anneal.PLATINUM_Z) is not None
    assert anneal.Gupta.homonuclear_r0(1) is None

    zn = anneal.Gupta.zinc(4)
    pt = anneal.Gupta.platinum(4)
    cu_au = anneal.Gupta.copper_gold(2, 2)
    pt_cu = anneal.Gupta.platinum_copper(2, 2)
    by_z = anneal.Gupta.elements([29, 29, 79, 79])

    rng = np.random.default_rng(23)
    x = rng.uniform(-1.5, 1.5, size=12)
    for pot in (zn, pt, cu_au, pt_cu):
        np.testing.assert_allclose(pot.grad(x), _fd_grad(pot, x), rtol=1e-5, atol=1e-6)
    assert math.isclose(cu_au.eval(x), by_z.eval(x), rel_tol=1e-12)


def test_tip4p_and_gmin_tip4p_potentials_and_helpers():
    tip = anneal.Tip4pCluster(3)
    gtip = anneal.GminTip4p(3)
    assert tip.n_molecules == 3
    assert tip.dim == 18
    assert gtip.n_molecules == 3
    assert gtip.dim == 18
    assert anneal.Tip4pCluster.ccd_minimum(3) is not None
    assert anneal.Tip4pCluster.ccd_minimum(1) is None

    sites_template = anneal.gmin_tip4p_sites()
    assert sites_template.shape == (4, 3)

    rng = np.random.default_rng(29)
    # Place three water COMs well separated so R^-12 stays moderate.
    block = np.zeros(18, dtype=np.float64)
    block[0:3] = [0.0, 0.0, 0.0]
    block[3:6] = [3.2, 0.2, -0.1]
    block[6:9] = [1.5, 3.0, 0.3]
    block[9:18] = rng.uniform(-0.4, 0.4, size=9)

    folded = tip.fold_rotations(block)
    assert math.isclose(tip.eval(block), tip.eval(folded), rel_tol=1e-10)
    np.testing.assert_allclose(tip.grad(block), _fd_grad(tip, block), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gtip.grad(block), _fd_grad(gtip, block), rtol=1e-4, atol=1e-5)

    interleaved = anneal.interleaved_gmin_from_block(block)
    round_trip = anneal.block_from_interleaved_gmin(interleaved)
    np.testing.assert_allclose(round_trip, block, rtol=1e-12, atol=1e-12)

    for mol_idx in range(3):
        mol_sites = tip.molecule_sites(block, mol_idx)
        assert mol_sites.shape == (4, 3)

    # Assemble O-H-H Cartesian coordinates from GminTip4p reference sites.
    ohh = []
    for centre in ([0.0, 0.0, 0.0], [3.2, 0.2, -0.1], [1.5, 3.0, 0.3]):
        placed_ohh = sites_template[:3] + np.asarray(centre, dtype=np.float64)
        ohh.extend(placed_ohh.ravel())
    ohh_arr = np.asarray(ohh, dtype=np.float64)
    rigid = anneal.rigid_from_ohh(ohh_arr)
    assert rigid.shape == (18,)
    e_ohh = anneal.energy_ohh(ohh_arr)
    assert math.isclose(e_ohh, gtip.eval(rigid), rel_tol=1e-10, abs_tol=1e-10)


def test_native_rigid_interleaved_search_matches_block_rust_path():
    pot = anneal.GminTip4p(2)
    block_start = np.array(
        [
            0.0,
            0.0,
            0.0,
            3.2,
            0.2,
            -0.1,
            0.15,
            -0.1,
            0.05,
            -0.2,
            0.1,
            0.25,
        ],
        dtype=np.float64,
    )
    interleaved_start = anneal.interleaved_gmin_from_block(block_start)

    block = anneal.cluster_search(
        pot,
        None,
        2,
        160,
        seed=13,
        rigid=True,
        layout="block",
        start=block_start,
    )
    interleaved = anneal.cluster_search(
        pot,
        None,
        2,
        160,
        seed=13,
        rigid=True,
        layout="interleaved",
        start=interleaved_start,
    )

    interleaved_best_block = anneal.block_from_interleaved_gmin(interleaved["best"])
    assert interleaved["best_energy"] == pytest.approx(block["best_energy"], abs=1e-12)
    np.testing.assert_allclose(interleaved_best_block, block["best"], rtol=0.0, atol=1e-12)

    block_forces = -pot.grad(block["best"])
    interleaved_forces = anneal.interleaved_gmin_from_block(
        -pot.grad(interleaved_best_block)
    )
    np.testing.assert_allclose(
        interleaved_forces,
        anneal.interleaved_gmin_from_block(block_forces),
        rtol=0.0,
        atol=1e-12,
    )


def test_native_potentials_work_with_quench_cluster_search_and_objectives():
    lj4 = anneal.PairPotential.lennard_jones(4)
    rng = np.random.default_rng(31)
    x0 = rng.uniform(-0.9, 0.9, size=12)

    q_native = anneal.quench(lj4, x0, 1e-6, max_iter=500)
    q_py = anneal.quench(lj4.value_and_gradient, x0, 1e-6, max_iter=500)
    assert q_native["calls"] > 0
    assert anneal.energy_hits(q_native["energy"], -6.0)
    assert math.isclose(q_native["energy"], q_py["energy"], rel_tol=1e-10, abs_tol=1e-10)
    assert q_native["calls"] == q_py["calls"]

    s_native = anneal.cluster_search(lj4, None, 4, 400, seed=5, target=-6.0)
    s_py = anneal.cluster_search(lj4.eval, lj4.grad, 4, 400, seed=5, target=-6.0)
    assert anneal.energy_hits(s_native["best_energy"], -6.0)
    assert math.isclose(
        s_native["best_energy"],
        s_py["best_energy"],
        rel_tol=1e-10,
        abs_tol=1e-10,
    )
    assert s_native["hops"] == s_py["hops"]

    try:
        s_ras = anneal.cluster_search(
            lj4,
            None,
            4,
            400,
            seed=5,
            ras=True,
            target=-6.0,
        )
    except ValueError as exc:
        assert str(exc) == "ras=True requires building with the graphkey feature"
    else:
        assert anneal.energy_hits(s_ras["best_energy"], -6.0)

    obj = lj4.to_objective()
    assert obj.dim == 12
    np.testing.assert_allclose(obj.grad(x0), lj4.grad(x0), rtol=1e-12, atol=1e-12)
    opt_out = anneal.global_optimize_objective(obj, 500, seed=5, x0=x0)
    assert math.isfinite(opt_out["best_val"])
    assert opt_out["best_val"] <= lj4.eval(x0)


def test_potential_calculator_standalone_and_with_ase():
    lj4 = anneal.PairPotential.lennard_jones(4)
    calc = anneal.PotentialCalculator(lj4)

    class _DummyAtoms:
        def __init__(self, positions: np.ndarray) -> None:
            self._positions = np.array(positions, dtype=np.float64)

        def get_positions(self) -> np.ndarray:
            return self._positions.copy()

        def __len__(self) -> int:
            return int(self._positions.shape[0])

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.55, 0.95, 0.0],
            [0.55, 0.32, 0.9],
        ],
        dtype=np.float64,
    )
    dummy = _DummyAtoms(pos)
    energy = calc.get_potential_energy(dummy)
    forces = calc.get_forces(dummy)
    assert math.isclose(energy, lj4.eval(pos.ravel()), rel_tol=1e-12)
    np.testing.assert_allclose(
        forces,
        -lj4.grad(pos.ravel()).reshape(4, 3),
        rtol=1e-12,
        atol=1e-12,
    )

    ase_mod = pytest.importorskip("ase")
    atoms = ase_mod.Atoms("Ar4", positions=pos.copy())
    atoms.calc = anneal.PotentialCalculator(lj4)
    opt = anneal.Anneal(atoms, budget=400, seed=7, mode="search", logfile=None)
    assert opt.run(fmax=1e-4) is True
    assert anneal.energy_hits(atoms.get_potential_energy(), -6.0)


def test_potential_calculator_refreshes_after_in_place_move():
    """An in-place move must not reuse the energy cached for the old geometry."""
    ase_mod = pytest.importorskip("ase")
    lj4 = anneal.PairPotential.lennard_jones(4)
    edge = 2.0 ** (1.0 / 6.0)
    pts = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts -= pts.mean(axis=0)
    pts *= 0.996 * edge / np.linalg.norm(pts[0] - pts[1])
    atoms = ase_mod.Atoms("Ar4", positions=pts.copy())
    calc = anneal.PotentialCalculator(lj4)
    atoms.calc = calc
    before = float(calc.get_property("energy", atoms))
    atoms.positions[0, 0] += 0.35
    assert calc.calculation_required(atoms, ["energy"]) is True
    moved = float(calc.get_property("energy", atoms))
    fresh = atoms.copy()
    fresh.calc = anneal.PotentialCalculator(lj4)
    truth = float(fresh.get_potential_energy())
    direct = float(lj4.eval(np.asarray(atoms.get_positions(), dtype=np.float64).ravel()))
    assert moved == pytest.approx(truth, rel=0, abs=1e-12)
    assert moved == pytest.approx(direct, rel=0, abs=1e-12)
    assert moved != pytest.approx(before, rel=1e-6, abs=1e-6)
