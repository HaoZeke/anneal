# Copyright (c) anneal developers
"""Parity tests for box, DMC, portfolio, and BasinBias Python bindings."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

import anneal


def _sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.dot(x - 0.5, x - 0.5))


def _sphere_grad(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 2.0 * (x - 0.5)


def test_dmc_population_defaults_match_rust_exactly():
    assert anneal.DEFAULT_TARGET_WALKERS == 24
    assert anneal.DEFAULT_STEPS_PER_CONTROL == 2
    assert anneal.DEFAULT_BETA0 == 1.0

    sig = inspect.signature(anneal.dmc_population_optimize)
    assert sig.parameters["target_n"].default is None
    assert sig.parameters["steps_per_control"].default == 2
    assert sig.parameters["beta0"].default == 1.0

    low = np.full(4, -3.0, dtype=np.float64)
    high = np.full(4, 3.0, dtype=np.float64)
    rec_n = anneal.recommend_target_n(200, 4)
    assert 8 <= rec_n <= 36

    out_default = anneal.dmc_population_optimize(_sphere, low, high, 200, seed=11)
    out_explicit = anneal.dmc_population_optimize(
        _sphere,
        low,
        high,
        200,
        seed=11,
        target_n=rec_n,
        steps_per_control=anneal.DEFAULT_STEPS_PER_CONTROL,
        beta0=anneal.DEFAULT_BETA0,
    )
    assert math.isclose(out_default["best_val"], out_explicit["best_val"], rel_tol=1e-12)
    np.testing.assert_allclose(out_default["best_pos"], out_explicit["best_pos"], rtol=1e-12)
    assert out_default["n_evals"] == out_explicit["n_evals"]
    assert out_default["n_grads"] == out_explicit["n_grads"]
    assert out_default["controls"] == out_explicit["controls"]
    assert out_default["final_population"] == out_explicit["final_population"]

    with pytest.raises(ValueError, match="target_n"):
        anneal.dmc_population_optimize(_sphere, low, high, 100, target_n=1)
    with pytest.raises(ValueError, match="beta0"):
        anneal.dmc_population_optimize(_sphere, low, high, 100, beta0=0.0)


def test_box_ensemble_and_minimize_expose_escape_and_coverage_defaults():
    assert anneal.IDENTITY_TOL == 1e-3
    low = np.full(3, -4.0, dtype=np.float64)
    high = np.full(3, 4.0, dtype=np.float64)

    out_default = anneal.box_ensemble_optimize(
        _sphere,
        low,
        high,
        budget=160,
        seed=3,
        grad_fn=_sphere_grad,
        replicas=2,
    )
    out_explicit = anneal.box_ensemble_optimize(
        _sphere,
        low,
        high,
        budget=160,
        seed=3,
        grad_fn=_sphere_grad,
        replicas=2,
        identity_tol=1e-3,
        shared_deposits=8,
        shared_visit_policy="tabu",
        escape="gaussian",
        coverage_shared=True,
        coverage_radius=0.05,
        coverage_neighbors=0,
        coverage_height=0.1,
        coverage_well_tempering=5.0,
        coverage_peer_weight=1.0,
    )
    assert math.isclose(out_default["best_val"], out_explicit["best_val"], rel_tol=1e-12)
    np.testing.assert_allclose(out_default["best_pos"], out_explicit["best_pos"], rtol=1e-12)

    out_langevin_colored = anneal.box_ensemble_optimize(
        _sphere,
        low,
        high,
        budget=180,
        seed=5,
        grad_fn=_sphere_grad,
        replicas=2,
        shared_visit_policy="recognition",
        escape="langevin",
        langevin_steps=16,
        langevin_omega0=0.2,
        langevin_dt=0.01,
        langevin_noise="colored",
    )
    assert math.isfinite(out_langevin_colored["best_val"])

    out_langevin_white = anneal.ensemble_optimize(
        _sphere,
        low,
        high,
        budget=180,
        seed=5,
        grad_fn=_sphere_grad,
        replicas=2,
        escape="langevin",
        langevin_steps=8,
        langevin_noise="white",
        langevin_friction=1.5,
        coverage_height=0.05,
        coverage_well_tempering=3.0,
        coverage_peer_weight=0.5,
    )
    assert math.isfinite(out_langevin_white["best_val"])

    min_res = anneal.minimize(
        _sphere,
        np.zeros(3),
        list(zip(low, high)),
        jac=_sphere_grad,
        budget=160,
        seed=5,
        replicas=2,
        escape="langevin",
        langevin_steps=8,
        coverage_height=0.05,
        coverage_well_tempering=3.0,
        coverage_peer_weight=0.5,
    )
    assert min_res.success is True
    assert math.isfinite(min_res.fun)

    with pytest.raises(ValueError, match="requires grad_fn"):
        anneal.ensemble_optimize(_sphere, low, high, budget=80, escape="langevin")
    with pytest.raises(TypeError, match="integer"):
        anneal.ensemble_optimize(
            _sphere,
            low,
            high,
            budget=80,
            grad_fn=_sphere_grad,
            coverage_neighbors=1.5,
        )


def test_global_optimize_and_objective_parity():
    low = np.full(3, -3.0, dtype=np.float64)
    high = np.full(3, 3.0, dtype=np.float64)

    sig = inspect.signature(anneal.global_optimize)
    assert sig.parameters["coverage_radius"].default == 0.05
    assert sig.parameters["coverage_neighbors"].default == 0
    assert sig.parameters["coverage_height"].default == 0.1
    assert sig.parameters["coverage_well_tempering"].default == 5.0
    assert sig.parameters["coverage_peer_weight"].default == 1.0

    multi_default = anneal.global_optimize(
        _sphere,
        low,
        high,
        240,
        seed=7,
        grad_fn=_sphere_grad,
        replicas=2,
    )
    multi_explicit = anneal.global_optimize(
        _sphere,
        low,
        high,
        240,
        seed=7,
        grad_fn=_sphere_grad,
        replicas=2,
        coverage_height=0.1,
        coverage_well_tempering=5.0,
        coverage_peer_weight=1.0,
    )
    assert math.isclose(multi_default["best_val"], multi_explicit["best_val"], rel_tol=1e-12)

    bounds = anneal.Bounds(low, high, 1e-9)
    obj = anneal.PyObjective(_sphere, bounds, _sphere_grad)
    out_auto = anneal.global_optimize_objective(obj, 150, seed=4, policy="auto")
    out_legacy = anneal.global_optimize_objective(obj, 150, seed=4, policy="legacy")
    out_noisy = anneal.global_optimize_objective(
        obj,
        150,
        seed=4,
        use_gradient=False,
        noise_sigma=0.05,
        policy="legacy",
    )
    assert math.isfinite(out_auto["best_val"])
    assert math.isfinite(out_legacy["best_val"])
    assert math.isfinite(out_noisy["best_val"])
    with pytest.raises(ValueError, match="policy"):
        anneal.global_optimize_objective(obj, 50, policy="invalid")


def test_basin_bias_osa_and_standalone_optimisers():
    # Rust BasinBias::new explicitly permits w0 == 0.0 (region index without penalty).
    zero_bias = anneal.BasinBias(4, merge_radius=0.05, w0=0.0, gamma=5.0)
    assert zero_bias.height == 0.0
    assert zero_bias.merge_radius == 0.05

    x1 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    x2 = x1 * 1.4
    zero_bias.deposit(x1, 0.8)
    assert zero_bias.n_basins == 1
    assert zero_bias.potential(x1) == 0.0

    bias = anneal.BasinBias(4, merge_radius=0.05, w0=0.25, gamma=5.0)
    bias.deposit(x1, 0.8)
    bias.deposit(x1, 0.8)
    bias.deposit(x2, 0.8)
    assert bias.n_basins == 2
    assert bias.well_depth(0) > bias.well_depth(1) > 0.0
    assert len(bias.wells()) == 2
    top = bias.deepest_wells(1)
    assert len(top) == 1
    assert math.isclose(top[0][1], bias.deepest, rel_tol=1e-12)
    bias.merge_radius = 0.1
    bias.height = 0.5
    assert bias.merge_radius == 0.1
    assert bias.height == 0.5
    bias.cap_basins(2)
    with pytest.raises(IndexError):
        bias.well_depth(9)

    acc_rate, mean_evals = anneal.osa_acceptance_rate(0.2, 1.0, 0.1, trials=500, seed=1)
    assert 0.0 <= acc_rate <= 1.0
    assert mean_evals >= 1.0

    low = np.full(2, -2.0, dtype=np.float64)
    high = np.full(2, 2.0, dtype=np.float64)
    for fn in (anneal.gpmd_optimize, anneal.amsa_optimize, anneal.bfwt_optimize):
        res = fn(_sphere, low, high, 60, seed=2, grad_fn=_sphere_grad)
        assert isinstance(res["best_pos"], np.ndarray)
        assert res["best_pos"].dtype == np.float64
