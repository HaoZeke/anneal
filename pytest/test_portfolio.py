"""Tests for the native Thompson-allocated portfolio driver."""

import math

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from experiments.tensor_surrogate import AdditiveSurrogate  # noqa: E402


def _styblinski_tang(dim):
    def fn(x):
        return 0.5 * float(np.sum(x**4 - 16.0 * x**2 + 5.0 * x))

    def grad(x):
        return 0.5 * (4.0 * x**3 - 32.0 * x + 5.0)

    low = np.full(dim, -5.0)
    high = np.full(dim, 5.0)
    return fn, grad, low, high


def _rastrigin(dim):
    def fn(x):
        return 10.0 * dim + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def grad(x):
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    low = np.full(dim, -5.12)
    high = np.full(dim, 5.12)
    return fn, grad, low, high


def test_budget_is_never_exceeded():
    dim = 6
    fn, grad, low, high = _rastrigin(dim)
    out = anneal.global_optimize(fn, low, high, budget=600, seed=7, grad_fn=grad)
    assert out["n_evals"] + out["n_grads"] <= 600
    assert math.isfinite(out["best_val"])


def test_budget_respected_without_gradients():
    dim = 4
    fn, _, low, high = _rastrigin(dim)
    out = anneal.global_optimize(fn, low, high, budget=400, seed=3)
    assert out["n_evals"] <= 400
    assert out["n_grads"] == 0
    assert math.isfinite(out["best_val"])


def test_portfolio_reaches_global_basin_on_styblinski_tang():
    dim = 6
    fn, grad, low, high = _styblinski_tang(dim)
    hits = 0
    for seed in range(4):
        out = anneal.global_optimize(
            fn, low, high, budget=1500, seed=seed, grad_fn=grad
        )
        if out["best_val"] < -39.166 * dim * 0.99:
            hits += 1
    assert hits >= 3, f"global basin hit on {hits}/4 seeds"


def test_portfolio_beats_uniform_random():
    dim = 6
    fn, grad, low, high = _styblinski_tang(dim)
    budget = 1500
    out = anneal.global_optimize(fn, low, high, budget=budget, seed=11, grad_fn=grad)
    rng = np.random.default_rng(11)
    rand_best = min(fn(rng.uniform(low, high)) for _ in range(budget))
    assert out["best_val"] < rand_best


def test_arm_statistics_are_reported():
    dim = 10
    fn, grad, low, high = _rastrigin(dim)
    out = anneal.global_optimize(fn, low, high, budget=4000, seed=5, grad_fn=grad)
    assert "explore" in out["arm_pulls"]
    assert sum(out["arm_pulls"].values()) > 0
    assert set(out["arm_successes"]) == set(out["arm_pulls"])


def test_additive_surrogate_from_points_recovers_separable():
    rng = np.random.default_rng(5)
    dim = 3
    low = np.full(dim, -2.0)
    high = np.full(dim, 2.0)
    X = rng.uniform(low, high, size=(400, dim))
    y = np.sum(X**2, axis=1)
    surr = AdditiveSurrogate.from_points(X, y, low, high, degree=6)
    samples = surr.sample(64, 0.05, rng)
    # Low-temperature draws concentrate near the separable minimum at 0.
    assert float(np.median(np.abs(samples))) < 0.5
    assert surr.pilot_work_units() == 0.0
