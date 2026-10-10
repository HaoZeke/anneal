"""A positive portfolio budget must return an evaluated feasible incumbent."""

import math

import numpy as np
import pytest

import anneal


@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize("policy", ["auto", "legacy"])
def test_values_portfolio_tiny_budget_returns_a_paid_incumbent(budget, policy):
    evaluations = []
    low = np.array([-2.0, -3.0])
    high = np.array([4.0, 5.0])
    minimum = np.array([0.375, -0.625])

    def objective(x):
        point = np.asarray(x, dtype=float).copy()
        delta = point - minimum
        value = 7.0 + float(np.dot(delta, delta))
        evaluations.append((point, value))
        return value

    result = anneal.global_optimize(
        objective,
        low,
        high,
        budget=budget,
        seed=7,
        grad_fn=None,
        policy=policy,
    )

    assert 0 < len(evaluations) <= budget
    assert result["n_evals"] == len(evaluations)
    assert result["n_grads"] == 0
    assert result["n_evals"] + result["n_grads"] <= budget
    assert math.isfinite(result["best_val"])
    assert result["best_pos"].shape == low.shape
    assert np.all(np.isfinite(result["best_pos"]))
    assert np.all(result["best_pos"] >= low)
    assert np.all(result["best_pos"] <= high)
    assert all(np.all(point >= low) and np.all(point <= high) for point, _ in evaluations)
    assert any(
        np.array_equal(result["best_pos"], point) and result["best_val"] == value
        for point, value in evaluations
    ), "the returned coordinates and value must match a charged evaluation"
