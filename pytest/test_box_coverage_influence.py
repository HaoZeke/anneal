"""Delivered coverage and direct influence on search decisions are distinct."""

import anneal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "driver,with_gradient",
    [
        ("box_ensemble_optimize", True),
        ("ensemble_optimize", True),
        ("ensemble_optimize", False),
    ],
)
@pytest.mark.parametrize("history", ["shared", "private", "none"])
def test_box_results_retain_direct_peer_influence(driver, with_gradient, history):
    calls = {"evals": 0, "grads": 0}

    def objective(x):
        calls["evals"] += 1
        return float(np.dot(x, x))

    def gradient(x):
        calls["grads"] += 1
        return 2.0 * x

    result = getattr(anneal, driver)(
        objective,
        np.full(5, -2.0),
        np.full(5, 2.0),
        budget=512,
        seed=7,
        grad_fn=gradient if with_gradient else None,
        replicas=4,
        history=history,
    )
    assert result["n_evals"] == calls["evals"]
    assert result["n_grads"] == calls["grads"]
    assert calls["evals"] + calls["grads"] <= 512
    decisions = result["coverage_decisions"]
    assert 0 < decisions["comparisons"] <= result["hops"]
    assert decisions["unresolved"] == 0
    assert 0 <= decisions["accepted"] <= decisions["comparisons"]
    assert (
        0
        <= decisions["probability_changes"]
        <= decisions["peer_delta_changes"]
        <= decisions["peer_overlap"]
        <= decisions["comparisons"]
    )
    assert 0 <= decisions["drawn_comparisons"] <= decisions["comparisons"]
    assert 0 <= decisions["drawn_disagreements"] <= min(
        decisions["drawn_comparisons"], decisions["probability_changes"]
    )
    assert 0.0 <= decisions["max_probability_change"] <= 1.0
    assert 0.0 <= decisions["probability_change_sum"] <= decisions["probability_changes"]
    for key in ("max_abs_peer_delta", "max_abs_peer_delta_over_temperature"):
        assert np.isfinite(decisions[key]) and decisions[key] >= 0.0
    if history != "shared":
        assert decisions["peer_overlap"] == 0
        assert decisions["probability_change_sum"] == 0.0
        assert decisions["max_probability_change"] == 0.0
        assert decisions["max_abs_peer_delta"] == 0.0
        assert decisions["max_abs_peer_delta_over_temperature"] == 0.0


def test_values_portfolio_reports_no_box_acceptance_decisions():
    result = anneal.ensemble_optimize(
        lambda x: float(np.dot(x, x)),
        np.full(5, -2.0),
        np.full(5, 2.0),
        budget=128,
        seed=7,
        replicas=1,
    )
    assert result["coverage_decisions"]
    assert all(value == 0 for value in result["coverage_decisions"].values())
