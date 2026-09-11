"""Bindings retain native coverage-return actions, separately from delivery."""

import anneal
import numpy as np
import pytest


FIELDS = (
    "coverage_recrossings",
    "coverage_peer_recrossings",
    "coverage_peer_only_recrossings",
    "coverage_escape_updates",
)


@pytest.mark.parametrize(
    "driver,with_gradient",
    [
        ("box_ensemble_optimize", True),
        ("ensemble_optimize", True),
        ("ensemble_optimize", False),
    ],
)
@pytest.mark.parametrize("history", ["none", "shared"])
def test_box_results_expose_native_escape_actions(driver, with_gradient, history):
    calls = {"evals": 0, "grads": 0}

    def objective(x):
        calls["evals"] += 1
        return float(0.5 * np.dot(x, x))

    def gradient(x):
        calls["grads"] += 1
        return x.copy()

    result = getattr(anneal, driver)(
        objective,
        np.array([-1.0]),
        np.array([1.0]),
        budget=1024,
        seed=7,
        grad_fn=gradient if with_gradient else None,
        x0=np.array([0.0]),
        replicas=4,
        history=history,
    )
    recrossings, peers, peer_only, updates = (result[field] for field in FIELDS)
    assert 0 <= peer_only <= peers <= recrossings <= result["hops"]
    assert 0 <= updates <= recrossings
    assert recrossings + result["history_observations"] <= result["hops"] + 4
    assert result["n_evals"] == calls["evals"]
    assert result["n_grads"] == calls["grads"]
    assert calls["evals"] + calls["grads"] <= 1024
    assert result["best_val"] == 0.0
    np.testing.assert_array_equal(result["best_pos"], [0.0])
    if history == "none":
        assert updates > 0
        assert peers == peer_only == result["history_observations"] == 0


def test_values_portfolio_does_not_claim_box_escape_feedback():
    result = anneal.ensemble_optimize(
        lambda x: float(np.dot(x, x)),
        np.full(5, -2.0),
        np.full(5, 2.0),
        budget=128,
        seed=7,
        replicas=1,
    )
    assert all(result[field] == 0 for field in FIELDS)


@pytest.mark.parametrize("replicas", [1, 4])
def test_values_results_expose_native_novelty_actions(replicas):
    result = anneal.ensemble_optimize(
        lambda x: 0.0,
        np.full(8, -1.0),
        np.full(8, 1.0),
        budget=512,
        seed=7,
        replicas=replicas,
        history="none",
    )
    assert (
        0
        <= result["coverage_novelty_updates"]
        <= result["coverage_novel_arrivals"]
        <= result["hops"]
    )
    if replicas == 1:
        assert result["coverage_novel_arrivals"] == 0
    else:
        assert result["coverage_novelty_updates"] > 0
    assert result["history_observations"] == 0
