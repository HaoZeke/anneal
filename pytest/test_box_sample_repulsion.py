"""Bindings expose native sampled-peer separation without counting extra work."""

import anneal
import numpy as np
import pytest


FIELDS = (
    "coverage_published_samples",
    "coverage_applied_foreign_samples",
    "coverage_sample_overlaps",
    "coverage_repelled_proposals",
    "coverage_constrained_repulsions",
)


@pytest.mark.parametrize(
    "driver,with_gradient",
    [
        ("box_ensemble_optimize", True),
        ("ensemble_optimize", True),
        ("ensemble_optimize", False),
    ],
)
@pytest.mark.parametrize("history", ["private", "shared"])
def test_bindings_retain_native_sample_repulsion(driver, with_gradient, history):
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
        budget=512,
        seed=0,
        grad_fn=gradient if with_gradient else None,
        x0=np.array([0.0]),
        replicas=2,
        history=history,
    )
    published, received, overlaps, moved, constrained = (result[field] for field in FIELDS)
    assert overlaps == moved + constrained
    assert result["n_evals"] == calls["evals"]
    assert result["n_grads"] == calls["grads"]
    assert calls["evals"] + calls["grads"] <= 512
    assert result["best_val"] == 0.0
    np.testing.assert_array_equal(result["best_pos"], [0.0])
    if history == "shared":
        assert published > 0
        assert received > 0
        assert moved > 0
    else:
        assert published == received == overlaps == moved == constrained == 0


def test_values_portfolio_does_not_report_sample_communication():
    result = anneal.ensemble_optimize(
        lambda x: float(np.dot(x, x)),
        np.full(5, -2.0),
        np.full(5, 2.0),
        budget=128,
        replicas=1,
    )
    assert all(result[field] == 0 for field in FIELDS)
