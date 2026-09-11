"""Coverage telemetry describes evaluated regions, not certified minima."""

import anneal
import numpy as np
import pytest


@pytest.mark.parametrize(
    "driver,with_gradient",
    [("box_ensemble_optimize", True), ("ensemble_optimize", True), ("ensemble_optimize", False)],
)
@pytest.mark.parametrize("history", ["shared", "private", "none"])
def test_box_coverage_survives_binding_without_certified_minima(driver, with_gradient, history):
    curvature = 1000.0 ** (np.arange(8) / 7.0)
    low, high = np.full(8, -5.12), np.full(8, 5.12)
    calls = {"evals": 0, "grads": 0}

    def objective(x):
        assert x.shape == (8,)
        assert np.all(np.isfinite(x)) and np.all(x >= low) and np.all(x <= high)
        calls["evals"] += 1
        return float(0.5 * np.dot(curvature * x, x))

    def gradient(x):
        calls["grads"] += 1
        return curvature * x

    budget = 256 if with_gradient else 512
    result = getattr(anneal, driver)(
        objective, low, high, budget=budget, seed=0,
        grad_fn=gradient if with_gradient else None,
        x0=np.full(8, 2.5), replicas=4, history=history,
    )
    if driver == "box_ensemble_optimize":
        assert (result["n_evals"], result["n_grads"]) == (calls["evals"], calls["grads"])
    else:
        assert result["charged"] == calls["evals"] + calls["grads"]
    assert calls["evals"] + calls["grads"] <= budget
    assert result["history_minima"] == 0
    assert result["coverage_observations"] > 4
    assert len(result["coverage_regions_per_chain"]) == 4
    assert all(n > 0 for n in result["coverage_regions_per_chain"])
    if history == "shared":
        assert result["coverage_published"] == result["coverage_observations"]
        assert result["coverage_applied_foreign"] > 0
        assert result["coverage_applied_foreign"] <= 3 * result["coverage_observations"]
    else:
        assert result["coverage_published"] == 0
        assert result["coverage_applied_foreign"] == 0
    assert result["coverage_capped_foreign"] >= 0


def test_values_only_single_portfolio_reports_no_coverage_exchange():
    result = anneal.ensemble_optimize(
        lambda x: float(np.dot(x, x)), np.full(3, -2.0), np.full(3, 2.0),
        budget=40, seed=0, replicas=1,
    )
    assert result["coverage_observations"] == 0
    assert result["coverage_published"] == 0
    assert result["coverage_applied_foreign"] == 0
    assert result["coverage_capped_foreign"] == 0
    assert result["coverage_regions_per_chain"] == []
