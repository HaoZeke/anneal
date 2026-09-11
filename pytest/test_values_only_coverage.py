"""Scalar-only parameter fitting can share coverage without minimum certificates."""

import anneal
import numpy as np
import pytest


@pytest.mark.parametrize("entry", ["ensemble_optimize", "minimize"])
@pytest.mark.parametrize("shared", [False, True])
def test_parameter_positions_communicate_without_derivatives_or_minima(entry, shared):
    low = np.array([-2.0, 0.0, 100.0, -1.0, 0.001])
    high = np.array([2.0, 0.0, 120.0, 1.0, 0.1])
    start = (low + high) / 2.0
    evaluated = []

    def value_only(parameters):
        evaluated.append(np.array(parameters, copy=True))
        return 0.0

    options = dict(
        budget=512,
        replicas=4,
        seed=0,
        history="none",
        coverage_shared=shared,
        coverage_radius=1.0,
    )
    if entry == "minimize":
        result = anneal.minimize(
            value_only, start, np.column_stack((low, high)), jac=None, **options
        )
        diagnostics = result.diagnostics
        assert result.nfev == len(evaluated)
        assert result.njev == 0
    else:
        diagnostics = anneal.ensemble_optimize(
            value_only, low, high, x0=start, grad_fn=None, **options
        )

    points = np.asarray(evaluated)
    assert points.shape == (diagnostics["n_evals"], 5)
    assert np.all(np.isfinite(points))
    assert np.all(points >= low) and np.all(points <= high)
    np.testing.assert_array_equal(points[:, 1], 0.0)
    assert 0 < len(points) == diagnostics["charged"] <= 512
    assert diagnostics["n_grads"] == 0
    assert diagnostics["history_minima"] == diagnostics["history_observations"] == 0
    assert diagnostics["history_seconds"] == 0.0
    assert diagnostics["hops"] >= 4
    assert diagnostics["best_val"] == 0.0
    np.testing.assert_array_equal(diagnostics["best_pos"], start)

    free = high > low
    offsets = np.max(
        np.abs((points[:, free] - start[free]) / (high - low)[free]), axis=1
    )
    assert np.count_nonzero((offsets > 0.0) & (offsets <= 1e-5)) == 0
    if shared:
        assert diagnostics["coverage_published_samples"] > 0
        assert diagnostics["coverage_applied_foreign_samples"] > 0
        assert diagnostics["coverage_repelled_proposals"] > 0
    else:
        assert diagnostics["coverage_published_samples"] == 0
        assert diagnostics["coverage_applied_foreign_samples"] == 0
        assert diagnostics["coverage_repelled_proposals"] == 0


@pytest.mark.parametrize("entry", ["ensemble_optimize", "minimize"])
@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_coverage_radius_is_rejected_before_evaluation(entry, radius):
    evaluated = []

    def value_only(parameters):
        evaluated.append(parameters.copy())
        return 0.0

    options = dict(
        budget=32, history="none", coverage_shared=True, coverage_radius=radius
    )
    with pytest.raises(ValueError, match="coverage radius"):
        if entry == "minimize":
            anneal.minimize(value_only, [0.0], [(-1.0, 1.0)], **options)
        else:
            anneal.ensemble_optimize(value_only, [-1.0], [1.0], **options)
    assert evaluated == []
