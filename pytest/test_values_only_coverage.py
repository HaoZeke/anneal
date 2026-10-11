"""Scalar-only parameter fitting can share coverage without minimum certificates."""

import anneal
import numpy as np
import pytest


@pytest.mark.parametrize("shared", [False, True])
def test_parameter_positions_communicate_without_derivatives_or_minima(shared):
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

    if shared:
        assert diagnostics["coverage_published_samples"] > 0
        assert diagnostics["coverage_applied_foreign_samples"] > 0
        assert diagnostics["coverage_repelled_proposals"] > 0
    else:
        assert diagnostics["coverage_published_samples"] == 0
        assert diagnostics["coverage_applied_foreign_samples"] == 0
        assert diagnostics["coverage_repelled_proposals"] == 0


@pytest.mark.parametrize("entry", ["global_optimize", "ensemble_optimize", "minimize"])
@pytest.mark.parametrize("radius", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_coverage_radius_is_rejected_before_evaluation(entry, radius):
    evaluated = []

    def value_only(parameters):
        evaluated.append(parameters.copy())
        return 0.0

    options = dict(budget=32, coverage_shared=True, coverage_radius=radius)
    if entry != "global_optimize":
        options["history"] = "none"
    with pytest.raises(ValueError, match="coverage_radius must be finite and positive"):
        if entry == "minimize":
            anneal.minimize(value_only, [0.0], [(-1.0, 1.0)], **options)
        else:
            getattr(anneal, entry)(value_only, [-1.0], [1.0], **options)
    assert evaluated == []


@pytest.mark.parametrize("entry", ["global_optimize", "ensemble_optimize", "minimize"])
def test_a_coverage_radius_that_is_not_a_number_is_a_type_error(entry):
    evaluated = []

    def value_only(parameters):
        evaluated.append(parameters.copy())
        return 0.0

    with pytest.raises(TypeError, match="coverage_radius must be a number, got 'wide'"):
        if entry == "minimize":
            anneal.minimize(value_only, [0.0], [(-1.0, 1.0)], coverage_radius="wide")
        else:
            run = getattr(anneal, entry)
            run(value_only, [-1.0], [1.0], 32, coverage_radius="wide")
    assert evaluated == []
