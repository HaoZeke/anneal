"""The common box entry retains work and coverage from its search engine."""

import anneal
import numpy as np
import pytest
from anneal._core import global_optimize_replicas


def counted_box():
    low = np.array([-2.0, 0.0, -2.0, -2.0, -2.0])
    high = np.array([2.0, 0.0, 2.0, 2.0, 2.0])
    calls = {"evals": 0, "grads": 0}

    def check(x):
        assert x.shape == (5,)
        assert np.all(np.isfinite(x))
        assert np.all(x >= low) and np.all(x <= high)

    def objective(x):
        check(x)
        calls["evals"] += 1
        return float(np.dot(x, x))

    def gradient(x):
        check(x)
        calls["grads"] += 1
        return 2.0 * x

    return objective, gradient, low, high, calls


@pytest.mark.parametrize("with_gradient", [False, True])
@pytest.mark.parametrize("replicas", [1, 4])
def test_minimize_counts_objectives_separately_from_gradients(with_gradient, replicas):
    objective, gradient, low, high, calls = counted_box()
    result = anneal.minimize(
        objective,
        np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
        np.column_stack((low, high)),
        jac=gradient if with_gradient else None,
        budget=256,
        seed=7,
        replicas=replicas,
    )
    assert result.nfev == calls["evals"]
    assert result.njev == calls["grads"]
    assert result.charged == result.nfev + result.njev <= 256
    assert result.nfev > 0
    assert (result.njev > 0) == with_gradient
    assert result.fun == float(np.dot(result.x, result.x))
    assert result.success
    assert "not certified" in result.message


@pytest.mark.parametrize("with_gradient", [False, True])
@pytest.mark.parametrize("replicas", [1, 4])
def test_ensemble_result_retains_work_breakdown(with_gradient, replicas):
    objective, gradient, low, high, calls = counted_box()
    result = anneal.ensemble_optimize(
        objective,
        low,
        high,
        budget=256,
        seed=7,
        grad_fn=gradient if with_gradient else None,
        x0=np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
        replicas=replicas,
    )
    assert result["n_evals"] == calls["evals"]
    assert result["n_grads"] == calls["grads"]
    assert result["charged"] == result["n_evals"] + result["n_grads"] <= 256
    assert result["history_observations"] >= result["history_minima"]
    if replicas == 1 and not with_gradient:
        assert result["hops"] == result["history_observations"] == 0
    else:
        assert result["hops"] > 0


@pytest.mark.parametrize(
    ("with_gradient", "replicas"), [(False, 1), (False, 4), (True, 1)]
)
def test_minimize_retains_engine_diagnostics(with_gradient, replicas):
    objective, gradient, low, high, _ = counted_box()
    x0 = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    result = anneal.minimize(
        objective,
        x0,
        np.column_stack((low, high)),
        jac=gradient if with_gradient else None,
        budget=256,
        seed=7,
        replicas=replicas,
    )
    free = high > low

    def full(z):
        x = x0.copy()
        x[free] = z
        return x

    def free_objective(z):
        return objective(full(z))

    def free_gradient(z):
        return gradient(full(z))[free]

    if replicas == 1:
        engine = anneal.global_optimize(
            free_objective,
            low[free],
            high[free],
            256,
            seed=7,
            grad_fn=free_gradient if with_gradient else None,
            x0=x0[free],
        )
    else:
        engine = global_optimize_replicas(
            free_objective, low[free], high[free], 256, x0[free], 7, replicas, "shared"
        )
    assert result.diagnostics.keys() == engine.keys()
    np.testing.assert_array_equal(result.x, full(engine["best_pos"]))
    np.testing.assert_array_equal(result.diagnostics["best_pos"], result.x)
    assert result.fun == engine["best_val"]
    for key in ("best_val", "n_evals", "n_grads", "arm_pulls", "arm_successes"):
        assert result.diagnostics[key] == engine[key]
    if replicas > 1:
        assert result.nit == result.diagnostics["n_batches"] == engine["n_batches"]
    else:
        assert result.nit == result.charged
