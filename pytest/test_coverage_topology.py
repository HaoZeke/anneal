"""Scalar clients select the native coverage graph without derivative callbacks."""

import inspect

import anneal
import numpy as np
import pytest


def require_control(entry):
    parameter = inspect.signature(getattr(anneal, entry)).parameters.get(
        "coverage_neighbors"
    )
    assert parameter is not None, f"{entry} must expose native coverage topology"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == (0 if entry == "global_optimize" else None)


def scalar_run(entry, calls, *, replicas=5, budget=10, **settings):
    require_control(entry)

    def objective(x):
        assert x.shape == (8,)
        assert np.all(np.isfinite(x)) and np.all(np.abs(x) <= 2.0)
        calls.append(x.copy())
        return 1.0

    if entry == "minimize":
        result = anneal.minimize(
            objective,
            np.zeros(8),
            [(-2.0, 2.0)] * 8,
            jac=None,
            replicas=replicas,
            budget=budget,
            seed=17,
            **settings,
        )
        return result.diagnostics
    options = dict(settings)
    if entry == "ensemble_optimize":
        options["x0"] = np.zeros(8)
    return getattr(anneal, entry)(
        objective,
        np.full(8, -2.0),
        np.full(8, 2.0),
        replicas=replicas,
        budget=budget,
        seed=17,
        **options,
    )


@pytest.mark.parametrize(
    "replicas,neighbors,delivered",
    [(2, 1, 2), (3, 1, 6), (5, 0, 20), (5, 1, 10), (5, 2, 20), (5, 100, 20)],
)
def test_scalar_portfolio_delivery_obeys_ring_neighbourhood(replicas, neighbors, delivered):
    calls = []
    result = scalar_run(
        "global_optimize",
        calls,
        replicas=replicas,
        budget=2 * replicas,
        coverage_shared=True,
        coverage_radius=0.8,
        coverage_neighbors=neighbors,
    )
    assert result["n_evals"] == len(calls) == 2 * replicas
    assert result["n_grads"] == 0
    assert result["charged"] == 2 * replicas
    assert result["best_val"] == 1.0
    assert result["coverage_published_samples"] == 2 * replicas
    assert result["coverage_applied_foreign_samples"] == delivered
    assert result["coverage_repelled_proposals"] == replicas


def test_private_control_ignores_neighbourhood():
    traces = []
    for neighbors in [0, 1]:
        calls = []
        result = scalar_run(
            "global_optimize", calls, coverage_shared=False, coverage_neighbors=neighbors
        )
        assert result["n_evals"] == len(calls) == 10
        assert result["n_grads"] == 0
        assert result["coverage_published_samples"] == 0
        assert result["coverage_applied_foreign_samples"] == 0
        assert result["coverage_repelled_proposals"] == 0
        traces.append(sorted(tuple(x) for x in calls))
    assert traces[0] == traces[1]


@pytest.mark.parametrize("entry", ["ensemble_optimize", "minimize"])
def test_scalar_hop_clients_accept_same_neighbourhood_control(entry):
    calls = []
    result = scalar_run(
        entry,
        calls,
        budget=500,
        history="none",
        coverage_shared=True,
        coverage_radius=0.8,
        coverage_neighbors=1,
    )
    assert result["n_evals"] == len(calls) <= 500
    assert result["n_grads"] == 0
    assert result["best_val"] == 1.0
    assert result["coverage_published_samples"] > 0
    assert result["coverage_applied_foreign_samples"] > 0


@pytest.mark.parametrize("entry", ["global_optimize", "ensemble_optimize", "minimize"])
@pytest.mark.parametrize("neighbors", [-1, 1.5])
def test_invalid_neighbourhood_costs_no_callbacks(entry, neighbors):
    require_control(entry)
    calls = []
    with pytest.raises((TypeError, ValueError, OverflowError)):
        scalar_run(entry, calls, coverage_neighbors=neighbors)
    assert not calls
