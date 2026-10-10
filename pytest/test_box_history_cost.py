"""History elapsed time and refusals are visible without spending PES calls."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import anneal
import anneal._core as native


@pytest.fixture(scope="module", autouse=True)
def native_provenance(record_testsuite_property):
    """Record the loaded artifact, without inferring its source revision."""
    extension = Path(native.__file__).resolve()
    digest = hashlib.sha256()
    with extension.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    record_testsuite_property("history_cost_native_path", str(extension))
    record_testsuite_property("history_cost_native_sha256", digest.hexdigest())
    record_testsuite_property("history_cost_version", anneal.__version__)
    record_testsuite_property(
        "history_cost_wrapper_path", str(Path(anneal.__file__).resolve())
    )


class _Quadratic:
    def __init__(self):
        self.evaluations = []
        self.gradients = []

    def value(self, x):
        x = np.asarray(x, dtype=float)
        self.evaluations.append(x.copy())
        return float(np.dot(x, x))

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        self.gradients.append(x.copy())
        return 2.0 * x


def _assert_history_cost(result, *, enabled, refusals):
    required = {"history_seconds", "history_refusals"}
    assert required <= result.keys(), f"missing history cost: {required - result.keys()}"
    assert type(result["history_refusals"]) is int
    assert result["history_refusals"] == refusals
    assert math.isfinite(result["history_seconds"])
    if enabled:
        assert result["history_seconds"] > 0.0
    else:
        assert result["history_seconds"] == 0.0


@pytest.mark.parametrize("entry", ["box_ensemble_optimize", "ensemble_optimize"])
@pytest.mark.parametrize("history", ["private", "none"])
@pytest.mark.parametrize("start,refusals", [(0.0, 0), (0.5, 1)])
def test_gradient_history_cost_preserves_exact_pes_charges(
    entry, history, start, refusals
):
    objective = _Quadratic()
    result = getattr(anneal, entry)(
        objective.value,
        [-1.0],
        [1.0],
        budget=2,
        seed=7,
        grad_fn=objective.gradient,
        x0=[start],
        replicas=1,
        history=history,
    )

    assert (len(objective.evaluations), len(objective.gradients)) == (1, 1)
    np.testing.assert_array_equal(objective.evaluations[0], [start])
    np.testing.assert_array_equal(objective.gradients[0], [start])
    np.testing.assert_array_equal(result["best_pos"], [start])
    assert result["best_val"] == start * start
    admitted = int(history != "none" and refusals == 0)
    assert result["history_minima"] == admitted
    if entry == "box_ensemble_optimize":
        assert (result["n_evals"], result["n_grads"]) == (1, 1)
        assert result["history_observations"] == admitted
        assert result["hops"] == 0
    else:
        assert result["charged"] == 2
    _assert_history_cost(
        result,
        enabled=history != "none",
        refusals=refusals if history != "none" else 0,
    )


@pytest.mark.parametrize("history", ["private", "none"])
def test_values_only_hop_history_cost_preserves_exact_pes_charges(history):
    objective = _Quadratic()
    result = anneal.ensemble_optimize(
        objective.value,
        [-1.0],
        [1.0],
        budget=6,
        seed=7,
        x0=[0.0],
        replicas=2,
        history=history,
    )

    assert (len(objective.evaluations), len(objective.gradients)) == (6, 0)
    assert result["charged"] == 6
    np.testing.assert_array_equal(result["best_pos"], [0.0])
    assert result["best_val"] == 0.0
    # Each replica pays one initial value and two finite-difference values.
    np.testing.assert_array_equal(objective.evaluations[0], [0.0])
    np.testing.assert_array_equal(objective.evaluations[1], [0.0])
    np.testing.assert_array_equal(objective.evaluations[3], objective.evaluations[4])
    second_start = objective.evaluations[3][0]
    assert 1e-3 < abs(second_start) < 1.0
    assert result["history_minima"] == int(history != "none")
    _assert_history_cost(
        result, enabled=history != "none", refusals=int(history != "none")
    )


@pytest.mark.parametrize("history", ["shared", "private", "none"])
def test_single_replica_values_portfolio_has_zero_history_cost(history):
    objective = _Quadratic()
    result = anneal.ensemble_optimize(
        objective.value,
        [-1.0],
        [1.0],
        budget=1,
        seed=7,
        replicas=1,
        history=history,
    )

    assert (len(objective.evaluations), len(objective.gradients)) == (1, 0)
    assert result["charged"] == 1
    assert math.isfinite(result["best_val"])
    assert result["history_minima"] == 0
    _assert_history_cost(result, enabled=False, refusals=0)
