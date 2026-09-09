"""Communicating box hops share a Euclidean history, not a cluster move."""

import math

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")


def _two_well():
    a = np.array([2.0, 2.0])
    b = np.array([-2.0, -2.0])

    def fn(x):
        x = np.asarray(x, dtype=float)
        ea = float(np.dot(x - a, x - a))
        eb = float(np.dot(x - b, x - b)) - 0.5
        return min(ea, eb)

    def grad(x):
        x = np.asarray(x, dtype=float)
        da = x - a
        db = x - b
        ea = float(np.dot(da, da))
        eb = float(np.dot(db, db)) - 0.5
        return 2.0 * da if ea < eb else 2.0 * db

    return fn, grad, np.full(2, -5.0), np.full(2, 5.0)


def test_box_ensemble_respects_budget_and_box():
    fn, grad, low, high = _two_well()
    out = anneal.box_ensemble_optimize(
        fn, low, high, budget=200, seed=7, grad_fn=grad, replicas=4
    )
    assert out["n_evals"] + out["n_grads"] <= 200
    assert math.isfinite(out["best_val"])
    assert np.all(out["best_pos"] >= low - 1e-8)
    assert np.all(out["best_pos"] <= high + 1e-8)
    assert out["hops"] > 0
    assert out["history_minima"] >= 1


def test_box_ensemble_rejects_cluster_sized_zero_replicas():
    fn, _, low, high = _two_well()
    with pytest.raises(ValueError, match="replicas"):
        anneal.box_ensemble_optimize(fn, low, high, budget=20, replicas=0)
