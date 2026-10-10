"""anneal.minimize is the SciPy/ChemFit box entry."""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")


def test_minimize_sphere_with_jac():
    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    def jac(x):
        return 2.0 * np.asarray(x, dtype=float)

    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
    res = anneal.minimize(fn, np.ones(3), bounds, jac=jac, budget=64, seed=3)
    assert np.isfinite(res.fun)
    assert res.nfev > 0
    assert res.nfev <= 64
    assert res.x.shape == (3,)
    assert np.all(res.x >= -2.0 - 1e-8)
    assert np.all(res.x <= 2.0 + 1e-8)
    assert res.success


def test_minimize_rejects_1d_bounds():
    def fn(x):
        return float(np.dot(x, x))

    with pytest.raises(ValueError, match="pairs"):
        anneal.minimize(fn, np.ones(2), np.array([-1.0, 1.0]), budget=16)


def test_minimize_jsonl_store_round_trips(tmp_path):
    def fn(x):
        x = np.asarray(x, dtype=float)
        return float(np.dot(x, x))

    store = tmp_path / "anneal_params.jsonl"
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    first = anneal.minimize(
        fn, np.ones(2), bounds, budget=32, seed=3, replicas=2, store=store
    )
    assert store.is_file()
    second = anneal.minimize(
        fn, np.full(2, 1.5), bounds, budget=32, seed=4, replicas=2, store=store
    )
    assert second.fun <= first.fun + 1e-12
    rows = anneal.JsonlParameterStore(store).observations()
    assert len(rows) >= 2
