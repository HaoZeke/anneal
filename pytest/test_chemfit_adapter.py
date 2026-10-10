"""Tests for the ChemFit adapter (``anneal.chemfit``).

ChemFit itself is not a test dependency: a duck-typed stub implements the
user-driven ``init``/``ask``/``tell``/``finish`` protocol with ChemFit's
best-seen tracking semantics.
"""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from anneal.chemfit import (  # noqa: E402
    fit_anneal,
    flatten_parameters,
    unflatten_parameters,
)


class StubFitter:
    """Minimal ChemFit ``Fitter`` double over nested scalar+array params."""

    def __init__(self, initial_params, bounds=None, target=None):
        self.initial_parameters = initial_params
        self.bounds = dict(bounds or {})
        self.target = (
            np.zeros(3) if target is None else np.asarray(target, dtype=np.float64)
        )
        self.tells = 0
        self.finished_with = None

    def init(self):
        self._best_loss = None
        self._best_params = None

    def ask(self, params):
        positions = np.asarray(params["positions"], dtype=np.float64)
        loss = float(np.sum((positions - self.target) ** 2) + (params["eps"] - 1.0) ** 2)
        if self._best_loss is None or loss < self._best_loss:
            self._best_loss = loss
            self._best_params = {
                "positions": positions.copy(),
                "eps": params["eps"],
            }
        return loss

    def tell(self, step=None):
        self.tells += 1

    def finish(self, opt_params=None):
        self.finished_with = (
            dict(opt_params) if opt_params is not None else dict(self._best_params)
        )
        return self.finished_with


def _stub():
    return StubFitter(
        {"positions": np.array([[2.5, 0.0, -1.0]]), "eps": 2.0},
        bounds={"eps": (0.5, 3.0)},
    )


def test_flatten_round_trips_nested_params():
    params = {
        "positions": np.arange(6, dtype=float).reshape(2, 3),
        "cell": {"a": 1.5, "b": 2.0},
        "eps": 0.7,
    }
    vec, spec = flatten_parameters(params)
    assert vec.shape == (9,)
    back = unflatten_parameters(vec, spec, params)
    assert back["positions"].shape == (2, 3)
    assert np.allclose(back["positions"], params["positions"])
    assert back["cell"] == {"a": 1.5, "b": 2.0}
    assert back["eps"] == pytest.approx(0.7)
    # The template is never mutated.
    assert np.allclose(params["positions"], np.arange(6).reshape(2, 3))


def test_flatten_rejects_non_numeric_leaves():
    with pytest.raises(ValueError, match="not real-numeric"):
        flatten_parameters({"tag": "lj_1"})


def test_unflatten_rejects_length_mismatch():
    _, spec = flatten_parameters({"x": 1.0, "y": 2.0})
    with pytest.raises(ValueError, match="spec needs 2"):
        unflatten_parameters(np.zeros(3), spec, {"x": 1.0, "y": 2.0})


def test_fit_anneal_portfolio_finds_minimum_in_bounds():
    fitter = _stub()
    out = fit_anneal(fitter, 500, driver="portfolio", seed=0)
    positions = np.asarray(out["positions"])
    assert positions.shape == (1, 3)
    assert np.all(positions >= -3.0) and np.all(positions <= 3.0)
    assert 0.5 <= out["eps"] <= 3.0
    loss = float(np.sum(positions**2) + (out["eps"] - 1.0) ** 2)
    assert loss == pytest.approx(0.0, abs=1e-6)
    assert fitter.tells > 0
    assert fitter.finished_with is not None


def test_fit_anneal_classical_drivers_respect_bounds_and_budget():
    for driver in ("boltzmann", "fast", "gsa"):
        fitter = _stub()
        out = fit_anneal(
            fitter,
            400,
            driver=driver,
            seed=0,
            low=-3.0 * np.ones(4),
            high=3.0 * np.ones(4),
        )
        positions = np.asarray(out["positions"])
        assert np.all(positions >= -3.0) and np.all(positions <= 3.0)
        assert 0.5 <= out["eps"] <= 3.0
        # The budget counts the seeded start.
        assert fitter.tells <= 400
        start_loss = float(np.sum(np.array([[2.5, 0.0, -1.0]]) ** 2) + 1.0)
        got_loss = float(np.sum(positions**2) + (out["eps"] - 1.0) ** 2)
        assert got_loss < start_loss


def test_fit_anneal_defaults_to_fitter_initial_params():
    fitter = _stub()
    out = fit_anneal(fitter, 100, driver="boltzmann", seed=0)
    assert np.asarray(out["positions"]).shape == (1, 3)


def test_fit_anneal_rejects_bad_drivers_and_budgets():
    with pytest.raises(ValueError, match="driver must be"):
        fit_anneal(_stub(), 100, driver="simplex")
    with pytest.raises(ValueError, match="budget must be positive"):
        fit_anneal(_stub(), 0)


def test_fit_anneal_rejects_mismatched_explicit_bounds():
    with pytest.raises(ValueError, match="low/high have lengths"):
        fit_anneal(_stub(), 100, low=np.zeros(2), high=np.ones(4))
    with pytest.raises(ValueError, match="given together"):
        fit_anneal(_stub(), 100, low=np.zeros(4))


def test_fit_anneal_accepts_dict_and_vector_x0():
    fitter = _stub()
    dict_x0 = {"positions": np.zeros((1, 3)), "eps": 1.0}
    out = fit_anneal(fitter, 100, driver="boltzmann", seed=0, x0=dict_x0)
    assert np.asarray(out["positions"]).shape == (1, 3)
    out = fit_anneal(
        _stub(), 100, driver="boltzmann", seed=0, x0=np.zeros(4)
    )
    assert np.asarray(out["positions"]).shape == (1, 3)
    with pytest.raises(ValueError, match="must mirror"):
        fit_anneal(_stub(), 100, x0={"other": 1.0})
    with pytest.raises(ValueError, match="has length"):
        fit_anneal(_stub(), 100, x0=np.zeros(3))
