"""ChemFit session adapter: nested parameters, a hard box, and a supplied start."""

import numpy as np
import pytest

from anneal import Boltzmann
from anneal.chemfit import (
    flatten_params,
    resolve_bounds,
    run_benchmark,
    unflatten_params,
)


def test_positions_roundtrip_keeps_atom_shape():
    positions = np.arange(12, dtype=np.float64).reshape(4, 3)
    flat, spec = flatten_params({"positions": positions, "scale": 1.5})
    assert flat.shape == (13,)
    rebuilt = unflatten_params(flat, spec)
    assert rebuilt["positions"].shape == (4, 3)
    assert rebuilt["positions"] == pytest.approx(positions)
    assert rebuilt["scale"] == pytest.approx(1.5)


def test_scalar_bounds_broadcast_and_mirrored_bounds_match_the_fitter():
    initial = {"positions": np.zeros((2, 3))}
    low, high = resolve_bounds(initial, low=-3.0, high=3.0)
    assert low.shape == (6,)
    assert np.all(low == -3.0) and np.all(high == 3.0)

    mirrored = {
        "positions": (np.full((2, 3), -1.0), np.full((2, 3), 2.0)),
    }
    low, high = resolve_bounds(initial, fitter_bounds=mirrored)
    assert np.all(low == -1.0)
    assert np.all(high == 2.0)


class _Session:
    """Current ChemFit session: init / ask / tell / finish."""

    def __init__(self, bounds):
        self.bounds = bounds
        self.seen = []
        self.ready = False

    def init(self):
        self.ready = True

    def ask(self, params):
        assert self.ready
        pos = np.asarray(params["positions"], dtype=np.float64)
        self.seen.append(pos.copy())
        return float(np.sum(pos * pos))

    def tell(self):
        return None

    def finish(self, opt_params=None):
        return opt_params


class _Review:
    """Review-response names: evaluate / step / finish()."""

    def __init__(self):
        self.seen = []

    def evaluate(self, params):
        pos = np.asarray(params["positions"], dtype=np.float64)
        self.seen.append(pos.reshape(-1).copy())
        return float(np.sum(pos * pos))

    def step(self):
        return None

    def finish(self):
        return {"n": len(self.seen)}


def test_portfolio_starts_at_initial_params_and_stays_in_the_box():
    positions = np.array([[0.2, -0.4, 0.1], [1.5, -1.0, 0.0]])
    fitter = _Session({"positions": (-3.0, 3.0)})
    out = run_benchmark(
        {
            "fitter": fitter,
            "budget": 40,
            "initial_params": {"positions": positions},
        },
        method="portfolio",
        seed=1,
        low=-3.0,
        high=3.0,
    )
    assert fitter.seen
    assert fitter.seen[0] == pytest.approx(positions)
    for trial in fitter.seen:
        assert trial.shape == (2, 3)
        assert np.all(trial >= -3.0 - 1e-9)
        assert np.all(trial <= 3.0 + 1e-9)
    assert np.asarray(out["positions"]).shape == (2, 3)


def test_boltzmann_clips_the_start_and_never_leaves_the_box():
    fitter = _Review()
    start = np.array([4.0, -0.2, 0.5, -9.0, 0.0, 1.0])
    run_benchmark(
        {
            "fitter": fitter,
            "budget": 30,
            "initial_params": {"positions": start.reshape(2, 3)},
        },
        method="boltzmann",
        seed=3,
        steps_per_epoch=10,
        low=-1.0,
        high=1.0,
        preset=Boltzmann(t_init=2.0, sigma=4.0),
    )
    assert fitter.seen
    assert fitter.seen[0] == pytest.approx(np.array([1.0, -0.2, 0.5, -1.0, 0.0, 1.0]))
    for trial in fitter.seen:
        assert np.all(trial >= -1.0 - 1e-9)
        assert np.all(trial <= 1.0 + 1e-9)
