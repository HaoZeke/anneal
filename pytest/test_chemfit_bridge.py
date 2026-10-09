"""Bounds, x0, and ChemFit-bridge regression tests.

Covers the ChemFit bug report: classical `run` chains escaped
`[low, high]` (unconstrained presets), and there was no way to pass
initial parameters. Uses stub fitters speaking ChemFit 3.1's
ask/tell/finish protocol and ChemFit 4's evaluate/step names, so the suite
runs without chemfit installed.
"""

import numpy as np
import pytest

from anneal import Boltzmann, Fast, Gsa, run, run_qmc
from anneal.chemfit import ChemFitVector, chemfit_box, fit_chemfit


class StubFitter:
    """Minimal fitter with ChemFit's init/ask/tell/finish protocol."""

    def __init__(self, initial_parameters, bounds=None):
        self.initial_parameters = initial_parameters
        self.bounds = bounds or {}
        self.tells = 0
        self.evals = 0
        self.best_loss = None
        self.finished_with = None

    def init(self):
        self.evals = 0

    def ask(self, params):
        self.evals += 1
        loss = float(np.sum((np.asarray(params["positions"]) - 1.0) ** 2))
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
        return loss

    def tell(self, step=None):
        self.tells += 1

    def finish(self, opt_params=None):
        self.finished_with = dict(opt_params)
        return dict(opt_params)


def recording_sphere(record):
    def obj(x):
        record.append(np.asarray(x, dtype=np.float64).copy())
        return float(np.sum(np.asarray(x) ** 2))

    return obj


LOW = -3.0 * np.ones(6)
HIGH = 3.0 * np.ones(6)


@pytest.mark.parametrize(
    "preset",
    [
        Boltzmann(t_init=5.0, sigma=2.0),
        Fast(t_init=3.0, gamma=2.0),
        Gsa(t_init=3.0, q_v=2.62, q_a=1.7),
    ],
)
def test_run_never_evaluates_out_of_box(preset):
    seen = []
    h = run(
        recording_sphere(seen),
        LOW,
        HIGH,
        preset,
        n_epochs=10,
        steps_per_epoch=30,
        seed=0,
    )
    assert len(seen) > 0
    for x in seen:
        assert np.all(x >= LOW - 1e-9) and np.all(x <= HIGH + 1e-9)
    assert np.all(np.asarray(h.best_pos) >= LOW - 1e-9)
    assert np.all(np.asarray(h.best_pos) <= HIGH + 1e-9)


def test_run_honors_x0_start():
    x0 = np.array([0.1, -0.2, 0.3, -0.1, 0.05, -0.05])
    x0_val = float(np.sum(x0**2))
    h = run(
        lambda x: float(np.sum(np.asarray(x) ** 2)),
        LOW,
        HIGH,
        Boltzmann(t_init=5.0, sigma=0.5),
        n_epochs=1,
        steps_per_epoch=1,
        seed=123,
        x0=x0,
    )
    assert h.best_val <= x0_val + 1e-12


def test_run_rejects_bad_x0_length():
    with pytest.raises(ValueError, match="x0"):
        run(
            lambda x: 0.0,
            LOW,
            HIGH,
            Boltzmann(),
            n_epochs=1,
            steps_per_epoch=1,
            seed=0,
            x0=np.ones(3),
        )


def test_run_qmc_honors_bounds_and_x0():
    seen = []
    x0 = np.full(6, 0.25)
    h = run_qmc(
        recording_sphere(seen),
        LOW,
        HIGH,
        Gsa(t_init=1.0, q_v=2.2, q_a=1.5),
        n_starts=4,
        n_epochs=2,
        steps_per_epoch=3,
        seed=7,
        x0=x0,
    )
    assert len(seen) > 0
    for x in seen:
        assert np.all(x >= LOW - 1e-9) and np.all(x <= HIGH + 1e-9)
    assert h.best_val <= float(np.sum(x0**2)) + 1e-12


def test_vector_round_trips_nested_arrays():
    template = {"positions": np.zeros((2, 3)), "scale": 1.5, "cell": {"a": 2.0}}
    vector = ChemFitVector(template)
    assert vector.dim == 2 * 3 + 1 + 1
    params = {"positions": np.arange(6.0).reshape(2, 3), "scale": -0.5, "cell": {"a": 4.0}}
    back = vector.unpack(vector.pack(params))
    assert np.allclose(back["positions"], params["positions"])
    assert back["scale"] == pytest.approx(-0.5)
    assert back["cell"]["a"] == pytest.approx(4.0)


def test_chemfit_box_uses_bounds_then_span():
    fitter = StubFitter(
        {"positions": np.zeros((2, 2)), "free": 5.0},
        {"positions": (-1.0, 1.0)},
    )
    vector = ChemFitVector(dict(fitter.initial_parameters))
    low, high = chemfit_box(fitter, vector, default_span=2.0)
    assert np.all(low[:4] == -1.0) and np.all(high[:4] == 1.0)
    assert low[4] == pytest.approx(3.0) and high[4] == pytest.approx(7.0)


def test_chemfit_box_rejects_empty_bounds():
    fitter = StubFitter({"x": 0.0}, {"x": (1.0, 1.0)})
    vector = ChemFitVector(dict(fitter.initial_parameters))
    with pytest.raises(ValueError, match="empty"):
        chemfit_box(fitter, vector)


def test_fit_chemfit_classical_uses_protocol():
    rng = np.random.default_rng(0)
    fitter = StubFitter({"positions": rng.uniform(-3, 3, size=(4, 3))})
    out = fit_chemfit(fitter, budget=600, method="boltzmann", seed=0)
    positions = np.asarray(out["positions"])
    assert positions.shape == (4, 3)
    assert np.all(positions >= -3.0 - 1e-9) and np.all(positions <= 3.0 + 1e-9)
    assert fitter.evals > 0
    assert fitter.tells > 0
    assert fitter.finished_with is not None


def test_fit_chemfit_portfolio_beats_start():
    rng = np.random.default_rng(0)
    init = {"positions": rng.uniform(-3, 3, size=(4, 3))}
    start_loss = float(np.sum((init["positions"] - 1.0) ** 2))
    fitter = StubFitter(dict(init))
    out = fit_chemfit(fitter, budget=800, method="portfolio", seed=0)
    end_loss = float(np.sum((np.asarray(out["positions"]) - 1.0) ** 2))
    assert end_loss < start_loss


def test_fit_chemfit_rejects_bad_args():
    fitter = StubFitter({"positions": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="budget"):
        fit_chemfit(fitter, budget=0)
    with pytest.raises(ValueError, match="unknown method"):
        fit_chemfit(fitter, budget=10, method="newton")


class StepFitter:
    """ChemFit 4 lifecycle: ``evaluate`` and ``step``, no ``ask``/``tell``."""

    def __init__(self, initial_parameters, bounds=None):
        self.initial_parameters = initial_parameters
        self.bounds = bounds or {}
        self.seen = []
        self.steps = 0

    def init(self):
        self.seen.clear()

    def evaluate(self, params):
        positions = np.asarray(params["positions"], dtype=np.float64)
        self.seen.append(positions.copy())
        return float(np.sum((positions - 1.0) ** 2))

    def step(self, step=None):
        self.steps += 1

    def finish(self, opt_params=None):
        return dict(opt_params)


@pytest.mark.parametrize("method", ["portfolio", "boltzmann"])
def test_fit_chemfit_starts_at_the_initial_parameters_of_an_evaluate_step_fitter(method):
    rng = np.random.default_rng(0)
    start = rng.uniform(-3, 3, size=(4, 3))
    fitter = StepFitter({"positions": start}, {"positions": (-3.0, 3.0)})
    out = fit_chemfit(fitter, budget=400, method=method, seed=0)
    seen = np.array(fitter.seen)
    assert np.allclose(seen[0], start)
    assert np.all(seen >= -3.0) and np.all(seen <= 3.0)
    assert fitter.steps > 0
    end = float(np.sum((np.asarray(out["positions"]) - 1.0) ** 2))
    assert end < float(np.sum((start - 1.0) ** 2))


def test_run_fitter_translates_its_method_names_for_a_native_fit_anneal():
    from anneal.chemfit import run_fitter

    class NativeFitter:
        def __init__(self):
            self.calls = []

        def fit_anneal(self, **kwargs):
            self.calls.append(kwargs)
            return {}

    fitter = NativeFitter()
    run_fitter(fitter, 10)
    run_fitter(fitter, 10, method="sa")
    run_fitter(fitter, 10, method="gsa")
    assert [call["method"] for call in fitter.calls] == ["portfolio", "boltzmann", "gsa"]


def test_run_fitter_passes_preset_and_keywords_down_the_fallback_path():
    from anneal import Boltzmann, Gsa
    from anneal.chemfit import run_fitter

    def fitter():
        return StepFitter({"positions": np.zeros((2, 3)) + 0.5}, {"positions": (-3.0, 3.0)})

    f = fitter()
    run_fitter(f, 50, method="boltzmann", preset=Boltzmann(t_init=1.0, sigma=0.05), steps_per_epoch=7)
    assert len(f.seen) == 50
    with pytest.raises(ValueError, match="does not match driver"):
        run_fitter(fitter(), 50, method="boltzmann", preset=Gsa())
    with pytest.raises(ValueError, match="does not take preset"):
        run_fitter(fitter(), 50, preset=Boltzmann())
    with pytest.raises(ValueError, match="does not take steps_per_epoch"):
        run_fitter(fitter(), 50, steps_per_epoch=7)
    with pytest.raises(TypeError, match="gradient"):
        run_fitter(fitter(), 50, gradient=lambda p: p)
    with pytest.raises(TypeError, match="stepz_per_epoch"):
        run_fitter(fitter(), 50, method="sa", stepz_per_epoch=7)
