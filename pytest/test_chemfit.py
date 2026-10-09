"""Tests for anneal.chemfit, which fits a ChemFit Fitter with anneal's drivers."""

import copy
import functools
import inspect
import math
import os
import subprocess
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import anneal
from anneal.chemfit import METHODS, fit, objective

chemfit = pytest.importorskip("chemfit")
if not hasattr(chemfit.Fitter, "step"):
    pytest.skip("ChemFit without the Fitter session API", allow_module_level=True)


def lj_energy(positions):
    """Lennard-Jones energy of ``(n, 3)`` atomic positions."""
    pos = np.asarray(positions, dtype=np.float64)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = (d**2).sum(-1)[np.triu_indices(len(pos), 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv6 = 1.0 / r2**3
        return float(np.sum(4.0 * (inv6 * inv6 - inv6)))


def quadratic(params):
    total = 0.0
    for value in params.values():
        if isinstance(value, dict):
            total += quadratic(value)
        else:
            total += float(np.sum((np.asarray(value, dtype=np.float64) - 0.3) ** 2))
    return total


class Recorder:
    """ChemFit objective that keeps every parameter tree it evaluates."""

    def __init__(self, loss=quadratic):
        self.loss = loss
        self.seen = []

    def __call__(self, params):
        self.seen.append(copy.deepcopy(params))
        return self.loss(params)


def make_fitter(initial, bounds=None, loss=quadratic, **kwargs):
    recorder = Recorder(loss)
    fitter = chemfit.Fitter(recorder, initial_params=initial, bounds=bounds, **kwargs)
    return fitter, recorder


class Proposals:
    """Objective wrapper that keeps every candidate a driver proposes, including
    the ones past the budget that never reach ChemFit."""

    def __init__(self, view):
        self.view = view
        self.seen = []

    def __call__(self, x):
        self.seen.append(np.array(x))
        return self.view(x)

    def eval_batch(self, X):
        self.seen.extend(np.array(X))
        return self.view.eval_batch(X)


def spy_on_driver(monkeypatch, name):
    """Record the options ``fit`` gives the driver ``name`` and what it proposes."""
    calls = []
    driver = getattr(anneal, name)

    @functools.wraps(driver)
    def spy(obj_fn, low, high, **kwargs):
        proposals = Proposals(obj_fn)
        calls.append((kwargs, proposals))
        return driver(proposals, low, high, **kwargs)

    monkeypatch.setattr(anneal, name, spy)
    return calls


def spy_on_evaluate(fitter, monkeypatch):
    """Record the size of every ``fitter.evaluate`` call, ``None`` for a mapping."""
    sizes = []
    evaluate = fitter.evaluate

    def spy(params, *args, **kwargs):
        sizes.append(len(params) if isinstance(params, list) else None)
        return evaluate(params, *args, **kwargs)

    monkeypatch.setattr(fitter, "evaluate", spy)
    return sizes


def assert_same_tree(got, want):
    assert got.keys() == want.keys()
    for key, value in want.items():
        if isinstance(value, dict):
            assert_same_tree(got[key], value)
        elif isinstance(value, np.ndarray):
            assert isinstance(got[key], np.ndarray)
            assert got[key].dtype == value.dtype
            assert np.array_equal(got[key], value)
        else:
            assert got[key] == value


INITIAL = {
    "positions": np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(4, 3),
    "lj": {"epsilon": 0.8, "sigma": 1.1},
}
BOUNDS = {
    "positions": (-2.0, 2.0),
    "lj": {"epsilon": (0.1, 2.0), "sigma": (0.5, 1.5)},
}
RUN_METHODS = ["boltzmann", "fast", "gsa"]
DRIVERS = {
    "portfolio": "global_optimize",
    "boltzmann": "run",
    "fast": "run",
    "gsa": "run",
    "qmc": "run_qmc",
    "dmc": "dmc_population_optimize",
}


def test_scalar_leaves_flatten_in_tree_order():
    initial = {"lj": {"epsilon": 0.8, "sigma": 1.1}, "shift": -0.2}
    bounds = {
        "lj": {"epsilon": (0.1, 2.0), "sigma": (0.5, 1.5)},
        "shift": (-1.0, 1.0),
    }
    fitter, _ = make_fitter(initial, bounds)

    view = objective(fitter)

    assert view.x0.tolist() == [0.8, 1.1, -0.2]
    assert view.low.tolist() == [0.1, 0.5, -1.0]
    assert view.high.tolist() == [2.0, 1.5, 1.0]
    assert_same_tree(view.unflatten(view.x0), initial)
    params = view.unflatten([1.5, 0.75, 0.25])
    assert params == {"lj": {"epsilon": 1.5, "sigma": 0.75}, "shift": 0.25}
    assert all(type(v) is float for v in (*params["lj"].values(), params["shift"]))
    assert view.flatten(params).tolist() == [1.5, 0.75, 0.25]


def test_array_leaves_keep_shape_and_dtype():
    initial = {
        "positions": np.arange(12, dtype=np.float32).reshape(4, 3) / 8,
        "charges": np.array([0.1, -0.1, 0.2, -0.2]),
        "scale": np.float64(2.0),
    }
    bounds = {"positions": (-3.0, 3.0), "charges": (-1.0, 1.0), "scale": (0.0, 5.0)}
    fitter, _ = make_fitter(initial, bounds)
    view = objective(fitter)
    x = np.random.default_rng(1).uniform(view.low, view.high)

    params = view.unflatten(x)

    assert view.x0.shape == (17,)
    assert np.array_equal(view.x0[:12], initial["positions"].ravel())
    assert_same_tree(view.unflatten(view.x0), initial)
    assert params["positions"].shape == (4, 3)
    assert params["positions"].dtype == np.float32
    assert np.array_equal(params["positions"], x[:12].reshape(4, 3).astype(np.float32))
    assert params["charges"].dtype == np.float64
    assert np.array_equal(params["charges"], x[12:16])
    assert params["scale"] == x[16]
    assert np.array_equal(view.flatten(params)[12:], x[12:])


def test_scalar_bounds_broadcast_over_an_array_leaf():
    fitter, _ = make_fitter({"positions": np.zeros((5, 3))}, {"positions": (-3.0, 3.0)})

    view = objective(fitter)

    assert np.array_equal(view.low, np.full(15, -3.0))
    assert np.array_equal(view.high, np.full(15, 3.0))


def test_per_element_bounds():
    low = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])
    fitter, _ = make_fitter(
        {"positions": np.zeros((2, 3)), "axes": np.zeros((2, 3))},
        {"positions": (low, -low), "axes": ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])},
    )

    view = objective(fitter)

    assert np.array_equal(view.low[:6], low.ravel())
    assert np.array_equal(view.high[:6], -low.ravel())
    assert view.low[6:].tolist() == [-1.0, -2.0, -3.0, -1.0, -2.0, -3.0]
    assert view.high[6:].tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]


def test_bounds_override_fitter_bounds_leaf_by_leaf():
    fitter, _ = make_fitter(
        {"a": 0.5, "b": {"c": 0.5}}, {"a": (0.0, 1.0), "b": {"c": (0.0, 1.0)}}
    )

    view = objective(fitter, bounds={"b": {"c": (-2.0, 2.0)}})

    assert view.low.tolist() == [0.0, -2.0]
    assert view.high.tolist() == [1.0, 2.0]
    assert fitter.bounds == {"a": (0.0, 1.0), "b": {"c": (0.0, 1.0)}}


@pytest.mark.parametrize("method", METHODS)
def test_first_evaluation_is_the_start(method):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    fit(fitter, 60, method=method, seed=2)

    assert_same_tree(recorder.seen[0], INITIAL)


@pytest.mark.parametrize(("method", "driver"), DRIVERS.items())
def test_each_method_runs_its_driver_from_the_start(method, driver, monkeypatch):
    takes_x0 = "x0" in inspect.signature(getattr(anneal, driver)).parameters
    calls = spy_on_driver(monkeypatch, driver)
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    view = objective(fitter)

    fit(fitter, 60, method=method, seed=2)

    ((kwargs, proposals),) = calls
    assert kwargs["seed"] == 2
    assert ("x0" in kwargs) is takes_x0
    if takes_x0:
        assert np.array_equal(kwargs["x0"], view.x0)
        # dmc reflects its start into the box, which can move it by an ulp.
        assert np.allclose(proposals.seen[0], view.x0, rtol=0.0, atol=1e-12)
    if driver in ("run", "run_qmc"):
        assert np.array_equal(proposals.seen[0], view.x0)
        starts = [np.array_equal(view.flatten(p), view.x0) for p in recorder.seen]
        assert sum(starts) == 1


@pytest.mark.parametrize("method", METHODS)
def test_budget_is_a_hard_cap(method):
    budget = 157
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    options = {"steps_per_epoch": 50} if method in RUN_METHODS else {}

    fit(fitter, budget, method=method, seed=4, **options)

    assert len(recorder.seen) <= budget
    assert sum(ctx.n_evals for ctx in fitter.contexts) == len(recorder.seen)
    if method in (*RUN_METHODS, "dmc"):
        assert len(recorder.seen) == budget


def test_proposals_past_the_budget_never_reach_chemfit(monkeypatch):
    returned = []

    def greedy(obj_fn, low, high, budget, seed=0):
        rng = np.random.default_rng(seed)
        returned.extend(obj_fn(rng.uniform(low, high)) for _ in range(budget - 5))
        returned.extend(obj_fn.eval_batch(rng.uniform(low, high, (10, low.size))))
        return {}

    monkeypatch.setattr(anneal, "global_optimize", greedy)
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    sizes = spy_on_evaluate(fitter, monkeypatch)

    fit(fitter, 25, batch_size=4)

    assert len(returned) == 30
    assert len(recorder.seen) == 25
    assert sizes == [None] * 21 + [4]
    assert all(math.isfinite(v) for v in returned[:24])
    assert all(v == math.inf for v in returned[24:])


@pytest.mark.parametrize("method", METHODS)
def test_a_budget_of_one_evaluates_only_the_start(method):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    best = fit(fitter, 1, method=method)

    assert len(recorder.seen) == 1
    assert_same_tree(best, INITIAL)


def test_view_budget_returns_inf_without_evaluating():
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    view = objective(fitter)
    view.budget = 3
    fitter.init(batch_size=2)

    losses = [view(view.x0) for _ in range(4)]
    batch = view.eval_batch(np.tile(view.x0, (2, 1)))

    assert all(math.isfinite(v) for v in losses[:3])
    assert losses[3] == math.inf
    assert batch.tolist() == [math.inf, math.inf]
    assert view.n_evals == len(recorder.seen) == 3


def test_callbacks_fire_through_step():
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    steps = []
    fitter.register_callback(lambda step, contexts: steps.append(step), n_steps=10)

    fit(fitter, 35, method="boltzmann", steps_per_epoch=10)

    assert len(recorder.seen) == 35
    assert steps == [10, 20, 30, 35]


def test_eval_batch_sends_one_evaluate_call_per_batch(monkeypatch):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    view = objective(fitter)
    sizes = spy_on_evaluate(fitter, monkeypatch)
    steps = []
    fitter.register_callback(lambda step, contexts: steps.append(step), n_steps=1)
    rows = np.random.default_rng(3).uniform(view.low, view.high, size=(7, view.x0.size))
    fitter.init(batch_size=3)

    losses = view.eval_batch(rows)

    assert sizes == [3, 3, 1]
    assert steps == [1, 2, 3]
    assert np.allclose(losses, [quadratic(view.unflatten(row)) for row in rows])
    assert len(recorder.seen) == view.n_evals == 7


def test_eval_batch_runs_a_batch_concurrently():
    barrier = threading.Barrier(4, timeout=10)

    @chemfit.objective()
    def meet(params):
        barrier.wait()
        return float(params["x"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        schedule = chemfit.ExecutorTreeScheduler(executor=pool).prepare(meet)
        fitter = chemfit.Fitter(
            schedule, initial_params={"x": 0.5}, bounds={"x": (0, 1)}
        )
        view = objective(fitter)
        fitter.init(batch_size=4)
        losses = view.eval_batch([[0.1], [0.2], [0.3], [0.4]])
        schedule.close()

    assert losses.tolist() == [0.1, 0.2, 0.3, 0.4]


def test_dmc_sends_walker_batches_through_one_evaluate_call(monkeypatch):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    sizes = spy_on_evaluate(fitter, monkeypatch)

    fit(fitter, 200, method="dmc", batch_size=4, seed=1)

    batches = [size for size in sizes if size is not None]
    assert max(batches) == 4
    assert sum(size or 1 for size in sizes) == len(recorder.seen) == 200


def test_portfolio_passes_x0_when_global_optimize_takes_it(monkeypatch):
    received = {}

    def global_optimize(obj_fn, low, high, budget, seed=0, x0=None):
        received["x0"] = x0
        obj_fn(x0)
        obj_fn((low + high) / 2)
        return {}

    monkeypatch.setattr(anneal, "global_optimize", global_optimize)
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    fit(fitter, 10)

    assert np.array_equal(received["x0"], objective(fitter).x0)
    assert len(recorder.seen) == 2
    assert_same_tree(recorder.seen[0], INITIAL)


def test_run_options_reach_the_preset_and_the_driver(monkeypatch):
    calls = spy_on_driver(monkeypatch, "run")
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    fit(fitter, 41, method="boltzmann", t_init=2.0, sigma=0.05, steps_per_epoch=8)

    ((kwargs, proposals),) = calls
    assert repr(kwargs["preset"]) == "Boltzmann(t_init=2.0, sigma=0.05)"
    assert (kwargs["n_epochs"], kwargs["steps_per_epoch"]) == (5, 8)
    assert len(proposals.seen) == len(recorder.seen) == 41


def test_qmc_takes_a_preset_and_splits_the_budget_over_its_starts(monkeypatch):
    calls = spy_on_driver(monkeypatch, "run_qmc")
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    preset = anneal.Gsa(t_init=3.0)

    fit(fitter, 100, method="qmc", preset=preset, n_starts=4, steps_per_epoch=10)

    ((kwargs, proposals),) = calls
    assert kwargs["preset"] is preset
    assert kwargs["n_starts"] == 4
    assert 1 + kwargs["n_epochs"] * kwargs["steps_per_epoch"] == 100 // 4
    assert len(proposals.seen) == len(recorder.seen) == 100


@pytest.mark.parametrize(
    ("method", "options"),
    [
        ("portfolio", {"policy": "auto"}),
        ("dmc", {"target_n": 8, "steps_per_control": 2}),
    ],
)
def test_portfolio_and_dmc_forward_their_options(method, options, monkeypatch):
    calls = spy_on_driver(monkeypatch, DRIVERS[method])
    fitter, _ = make_fitter(INITIAL, BOUNDS)

    fit(fitter, 50, method=method, **options)

    ((kwargs, _),) = calls
    assert kwargs["budget"] == 50
    assert {key: kwargs[key] for key in options} == options


def test_fit_is_reproducible_for_a_seed():
    runs = []
    for _ in range(2):
        fitter, recorder = make_fitter(INITIAL, BOUNDS)
        best = fit(fitter, 80, method="fast", seed=11)
        runs.append((best, [p["positions"] for p in recorder.seen]))

    assert_same_tree(runs[0][0], runs[1][0])
    assert all(
        np.array_equal(a, b) for a, b in zip(runs[0][1], runs[1][1], strict=True)
    )


@pytest.mark.parametrize("fail_at", [1, 5])
@pytest.mark.parametrize("method", ["boltzmann", "dmc"])
def test_an_objective_exception_stops_the_fit(method, fail_at):
    calls = []

    def flaky(params):
        calls.append(params)
        if len(calls) == fail_at:
            raise RuntimeError("simulation crashed")
        return quadratic(params)

    fitter = chemfit.Fitter(flaky, INITIAL, BOUNDS, log_exceptions=False)

    with pytest.raises(RuntimeError, match="simulation crashed"):
        fit(fitter, 200, method=method, batch_size=4)

    # ChemFit finishes the batch holding the failure before it raises.
    assert fail_at <= len(calls) <= fail_at + 3


def test_swallowed_exceptions_keep_the_fit_going():
    calls = []

    def flaky(params):
        calls.append(params)
        if len(calls) == 5:
            raise RuntimeError("simulation crashed")
        return quadratic(params)

    fitter = chemfit.Fitter(
        flaky, INITIAL, BOUNDS, swallow_exceptions=True, log_exceptions=False
    )

    best = fit(fitter, 50, method="boltzmann")

    assert len(calls) == 50
    assert quadratic(best) <= quadratic(INITIAL)


def test_a_callback_exception_stops_the_fit():
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    def callback(step, contexts):
        raise KeyError("checkpoint failed")

    fitter.register_callback(callback, n_steps=3)

    with pytest.raises(KeyError, match="checkpoint failed"):
        fit(fitter, 100, method="gsa")

    assert len(recorder.seen) == 3


@pytest.mark.parametrize(
    ("initial", "bounds", "message"),
    [
        ({"a": 0.5, "b": 0.5}, {"a": (0, 1)}, "parameter 'b' has no bound"),
        ({"lj": {"sigma": 1.0}}, {}, r"parameter 'lj\.sigma' has no bound"),
        ({"a": 0.5}, {"a": (None, 1.0)}, "parameter 'a' has a one-sided"),
        ({"a": 0.5}, {"a": (0.0, None)}, "parameter 'a' has a one-sided"),
        ({"a": 0.5}, {"a": (0.0, math.inf)}, "parameter 'a' has a one-sided"),
        (
            {"p": np.zeros(3)},
            {"p": ([-1.0, np.nan, -1.0], 1.0)},
            "parameter 'p' has a one-sided",
        ),
        ({"a": 1.5}, {"a": (0.0, 1.0)}, "start of 'a' lies outside its bounds"),
        (
            {"p": np.array([[0.0, 0.0, 3.5]])},
            {"p": (-3.0, 3.0)},
            "start of 'p' lies outside its bounds",
        ),
        ({"a": math.nan}, {"a": (0.0, 1.0)}, "start of 'a' lies outside its bounds"),
        ({"a": 0.5}, {"a": (1.0, 0.0)}, "lower bound of 'a' exceeds its upper"),
        ({"a": 0.5}, {"a": (0.0,)}, r"bounds of 'a' must be a \(lower, upper\)"),
        ({"a": 0.5}, {"a": 1.0}, r"bounds of 'a' must be a \(lower, upper\)"),
        ({"a": 0.5}, {"a": ("lo", "hi")}, "bounds of 'a' must be numbers"),
        (
            {"p": np.zeros((4, 3))},
            {"p": (np.zeros(4), np.ones(4))},
            "bounds of 'p' must be numbers or arrays that broadcast",
        ),
        ({"a": {}}, {}, "holds no values to fit"),
    ],
)
def test_bad_bounds_and_starts_are_refused(initial, bounds, message):
    fitter, recorder = make_fitter(initial, accept_unknown_bounds=True)

    with pytest.raises(ValueError, match=message):
        fit(fitter, 10, bounds=bounds)

    assert recorder.seen == []


@pytest.mark.parametrize("method", ["portfolio", "dmc"])
def test_equal_bounds_are_refused_where_the_driver_needs_an_open_box(method):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    pinned = (np.full((4, 3), -2.0), np.full((4, 3), 2.0))
    pinned[0][1, 2] = pinned[1][1, 2] = INITIAL["positions"][1, 2]

    with pytest.raises(ValueError, match=r"bounds of 'lj\.sigma' are equal"):
        fit(fitter, 10, method=method, bounds={"lj": {"sigma": (1.1, 1.1)}})
    with pytest.raises(ValueError, match="bounds of 'positions' are equal"):
        fit(fitter, 10, method=method, bounds={"positions": pinned})

    assert recorder.seen == []


@pytest.mark.parametrize("method", ["boltzmann", "fast", "gsa", "qmc"])
def test_equal_bounds_hold_a_value_fixed(method):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)

    fit(fitter, 50, method=method, bounds={"lj": {"sigma": (1.1, 1.1)}})

    assert len(recorder.seen) > 1
    assert {p["lj"]["sigma"] for p in recorder.seen} == {1.1}


def test_bounds_naming_no_parameter_are_refused():
    fitter, _ = make_fitter({"sigma": 1.0}, {"sigma": (0.5, 1.5)})

    with pytest.raises(ValueError, match="bounds name 'sgima', which is not a"):
        objective(fitter, bounds={"sgima": (0.5, 1.5)})
    with pytest.raises(ValueError, match=r"bounds name 'sigma\.lo', which is not a"):
        objective(fitter, bounds={"sigma": {"lo": 0.5}})


@pytest.mark.parametrize(
    ("leaf", "got"),
    [
        ("lj", "str"),
        (True, "bool"),
        ([1.0, 2.0], "list"),
        (1j, "complex"),
        (np.arange(3), "an array of int64"),
        (np.zeros(2, dtype=bool), "an array of bool"),
    ],
)
def test_leaves_that_are_not_real_are_refused(leaf, got):
    fitter, _ = make_fitter({"model": {"x": leaf}}, {"model": {"x": (0, 1)}})

    with pytest.raises(TypeError, match=rf"parameter 'model\.x' .*got {got}"):
        objective(fitter)


def test_fitter_without_the_session_api_is_refused():
    older = types.SimpleNamespace(
        initial_parameters={"a": 0.5},
        bounds={"a": (0.0, 1.0)},
        init=lambda num_workers=1: None,
        ask=lambda params: 0.0,
        tell=lambda step=None: None,
        finish=lambda opt_params=None: opt_params,
    )

    with pytest.raises(TypeError, match=r"session API .* has no evaluate, step"):
        fit(older, 10)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"method": "anneal"}, ValueError, "method must be one of portfolio"),
        ({"budget": 0}, ValueError, "budget must be at least 1"),
        (
            {"method": "boltzmann", "steps_per_epoch": 0},
            ValueError,
            "steps_per_epoch must be at least 1",
        ),
        (
            {"method": "qmc", "n_starts": 0},
            ValueError,
            "n_starts must be at least 1",
        ),
        ({"x0": np.zeros(14)}, TypeError, "takes no x0"),
        ({"method": "qmc", "n_epochs": 5}, TypeError, "qmc takes no option n_epochs"),
        ({"method": "fast", "sigma": 1.0}, TypeError, "sigma"),
    ],
)
def test_fit_refuses_bad_arguments(kwargs, error, message):
    fitter, recorder = make_fitter(INITIAL, BOUNDS)
    kwargs = {"budget": 20, **kwargs}

    with pytest.raises(error, match=message):
        fit(fitter, **kwargs)

    assert recorder.seen == []


def test_unflatten_and_flatten_check_their_input():
    fitter, _ = make_fitter(INITIAL, BOUNDS)
    view = objective(fitter)

    with pytest.raises(ValueError, match="expected 14 values, got 13"):
        view.unflatten(np.zeros(13))
    with pytest.raises(ValueError, match=r"params have no parameter 'lj\.sigma'"):
        view.flatten({"positions": np.zeros((4, 3)), "lj": {"epsilon": 1.0}})
    with pytest.raises(ValueError, match=r"'positions' has shape \(12,\), expected"):
        view.flatten({"positions": np.zeros(12), "lj": {"epsilon": 1, "sigma": 1}})


def test_anneal_imports_without_chemfit():
    code = (
        "import sys; sys.modules['chemfit'] = None; "
        "import anneal; print(*anneal.chemfit.METHODS)"
    )
    root = os.path.dirname(os.path.dirname(anneal.__file__))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, *sys.path])}

    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert out.stdout.split() == list(METHODS)


@pytest.mark.parametrize("method", METHODS)
def test_reporter_lj_positions_stay_in_the_box(method):
    n_atoms, budget = 13, 2000
    initial = np.random.default_rng(0).uniform(-1.5, 1.5, size=(n_atoms, 3))
    fitter, recorder = make_fitter(
        {"positions": initial}, loss=lambda p: lj_energy(p["positions"])
    )

    best = fit(fitter, budget, method=method, bounds={"positions": (-3.0, 3.0)})

    evaluated = np.array([p["positions"] for p in recorder.seen])
    energies = [lj_energy(pos) for pos in evaluated]
    assert evaluated.shape == (len(recorder.seen), n_atoms, 3)
    assert len(recorder.seen) <= budget
    assert np.array_equal(evaluated[0], initial)
    assert np.all((evaluated >= -3.0) & (evaluated <= 3.0))
    assert best["positions"].shape == (n_atoms, 3)
    assert np.all(np.abs(best["positions"]) <= 3.0)
    assert lj_energy(best["positions"]) == min(energies) < energies[0]
