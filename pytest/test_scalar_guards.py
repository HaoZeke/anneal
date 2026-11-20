# Copyright (c) anneal developers
"""A scalar argument that is NaN, infinite or out of range raises a typed
error, never a Rust panic.

A guard written as a comparison, such as ``x <= 0``, is false for NaN, so
NaN reached the native asserts and surfaced as ``PanicException``, which
``except Exception`` does not catch. Infinity passed a ``x > 0`` guard and
hung or overflowed. Counts too large to allocate raise ``MemoryError``.
"""

import inspect

import numpy as np
import pytest

import anneal
import anneal._core as core

NAN, INF = float("nan"), float("inf")
LOW = -np.ones(2)
HIGH = np.ones(2)


def f(x):
    return float(np.sum(np.asarray(x) ** 2))


def g(x):
    return 2.0 * np.asarray(x)


BOUNDS = core.Bounds(LOW, HIGH)
REQUIRED = {
    "obj_fn": f,
    "grad_fn": g,
    "preset": core.Boltzmann(),
    "budget": 30,
    "max_fevals": 30,
    "max_evals": 30,
    "n_epochs": 2,
    "steps_per_epoch": 2,
    "n_starts": 2,
    "max_fevals_per_start": 8,
    "x0": np.zeros(2),
    "center": np.zeros(2),
    "n": 3,
    "low": LOW,
    "high": HIGH,
    "objective": core.PyObjective(f, BOUNDS, grad_fn=g),
}
SPECIAL = {"cluster_search", "quench", "landfold", "landfold_book", "energy_hits", "osa_acceptance_rate"}


def call(name, **extra):
    function = getattr(core, name)
    kwargs = {
        parameter.name: REQUIRED[parameter.name]
        for parameter in inspect.signature(function).parameters.values()
        if parameter.default is inspect.Parameter.empty
    }
    if name == "box_ensemble_optimize":
        kwargs["grad_fn"] = g
    kwargs.update(extra)
    return function(**kwargs)


def float_parameters():
    for name in sorted(dir(core)):
        function = getattr(core, name)
        if name.startswith("_") or name in SPECIAL or not inspect.isbuiltin(function):
            continue
        for parameter in inspect.signature(function).parameters.values():
            if isinstance(parameter.default, float) and not isinstance(parameter.default, bool):
                yield name, parameter.name


FLOATS = list(float_parameters())


def test_the_sweep_reaches_the_guards_named_in_the_review():
    swept = set(FLOATS)
    for pair in [
        ("polish", "step0"),
        ("polish", "grad_tol"),
        ("qmc_polish", "step0"),
        ("qmc_polish_objective", "grad_tol"),
        ("shifted_qmc_polish", "step0"),
        ("run_hmc", "t_init"),
        ("run_hmc", "epsilon"),
        ("run_hmc", "q"),
        ("gle_langevin", "dt"),
        ("gle_langevin_preconditioned_objective", "dt"),
        ("additive_independence", "local_frac"),
    ]:
        assert pair in swept, pair


@pytest.mark.parametrize("name, parameter", FLOATS)
@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
def test_no_float_argument_reaches_a_rust_panic(name, parameter, value):
    try:
        call(name, **{parameter: value})
    except Exception:
        pass


@pytest.mark.parametrize("name", ["polish", "qmc_polish", "qmc_polish_objective", "shifted_qmc_polish"])
@pytest.mark.parametrize("value", [NAN, INF, -INF, 0.0, -1.0])
def test_polish_step0_must_be_positive_and_finite(name, value):
    with pytest.raises(anneal.InvalidArgumentError, match="step0 must be positive"):
        call(name, step0=value)


@pytest.mark.parametrize("name", ["polish", "qmc_polish", "qmc_polish_objective", "shifted_qmc_polish"])
@pytest.mark.parametrize("value", [NAN, INF, -INF, -1.0])
def test_polish_grad_tol_must_be_non_negative_and_finite(name, value):
    with pytest.raises(anneal.InvalidArgumentError, match="grad_tol must be non-negative"):
        call(name, grad_tol=value)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("t_init", NAN),
        ("t_init", INF),
        ("t_init", -INF),
        ("t_init", 0.0),
        ("t_init", -1.0),
        ("epsilon", NAN),
        ("epsilon", INF),
        ("epsilon", 0.0),
        ("epsilon", -1.0),
        ("q", NAN),
        ("q", -INF),
        ("q", INF),
        ("l_steps", 0),
    ],
)
def test_run_hmc_guards_reject_non_finite_and_out_of_range(parameter, value):
    with pytest.raises(anneal.InvalidArgumentError, match=parameter):
        call("run_hmc", **{parameter: value})


@pytest.mark.parametrize(
    "name",
    [
        "gle_langevin",
        "gle_langevin_objective",
        "gle_langevin_preconditioned",
        "gle_langevin_preconditioned_objective",
    ],
)
@pytest.mark.parametrize("dt", [NAN, INF, -INF, -1.0, 0.0])
def test_gle_dt_must_be_positive_and_finite(name, dt):
    with pytest.raises(anneal.InvalidArgumentError, match="dt must be positive"):
        call(name, dt=dt)


@pytest.mark.parametrize("local_frac", [NAN, INF, -INF, -0.1, 1.5])
def test_additive_local_frac_is_a_fraction(local_frac):
    with pytest.raises(anneal.InvalidArgumentError, match="local_frac"):
        call("additive_independence", local_frac=local_frac)


@pytest.mark.parametrize(
    "name, extra",
    [
        ("additive_independence", {"degree": 2**63}),
        ("additive_independence", {"grid_m": 2**63}),
        ("additive_independence", {"max_fevals": 2**63}),
        ("box_ensemble_optimize", {"replicas": 2**63}),
        ("ensemble_optimize", {"replicas": 2**63}),
        ("low_discrepancy_points", {"n": 2**63}),
        ("pilot_draws_qmc", {"n": 2**63}),
    ],
)
def test_a_count_too_large_to_allocate_is_a_memory_error(name, extra):
    with pytest.raises(MemoryError, match="too large"):
        call(name, **extra)


@pytest.mark.parametrize("name", ["run", "run_qmc"])
def test_an_epoch_count_past_memory_does_not_reserve_it(name):
    calls = []

    def counted(x):
        calls.append(1)
        return f(x)

    call(name, obj_fn=counted, n_epochs=2**63, max_evals=5)
    assert 0 < len(calls) <= 5


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"temp": INF}, "temp"),
        ({"temp": NAN}, "temp"),
        ({"sigma": INF}, "sigma"),
        ({"sigma": NAN}, "sigma"),
        ({"delta": NAN}, "delta"),
        ({"c_star": INF}, "c_star"),
    ],
)
def test_osa_acceptance_rate_rejects_non_finite_arguments(kwargs, message):
    arguments = {"delta": 1.0, "temp": 1.0, "sigma": 1.0, "trials": 10, **kwargs}
    with pytest.raises(anneal.InvalidArgumentError, match=message):
        core.osa_acceptance_rate(**arguments)


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_quench_rejects_a_start_that_is_not_finite(value):
    def fg(x):
        return float(np.sum(x**2)), 2.0 * np.asarray(x)

    start = np.ones(6)
    start[2] = value
    with pytest.raises(anneal.InvalidArgumentError, match="finite"):
        core.quench(fg, start, 1e-6, max_iter=5)


@pytest.mark.parametrize("sigma", [NAN, INF, -1.0, 0.0])
def test_landfold_rejects_an_explicit_sigma_it_cannot_use(sigma):
    rows = np.random.default_rng(0).normal(size=(30, 3))
    with pytest.raises(ValueError, match="sigma"):
        core.landfold(rows, landmarks=10, iterations=2, grid=8, sigma=sigma)


@pytest.mark.parametrize(
    "kwargs",
    [{"gamma": INF}, {"w0": INF}, {"merge_radius": INF}, {"gamma": NAN}],
)
def test_basin_bias_rejects_infinite_parameters(kwargs):
    with pytest.raises(anneal.InvalidArgumentError):
        anneal.BasinBias(4, **kwargs)
    with pytest.raises(anneal.InvalidArgumentError, match="temp"):
        anneal.BasinBias(4).deposit(np.zeros(12), INF)
