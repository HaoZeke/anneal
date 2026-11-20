# Copyright (c) anneal developers
"""A bad box or a bad state raises a typed error, never a Rust panic.

``Bounds::new`` panics on a pair of different lengths or an inverted pair, and
a panic reaches Python as ``PanicException``, a ``BaseException`` that skips
``except Exception``. Every driver validates its box first.
"""

import inspect

import numpy as np
import pytest

import anneal
import anneal._core as core

DRIVERS = [
    "run",
    "run_qmc",
    "run_hmc",
    "polish",
    "qmc_polish",
    "qmc_best1bin_scout",
    "qmc_gsa_global_search",
    "qmc_trust_region_poll",
    "shifted_qmc_polish",
    "additive_independence",
    "estimate_gle_omega0",
    "gle_langevin",
    "gle_langevin_preconditioned",
    "dmc_population_optimize",
    "gpmd_optimize",
    "amsa_optimize",
    "box_ensemble_optimize",
    "ensemble_optimize",
    "bfwt_optimize",
    "global_optimize",
    "low_discrepancy_points",
]

# landfold takes the bounds of a sketch-map embedding, not of a search box.
NOT_BOXES = {"landfold", "landfold_book"}
NEEDS_GRADIENT = {"box_ensemble_optimize"}

LOW = np.full(2, -1.0)
HIGH = np.ones(2)


def f(x):
    return float(np.sum(np.asarray(x) ** 2))


def g(x):
    return 2.0 * np.asarray(x)


REQUIRED = {
    "obj_fn": f,
    "grad_fn": g,
    "preset": core.Boltzmann(),
    "budget": 60,
    "max_fevals": 60,
    "max_evals": 60,
    "n_epochs": 2,
    "steps_per_epoch": 2,
    "n_starts": 2,
    "max_fevals_per_start": 10,
    "x0": np.zeros(2),
    "center": np.zeros(2),
    "n": 3,
}


def call(name, low, high):
    function = getattr(core, name)
    kwargs = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.default is inspect.Parameter.empty and parameter.name not in ("low", "high"):
            kwargs[parameter.name] = REQUIRED[parameter.name]
    kwargs.update(low=low, high=high)
    if name in NEEDS_GRADIENT:
        kwargs["grad_fn"] = g
    return function(**kwargs)


def test_every_driver_is_covered():
    drivers = set(DRIVERS)
    boxed = {
        name
        for name in dir(core)
        if inspect.isbuiltin(getattr(core, name))
        and not name.startswith("_")
        and {"low", "high"} <= set(inspect.signature(getattr(core, name)).parameters)
    }
    assert boxed - NOT_BOXES - drivers == set(), "add the new box driver to DRIVERS"


@pytest.mark.parametrize("name", DRIVERS)
def test_a_valid_box_is_accepted(name):
    call(name, LOW, HIGH)


@pytest.mark.parametrize("name", DRIVERS)
@pytest.mark.parametrize(
    "case, low, high",
    [
        ("inverted", HIGH, LOW),
        ("nan low", np.array([np.nan, -1.0]), HIGH),
        ("nan high", LOW, np.array([1.0, np.nan])),
    ],
)
def test_an_unordered_box_raises_bounds_error(name, case, low, high):
    with pytest.raises(anneal.BoundsError):
        call(name, low, high)


@pytest.mark.parametrize("name", DRIVERS)
def test_a_box_with_mismatched_sides_raises_a_typed_error(name):
    with pytest.raises(anneal.InvalidArgumentError, match="length|shape"):
        call(name, LOW, np.ones(3))


@pytest.mark.parametrize("n_points", [2, 4])
def test_basin_bias_rejects_a_state_of_the_wrong_length(n_points):
    bias = anneal.BasinBias(n_points)
    good = np.linspace(0.0, 1.0, 3 * n_points)
    bias.deposit(good, 1.0)
    assert bias.potential(good) > 0.0
    for size in (0, 3 * n_points - 1, 3 * n_points + 1):
        bad = np.zeros(size)
        with pytest.raises(anneal.InvalidArgumentError, match="length"):
            bias.potential(bad)
        with pytest.raises(anneal.InvalidArgumentError, match="length"):
            bias.deposit(bad, 1.0)
    assert bias.n_basins == 1
