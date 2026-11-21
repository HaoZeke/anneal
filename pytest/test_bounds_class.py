# Copyright (c) anneal developers
"""``anneal.Bounds`` and ``anneal.PyObjective`` raise typed errors.

Both classes used to be the ones ``eindir-core`` exports. Its ``Bounds``
asserts on a bad box and its ``PyObjective`` unwraps the result of a
callback, so a mistake reached Python as ``PanicException``, a
``BaseException`` that skips ``except Exception``.
"""

import math
import warnings

import numpy as np
import pytest

import anneal
import anneal._core as core


def f(x):
    return float(np.sum(np.asarray(x) ** 2))


def g(x):
    return 2.0 * np.asarray(x)


def box():
    return anneal.Bounds(-np.ones(2), np.ones(2), 1e-9)


def test_the_package_exports_the_core_classes():
    assert anneal.Bounds is core.Bounds
    assert anneal.PyObjective is core.PyObjective


@pytest.mark.parametrize(
    "low, high",
    [
        (np.zeros(2), np.ones(3)),
        (np.ones(2), np.zeros(2)),
        (np.array([np.nan]), np.ones(1)),
        (np.zeros(1), np.array([np.nan])),
        (np.array([-np.inf]), np.ones(1)),
        (np.zeros(1), np.array([np.inf])),
        (np.array([-1e308]), np.array([1e308])),
    ],
)
def test_a_bad_box_raises_bounds_error(low, high):
    with pytest.raises(anneal.BoundsError) as caught:
        anneal.Bounds(low, high)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("slack", [-1.0, math.nan, math.inf])
def test_a_bad_slack_raises_bounds_error(slack):
    with pytest.raises(anneal.BoundsError, match="slack"):
        anneal.Bounds(np.zeros(1), np.ones(1), slack)


def test_a_fixed_axis_is_a_box():
    bounds = anneal.Bounds(np.array([0.0, 2.0]), np.array([1.0, 2.0]))
    assert bounds.dims == 2
    assert bounds.slack == 1e-6
    np.testing.assert_array_equal(bounds.clip(np.array([5.0, 0.0])), [1.0, 2.0])


@pytest.mark.parametrize("size", [0, 1, 3])
def test_clip_refuses_a_point_of_another_length(size):
    with pytest.raises(anneal.InvalidArgumentError, match="length"):
        box().clip(np.zeros(size))


@pytest.mark.parametrize("size", [0, 1, 3])
def test_contains_is_false_for_a_point_of_another_length(size):
    assert not box().contains(np.zeros(size))


def test_contains_honours_the_slack():
    bounds = anneal.Bounds(np.zeros(1), np.ones(1), 0.1)
    assert bounds.contains(np.array([1.05]))
    assert not bounds.contains(np.array([1.2]))


def test_pyobjective_checks_its_arguments():
    objective = anneal.PyObjective(f, box(), grad_fn=g)
    assert objective.dim == 2
    assert isinstance(objective.bounds, anneal.Bounds)
    assert objective.eval(np.array([1.0, 2.0])) == 5.0
    np.testing.assert_array_equal(objective.grad(np.array([1.0, 2.0])), [2.0, 4.0])
    with pytest.raises(anneal.InvalidArgumentError, match="length"):
        objective.eval(np.zeros(3))
    with pytest.raises(anneal.InvalidArgumentError, match="length"):
        objective.grad(np.zeros(1))
    with pytest.raises(anneal.InvalidArgumentError, match="grad_fn"):
        anneal.PyObjective(f, box()).grad(np.zeros(2))
    with pytest.raises(anneal.InvalidArgumentError, match="2 numbers"):
        anneal.PyObjective(f, box(), grad_fn=lambda x: [1.0]).grad(np.zeros(2))
    with pytest.raises(TypeError):
        anneal.PyObjective(f, (np.zeros(2), np.ones(2)))


GRADIENT_DRIVERS = [
    lambda o: core.qmc_polish_objective(o, 2, 5),
    lambda o: core.gle_langevin_objective(o, 30),
    lambda o: core.gle_langevin_preconditioned_objective(o, 30),
    lambda o: core.global_optimize_objective(o, 30),
]
VALUE_DRIVERS = [
    lambda o: core.qmc_best1bin_scout_objective(o, 40),
    lambda o: core.qmc_gsa_global_search_objective(o, 40),
    lambda o: core.qmc_trust_region_poll_objective(o, np.zeros(2), 20),
    lambda o: core.global_optimize_objective(o, 30, use_gradient=False),
]


@pytest.mark.parametrize("run", GRADIENT_DRIVERS)
def test_a_gradient_driver_needs_a_grad_fn(run):
    with pytest.raises(anneal.InvalidArgumentError, match="grad_fn"):
        run(anneal.PyObjective(f, box()))


@pytest.mark.parametrize("run", GRADIENT_DRIVERS + VALUE_DRIVERS)
def test_an_objective_that_raises_is_scored_and_warned(run):
    calls = []

    def flaky(x):
        calls.append(1)
        if len(calls) % 3 == 0:
            raise RuntimeError("flaky")
        return f(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run(anneal.PyObjective(flaky, box(), grad_fn=g))
    assert math.isfinite(out["best_val"])
    assert any("flaky" in str(w.message) for w in caught)


@pytest.mark.parametrize("run", GRADIENT_DRIVERS + VALUE_DRIVERS)
def test_an_interrupt_in_the_objective_is_raised(run):
    def interrupted(x):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(anneal.PyObjective(interrupted, box(), grad_fn=g))


@pytest.mark.parametrize("run", GRADIENT_DRIVERS)
def test_a_gradient_of_the_wrong_length_is_a_type_error(run):
    with pytest.raises(TypeError, match="grad_fn must return 2 numbers"):
        run(anneal.PyObjective(f, box(), grad_fn=lambda x: np.zeros(3)))
