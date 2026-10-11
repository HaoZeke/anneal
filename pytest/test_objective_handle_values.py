"""Objective handles end the run on a bad value and score NaN as +inf."""

import math

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")
from anneal._core import PyObjective as CorePyObjective  # noqa: E402

LOW = np.array([-1.0, -1.0])
HIGH = np.array([1.0, 1.0])
CENTRE = np.array([0.25, -0.4])

RUNS = {
    "gle_langevin_objective": lambda obj: anneal.gle_langevin_objective(
        obj, max_fevals=120, seed=3
    ),
    "gle_langevin_preconditioned_objective": (
        lambda obj: anneal.gle_langevin_preconditioned_objective(
            obj, max_fevals=120, seed=3
        )
    ),
    "global_optimize_objective": lambda obj: anneal.global_optimize_objective(
        obj, 120, seed=3
    ),
    "global_optimize_objective without gradients": (
        lambda obj: anneal.global_optimize_objective(
            obj, 120, seed=3, use_gradient=False
        )
    ),
    "qmc_best1bin_scout_objective": (
        lambda obj: anneal.qmc_best1bin_scout_objective(
            obj, 60, seed=3, population_size=8
        )
    ),
    "qmc_gsa_global_search_objective": (
        lambda obj: anneal.qmc_gsa_global_search_objective(
            obj, 60, seed=3, n_chains=4
        )
    ),
    "qmc_polish_objective": lambda obj: anneal.qmc_polish_objective(
        obj, n_starts=4, max_fevals_per_start=30, seed=3
    ),
    "qmc_trust_region_poll_objective": (
        lambda obj: anneal.qmc_trust_region_poll_objective(
            obj, np.array([0.5, 0.5]), 60, seed=3
        )
    ),
}

HANDLES = [anneal.PyObjective, CorePyObjective]
HANDLE_IDS = ["package handle", "extension handle"]


def _from_the_third(x, n):
    return n >= 3


class Recorded:
    """The shifted quadratic and its gradient, and a call log.

    Where ``when(x, n)`` holds for the ``n``-th value, ``spoil()`` is
    returned in place of it.
    """

    def __init__(self, spoil=None, when=_from_the_third):
        self.spoil = spoil
        self.when = when
        self.calls = []

    def value(self, x):
        self.calls.append("value")
        if self.spoil is not None and self.when(x, self.calls.count("value")):
            return self.spoil()
        return float(np.sum((np.asarray(x) - CENTRE) ** 2))

    def gradient(self, x):
        self.calls.append("gradient")
        return 2.0 * (np.asarray(x, dtype=float) - CENTRE)

    def handle(self, make=anneal.PyObjective):
        bounds = anneal.Bounds(LOW, HIGH)
        if make is CorePyObjective:
            bounds = bounds._core
        return make(self.value, bounds, grad_fn=self.gradient)


def _ended_at_the_third_value(objective):
    assert objective.calls.count("value") == 3
    assert objective.calls[-1] == "value"


class NoValue(Exception):
    pass


def _refuse():
    raise NoValue("no value here")


@pytest.mark.parametrize("make", HANDLES, ids=HANDLE_IDS)
@pytest.mark.parametrize("entry", RUNS)
def test_a_raising_objective_ends_the_run_with_its_exception(entry, make):
    objective = Recorded(_refuse)
    with pytest.raises(NoValue, match="no value here"):
        RUNS[entry](objective.handle(make))
    _ended_at_the_third_value(objective)


NOT_NUMBERS = {
    "str": "flat",
    "None": None,
    "vector": np.array([1.0, 2.0]),
}


@pytest.mark.parametrize("entry", RUNS)
@pytest.mark.parametrize("returned", NOT_NUMBERS)
def test_a_value_that_is_not_a_number_ends_the_run_as_in_global_optimize(
    entry, returned
):
    def spoil():
        return NOT_NUMBERS[returned]

    with pytest.raises(TypeError) as callable_run:
        anneal.global_optimize(Recorded(spoil).value, LOW, HIGH, 120, seed=3)
    objective = Recorded(spoil)
    with pytest.raises(TypeError) as handle_run:
        RUNS[entry](objective.handle())
    assert str(handle_run.value) == str(callable_run.value)
    assert str(handle_run.value).startswith("obj_fn must return a float, got ")
    _ended_at_the_third_value(objective)


@pytest.mark.parametrize("entry", RUNS)
@pytest.mark.parametrize("returned", NOT_NUMBERS)
def test_the_extension_handle_ends_the_run_on_a_value_that_is_not_a_number(
    entry, returned
):
    objective = Recorded(lambda: NOT_NUMBERS[returned])
    with pytest.raises(TypeError):
        RUNS[entry](objective.handle(CorePyObjective))
    _ended_at_the_third_value(objective)


def _away_from_the_minimum(x, n):
    return float(np.sum((np.asarray(x) - CENTRE) ** 2)) > 0.1


@pytest.mark.parametrize("make", HANDLES, ids=HANDLE_IDS)
@pytest.mark.parametrize("entry", RUNS)
def test_a_nan_value_gives_the_run_an_infinite_one_gives(entry, make):
    nan = Recorded(lambda: math.nan, _away_from_the_minimum)
    inf = Recorded(lambda: math.inf, _away_from_the_minimum)
    got = RUNS[entry](nan.handle(make))
    expected = RUNS[entry](inf.handle(make))
    assert not math.isnan(got["best_val"])
    np.testing.assert_equal(got, expected)
    assert nan.calls == inf.calls
