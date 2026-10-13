"""Every function that takes a box refuses walls it cannot work with.

A NaN wall, ``low`` above ``high``, and an infinite wall where the function
needs finite ones are each a ``ValueError`` naming the coordinate, raised
before any call. Where an infinite wall or a width that overflows is taken,
every point asked is finite and inside the box.
"""

import math
import re

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")
from anneal._core import Bounds as CoreBounds  # noqa: E402
from anneal._core import PyObjective as CorePyObjective  # noqa: E402

INF, NAN = math.inf, math.nan

# low, high, and a point inside, or anywhere when the box holds none
BOXES = {
    "a NaN low wall": ([0.0, NAN], [1.0, 1.0], [0.5, 0.5]),
    "a NaN high wall": ([0.0, 0.0], [NAN, 1.0], [0.5, 0.5]),
    "low above high": ([0.0, 2.0], [1.0, 1.0], [0.5, 1.5]),
    "two walls at inf": ([0.0, INF], [1.0, INF], [0.5, 0.5]),
    "an infinite wall": ([-INF, 0.0], [0.0, 1.0], [-0.5, 0.5]),
    "a width that overflows": ([-1e308, 0.0], [1e308, 1.0], [0.0, 0.5]),
}

_NAN = "bounds must not be NaN at dimension {}"
_ABOVE = "low[1] must be less than or equal to high[1] (got 2, 1)"
_FINITE = "bounds must be finite at dimension {}"
_WIDTH = "bounds must be finite, with a finite width, at dimension {}"

# what each kind of function does with each box: the message it raises, or
# None when it runs
RULES = {
    "infinite walls": {
        "a NaN low wall": _NAN.format(1),
        "a NaN high wall": _NAN.format(0),
        "low above high": _ABOVE,
        "two walls at inf": "bounds must not have both walls at inf at dimension 1",
        "an infinite wall": None,
        "a width that overflows": None,
    },
    "finite walls": {
        "a NaN low wall": _NAN.format(1),
        "a NaN high wall": _NAN.format(0),
        "low above high": _ABOVE,
        "two walls at inf": _FINITE.format(1),
        "an infinite wall": _FINITE.format(0),
        "a width that overflows": None,
    },
    "a finite width": {
        "a NaN low wall": _NAN.format(1),
        "a NaN high wall": _NAN.format(0),
        "low above high": _ABOVE,
        "two walls at inf": _WIDTH.format(1),
        "an infinite wall": _WIDTH.format(0),
        "a width that overflows": _WIDTH.format(0),
    },
    "the drivers' box": {
        "a NaN low wall": _WIDTH.format(1),
        "a NaN high wall": _WIDTH.format(0),
        "low above high": _ABOVE,
        "two walls at inf": _WIDTH.format(1),
        "an infinite wall": _WIDTH.format(0),
        "a width that overflows": _WIDTH.format(0),
    },
}


class Recorded:
    """A bounded objective and its gradient, counting the points asked and
    those not finite or outside the box."""

    def __init__(self, low, high):
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.asked = 0
        self.bad = 0

    def note(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.asked += 1
        inside = np.all(x >= self.low) and np.all(x <= self.high)
        if not (np.all(np.isfinite(x)) and inside):
            self.bad += 1

    def f(self, x):
        self.note(x)
        s = np.asarray(x, dtype=np.float64) / (1.0 + np.abs(x))
        return float(np.sum((s - 0.2) ** 2))

    def g(self, x):
        self.note(x)
        x = np.asarray(x, dtype=np.float64)
        s = x / (1.0 + np.abs(x))
        return 2.0 * (s - 0.2) / (1.0 + np.abs(x)) / (1.0 + np.abs(x))


def _low_discrepancy_points(r, low, high, point):
    for x in anneal.low_discrepancy_points(low, high, 4):
        r.note(x)


ARRAY_FORM = {
    "polish": (
        "infinite walls",
        lambda r, low, high, point: anneal.polish(
            r.f, r.g, low, high, point, max_fevals=20
        ),
    ),
    "qmc_polish": (
        "finite walls",
        lambda r, low, high, point: anneal.qmc_polish(r.f, r.g, low, high, 4, 10),
    ),
    "qmc_best1bin_scout": (
        "infinite walls",
        lambda r, low, high, point: anneal.qmc_best1bin_scout(
            r.f, low, high, 40, population_size=8
        ),
    ),
    "qmc_trust_region_poll": (
        "finite walls",
        lambda r, low, high, point: anneal.qmc_trust_region_poll(
            r.f, low, high, point, 40
        ),
    ),
    "shifted_qmc_polish": (
        "infinite walls",
        lambda r, low, high, point: anneal.shifted_qmc_polish(
            r.f, r.g, low, high, 4, 10
        ),
    ),
    "additive_independence": (
        "a finite width",
        lambda r, low, high, point: anneal.additive_independence(
            r.f, low, high, 60, n_pilot=16
        ),
    ),
    "estimate_gle_omega0": (
        "finite walls",
        lambda r, low, high, point: anneal.estimate_gle_omega0(r.f, r.g, low, high),
    ),
    "gle_langevin": (
        "finite walls",
        lambda r, low, high, point: anneal.gle_langevin(r.f, r.g, low, high, 40),
    ),
    "gle_langevin_preconditioned": (
        "finite walls",
        lambda r, low, high, point: anneal.gle_langevin_preconditioned(
            r.f, r.g, low, high, 40
        ),
    ),
    "low_discrepancy_points": ("a finite width", _low_discrepancy_points),
    "run_qmc": (
        "the drivers' box",
        lambda r, low, high, point: anneal.run_qmc(
            r.f,
            low,
            high,
            anneal.Fast(1.0, 1.0),
            n_starts=2,
            n_epochs=2,
            steps_per_epoch=5,
            max_evals=20,
        ),
    ),
}

HANDLE_FORM = {
    "qmc_polish_objective": (
        "finite walls",
        lambda h, point: anneal.qmc_polish_objective(h, 4, 10),
    ),
    "qmc_best1bin_scout_objective": (
        "infinite walls",
        lambda h, point: anneal.qmc_best1bin_scout_objective(
            h, 40, population_size=8
        ),
    ),
    "qmc_gsa_global_search_objective": (
        "finite walls",
        lambda h, point: anneal.qmc_gsa_global_search_objective(h, 40, n_chains=4),
    ),
    "qmc_trust_region_poll_objective": (
        "finite walls",
        lambda h, point: anneal.qmc_trust_region_poll_objective(h, point, 40),
    ),
    "gle_langevin_objective": (
        "finite walls",
        lambda h, point: anneal.gle_langevin_objective(h, 40),
    ),
    "gle_langevin_preconditioned_objective": (
        "finite walls",
        lambda h, point: anneal.gle_langevin_preconditioned_objective(h, 40),
    ),
    "global_optimize_objective": (
        "finite walls",
        lambda h, point: anneal.global_optimize_objective(h, 40),
    ),
}


def _holds(r, run, rule):
    if rule is None:
        run()
        assert r.asked > 0
        assert r.bad == 0
    else:
        with pytest.raises(ValueError, match=f"^{re.escape(rule)}$"):
            run()
        assert r.asked == 0


@pytest.mark.parametrize("box", BOXES)
@pytest.mark.parametrize("name", ARRAY_FORM)
def test_an_array_form_function_takes_only_the_walls_it_works_with(name, box):
    kind, call = ARRAY_FORM[name]
    low, high, point = (np.array(v) for v in BOXES[box])
    r = Recorded(low, high)
    _holds(r, lambda: call(r, low, high, point), RULES[kind][box])


@pytest.mark.parametrize("box", [box for box in BOXES if box != "low above high"])
@pytest.mark.parametrize("name", HANDLE_FORM)
def test_an_objective_function_takes_only_the_walls_it_works_with(name, box):
    kind, call = HANDLE_FORM[name]
    low, high, point = (np.array(v) for v in BOXES[box])
    r = Recorded(low, high)
    handle = CorePyObjective(r.f, CoreBounds(low, high), r.g)
    _holds(r, lambda: call(handle, point), RULES[kind][box])


@pytest.mark.parametrize(
    "low, high, message",
    [
        ([0.0, NAN], [1.0, 1.0], "coordinate 1 has a NaN wall: low nan, high 1.0"),
        ([0.0, 0.0], [NAN, 1.0], "coordinate 0 has a NaN wall: low 0.0, high nan"),
    ],
)
def test_an_objective_over_the_extension_box_refuses_a_nan_wall(low, high, message):
    r = Recorded(low, high)
    bounds = CoreBounds(np.array(low), np.array(high))
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        anneal.PyObjective(r.f, bounds, r.g)
    assert r.asked == 0


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "low, high, i",
    [
        ([0.0, NAN], [1.0, 1.0], 1),
        ([0.0, 0.0], [NAN, 1.0], 0),
        ([0.0, -INF], [1.0, 1.0], 1),
        ([0.0, 2.0], [1.0, 1.0], 1),
    ],
)
def test_the_device_drivers_name_the_coordinate_they_refuse(low, high, i, batched):
    r = Recorded(low, high)
    kwargs = {"n_epochs": 1, "steps_per_epoch": 1}
    if batched:
        kwargs["n_chains"] = 4
    entry = anneal.run_ensemble if batched else anneal.run_device
    message = f"bounds must be finite with low <= high at dimension {i}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        entry(r.f, np.asarray(low), np.asarray(high), anneal.Boltzmann(), **kwargs)
    assert r.asked == 0


def test_an_objective_takes_infinite_walls():
    r = Recorded([-INF, 0.0], [0.0, INF])
    handle = anneal.PyObjective(r.f, anneal.Bounds([-INF, 0.0], [0.0, INF]), r.g)
    out = anneal.qmc_best1bin_scout_objective(handle, 40, population_size=8)
    assert np.all(np.isfinite(out["best_pos"]))
    assert r.asked > 0
    assert r.bad == 0
