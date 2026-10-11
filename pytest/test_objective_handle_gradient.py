"""Objective handles read a gradient of any layout and reject a wrong one."""

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
    "qmc_polish_objective": lambda obj: anneal.qmc_polish_objective(
        obj, n_starts=4, max_fevals_per_start=30, seed=3
    ),
}


def _strided(g):
    wide = np.zeros((g.size, 3))
    wide[:, 1] = g
    view = wide[:, 1]
    assert not view.flags.c_contiguous
    return view


LAYOUTS = {
    "strided view": _strided,
    "list": lambda g: g.tolist(),
    "column": lambda g: g.reshape(-1, 1),
}


class Recorded:
    """The shifted quadratic, its gradient given in ``layout``, and a call log."""

    def __init__(self, layout=None):
        self.layout = layout
        self.calls = []

    def value(self, x):
        self.calls.append("value")
        return float(np.sum((np.asarray(x) - CENTRE) ** 2))

    def gradient(self, x):
        self.calls.append("gradient")
        g = 2.0 * (np.asarray(x, dtype=float) - CENTRE)
        return g if self.layout is None else self.layout(g)

    def handle(self, make=anneal.PyObjective):
        bounds = anneal.Bounds(LOW, HIGH)
        if make is CorePyObjective:
            bounds = bounds._core
        return make(self.value, bounds, grad_fn=self.gradient)


def _same_run(left, right):
    assert left["best_val"] == right["best_val"]
    np.testing.assert_array_equal(left["best_pos"], right["best_pos"])
    assert left["n_evals"] == right["n_evals"]


@pytest.mark.parametrize("entry", RUNS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_any_gradient_layout_gives_the_contiguous_run(entry, layout):
    plain, laid_out = Recorded(), Recorded(LAYOUTS[layout])
    expected = RUNS[entry](plain.handle())
    _same_run(RUNS[entry](laid_out.handle()), expected)
    assert laid_out.calls == plain.calls
    assert "gradient" in plain.calls


@pytest.mark.parametrize("entry", RUNS)
def test_the_extension_handle_takes_a_strided_gradient_too(entry):
    plain, strided = Recorded(), Recorded(_strided)
    expected = RUNS[entry](plain.handle(CorePyObjective))
    _same_run(RUNS[entry](strided.handle(CorePyObjective)), expected)
    assert strided.calls == plain.calls


@pytest.mark.parametrize("make", [anneal.PyObjective, CorePyObjective])
@pytest.mark.parametrize("entry", RUNS)
@pytest.mark.parametrize("extra", [-1, 1])
def test_a_gradient_of_another_length_is_a_type_error(entry, make, extra):
    if extra > 0:
        objective = Recorded(lambda g: np.append(g, np.zeros(extra)))
    else:
        objective = Recorded(lambda g: g[:extra])
    message = f"grad_fn must return 2 numbers, got {2 + extra}"
    with pytest.raises(TypeError, match=message):
        RUNS[entry](objective.handle(make))
    assert objective.calls.count("gradient") == 1
    assert objective.calls[-1] == "gradient"


class NoGradient(Exception):
    pass


@pytest.mark.parametrize("make", [anneal.PyObjective, CorePyObjective])
@pytest.mark.parametrize("entry", RUNS)
def test_a_raising_gradient_ends_the_run_with_its_exception(entry, make):
    def refuse(g):
        raise NoGradient("no gradient here")

    objective = Recorded(refuse)
    with pytest.raises(NoGradient, match="no gradient here"):
        RUNS[entry](objective.handle(make))
    assert objective.calls.count("gradient") == 1
    assert objective.calls[-1] == "gradient"


@pytest.mark.parametrize("entry", RUNS)
def test_a_gradient_that_is_not_numbers_is_a_type_error(entry):
    objective = Recorded(lambda g: "steep")
    with pytest.raises(TypeError, match="grad_fn must return numbers, got str"):
        RUNS[entry](objective.handle())
    assert objective.calls[-1] == "gradient"
