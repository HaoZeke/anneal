"""minimize reads fun and jac as global_optimize does, whatever is fixed."""

import warnings

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

X0 = np.array([1.0, 0.25, -0.5])
BOXES = {
    "free": [(-2.0, 2.0)] * 3,
    "one fixed": [(-2.0, 2.0), (0.25, 0.25), (-2.0, 2.0)],
}
ALL_FIXED = [(1.0, 1.0), (0.25, 0.25), (-0.5, -0.5)]
EVERY_BOX = {**BOXES, "all fixed": ALL_FIXED}


class Logged:
    """A shifted quadratic and its gradient, reshaped by ``shape``, with a call log."""

    def __init__(self, shape=None):
        self.shape = shape
        self.calls = []

    def fun(self, x):
        self.calls.append("value")
        return float(np.sum((x - 0.3) ** 2))

    def jac(self, x):
        self.calls.append("gradient")
        g = 2.0 * (x - 0.3)
        return g if self.shape is None else self.shape(g)


def _raise(g):
    raise ZeroDivisionError("division by zero")


BAD_GRADIENTS = {
    "short": lambda g: g[:-1],
    "long": lambda g: np.append(g, 0.0),
    "None": lambda g: None,
    "text": lambda g: "steep",
}
BAD_GRADIENT_MESSAGES = {
    "short": "grad_fn must return 3 numbers, got 2",
    "long": "grad_fn must return 3 numbers, got 4",
    "None": "grad_fn must return 3 numbers, got 1",
    "text": "grad_fn must return 3 numbers",
}


@pytest.mark.parametrize("gradient", BAD_GRADIENTS)
def test_a_bad_jac_ends_the_run_as_it_does_with_no_fixed_coordinate(gradient):
    for bounds in BOXES.values():
        logged = Logged(BAD_GRADIENTS[gradient])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError) as caught:
                anneal.minimize(logged.fun, X0, bounds, logged.jac, budget=200)
        assert logged.calls.count("gradient") == 1
        assert logged.calls[-1] == "gradient"
        assert str(caught.value) == BAD_GRADIENT_MESSAGES[gradient]


def _scored_warnings(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    return result, [str(w.message) for w in caught if w.category is RuntimeWarning]


def test_a_raising_jac_is_scored_as_a_zero_gradient_as_global_optimize_scores_it():
    logged, free = Logged(_raise), Logged(_raise)

    def full(z):
        return np.array([z[0], 0.25, z[1]])

    result, warned = _scored_warnings(
        lambda: anneal.minimize(logged.fun, X0, BOXES["one fixed"], logged.jac, budget=200)
    )
    expected, expected_warned = _scored_warnings(
        lambda: anneal.global_optimize(
            lambda z: free.fun(full(z)),
            [-2.0, -2.0],
            [2.0, 2.0],
            200,
            grad_fn=lambda z: free.jac(full(z)),
            x0=X0[[0, 2]],
        )
    )
    assert logged.calls == free.calls and logged.calls.count("gradient") > 1
    assert result.fun == expected["best_val"]
    assert (result.nfev, result.njev) == (expected["n_evals"], expected["n_grads"])
    assert len(warned) == 1 and warned == expected_warned
    assert warned[0].endswith("the first was: ZeroDivisionError: division by zero")


@pytest.mark.parametrize("budget", [50, 100, 200, 400])
@pytest.mark.parametrize("box", BOXES)
def test_a_raising_jac_leaves_nfev_and_njev_the_calls_made(box, budget):
    logged = Logged(_raise)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = anneal.minimize(logged.fun, X0, BOXES[box], logged.jac, budget=budget)
    assert result.nfev == logged.calls.count("value")
    assert result.njev == logged.calls.count("gradient")


@pytest.mark.parametrize("gradient", BAD_GRADIENTS)
def test_every_coordinate_fixed_never_calls_jac(gradient):
    logged = Logged(BAD_GRADIENTS[gradient])
    result = anneal.minimize(logged.fun, X0, ALL_FIXED, logged.jac, budget=50)
    assert logged.calls == ["value"]
    assert (result.nfev, result.njev) == (1, 0)
    np.testing.assert_array_equal(result.x, X0)
    assert result.fun == logged.fun(X0)


@pytest.mark.parametrize("box", EVERY_BOX)
def test_a_raising_fun_is_scored_as_the_worst_value_and_warned_once(box):
    calls = []

    def fun(x):
        calls.append(1)
        raise ZeroDivisionError("division by zero")

    result, warned = _scored_warnings(lambda: anneal.minimize(fun, X0, EVERY_BOX[box], budget=50))
    assert result.fun == np.inf and not result.success
    assert result.nfev == len(calls) > 0
    assert len(warned) == 1
    assert warned[0].startswith(f"a callback raised {len(calls)} exception(s): ")
    assert warned[0].endswith("the first was: ZeroDivisionError: division by zero")


class Outer:
    class Nested(Exception):
        pass


class Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("no text")


EXCEPTIONS = {
    "ZeroDivisionError": ZeroDivisionError("division by zero"),
    "empty": ValueError(),
    "nested": Outer.Nested("deep"),
    "unprintable": Unprintable(),
}


@pytest.mark.parametrize("error", EXCEPTIONS)
def test_every_coordinate_fixed_warns_as_global_optimize_warns_for_one_call(error):
    def fun(x):
        raise EXCEPTIONS[error]

    result, warned = _scored_warnings(lambda: anneal.minimize(fun, X0, ALL_FIXED))
    _, expected = _scored_warnings(
        lambda: anneal.global_optimize(fun, [0.0], [1.0], 1, x0=[0.5])
    )
    assert len(warned) == 1 and warned == expected
    assert (result.fun, result.nfev) == (np.inf, 1)


NOT_NUMBERS = {"text": "1.5", "None": None, "vector": np.array([1.0, 2.0])}


@pytest.mark.parametrize("box", EVERY_BOX)
@pytest.mark.parametrize("value", NOT_NUMBERS)
def test_a_value_that_is_not_a_number_is_global_optimizes_type_error(box, value):
    calls = []

    def fun(x):
        calls.append(1)
        return NOT_NUMBERS[value]

    kind = type(NOT_NUMBERS[value]).__name__
    with pytest.raises(TypeError, match=f"^obj_fn must return a float, got {kind}$"):
        anneal.minimize(fun, X0, EVERY_BOX[box], budget=50)
    assert len(calls) == 1


NUMBERS = {"nan": (float("nan"), np.inf), "float32": (np.float32(1.5), 1.5), "int": (2, 2.0)}


@pytest.mark.parametrize("box", EVERY_BOX)
@pytest.mark.parametrize("value", NUMBERS)
def test_a_number_is_read_as_global_optimize_reads_it(box, value):
    given, read = NUMBERS[value]
    result = anneal.minimize(lambda x: given, X0, EVERY_BOX[box], budget=50)
    assert result.fun == read and type(result.fun) is float


@pytest.mark.parametrize("box", EVERY_BOX)
def test_keyboard_interrupt_from_fun_ends_the_run(box):
    calls = []

    def fun(x):
        calls.append(1)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        anneal.minimize(fun, X0, EVERY_BOX[box], budget=50)
    assert len(calls) == 1
