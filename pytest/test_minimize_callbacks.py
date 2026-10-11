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


@pytest.mark.parametrize("gradient", BAD_GRADIENTS)
def test_a_bad_jac_ends_the_run_as_it_does_with_no_fixed_coordinate(gradient):
    messages = {}
    for box, bounds in BOXES.items():
        logged = Logged(BAD_GRADIENTS[gradient])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(TypeError, match="grad_fn must return 3 numbers") as caught:
                anneal.minimize(logged.fun, X0, bounds, logged.jac, budget=200)
        assert logged.calls.count("gradient") == 1
        assert logged.calls[-1] == "gradient"
        messages[box] = str(caught.value)
    assert messages["one fixed"] == messages["free"]


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
