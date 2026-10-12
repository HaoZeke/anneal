"""The ensemble drivers report callback exceptions once, as the other drivers do."""

import warnings

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

LOW = [-2.0, -2.0, -2.0]
HIGH = [2.0, 2.0, 2.0]


def jac(x):
    return 2.0 * (x - 0.3)


DRIVERS = {
    "box_ensemble_optimize": lambda fun: anneal.box_ensemble_optimize(
        fun, LOW, HIGH, 200, grad_fn=jac
    ),
    "ensemble_optimize": lambda fun: anneal.ensemble_optimize(fun, LOW, HIGH, 200),
    "ensemble_optimize, one replica": lambda fun: anneal.ensemble_optimize(
        fun, LOW, HIGH, 200, replicas=1
    ),
    "ensemble_optimize with grad_fn": lambda fun: anneal.ensemble_optimize(
        fun, LOW, HIGH, 200, grad_fn=jac
    ),
    "ensemble_optimize with coverage": lambda fun: anneal.ensemble_optimize(
        fun, LOW, HIGH, 200, coverage_radius=0.25
    ),
}


def _runtime_warnings(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    return result, [str(w.message) for w in caught if w.category is RuntimeWarning]


@pytest.mark.parametrize("driver", DRIVERS)
def test_a_raising_fun_is_scored_as_the_worst_value_and_warned_once(driver):
    calls = []

    def fun(x):
        calls.append(1)
        raise ZeroDivisionError("division by zero")

    out, warned = _runtime_warnings(lambda: DRIVERS[driver](fun))
    assert out["best_val"] == np.inf
    assert out["n_evals"] == len(calls) > 0
    assert len(warned) == 1
    assert warned[0].startswith(f"a callback raised {len(calls)} exception(s): ")
    assert warned[0].endswith("the first was: ZeroDivisionError: division by zero")


FATAL = {
    "KeyboardInterrupt": (KeyboardInterrupt, None),
    "a value that is not a number": (TypeError, "^obj_fn must return a float, got str$"),
}


@pytest.mark.parametrize("error", FATAL)
@pytest.mark.parametrize("driver", DRIVERS)
def test_a_fatal_callback_error_still_ends_the_call(driver, error):
    kind, match = FATAL[error]
    calls = []

    def fun(x):
        calls.append(1)
        if kind is KeyboardInterrupt:
            raise KeyboardInterrupt
        return "1.5"

    def run():
        with pytest.raises(kind, match=match):
            DRIVERS[driver](fun)

    _, warned = _runtime_warnings(run)
    assert warned == []
    assert len(calls) > 0
