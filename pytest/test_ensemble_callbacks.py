"""The ensemble drivers handle callback exceptions as the other drivers do.

An ordinary exception from the objective or the gradient is scored and
counted in a single warning. A fatal one ends the call, and fewer callbacks
follow it than there are replicas.
"""

import itertools
import warnings
from functools import partial

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

LOW = [-2.0, -2.0, -2.0]
HIGH = [2.0, 2.0, 2.0]


def quadratic(x):
    return float(np.sum((x - 0.3) ** 2))


def jac(x):
    return 2.0 * (x - 0.3)


# Each path through the two drivers, by the options that select it.
DRIVERS = {
    "box_ensemble_optimize": partial(anneal.box_ensemble_optimize, grad_fn=jac),
    "ensemble_optimize": partial(anneal.ensemble_optimize),
    "ensemble_optimize, one replica": partial(anneal.ensemble_optimize, replicas=1),
    "ensemble_optimize with grad_fn": partial(anneal.ensemble_optimize, grad_fn=jac),
    "ensemble_optimize with coverage": partial(
        anneal.ensemble_optimize, coverage_radius=0.25
    ),
    "ensemble_optimize with grad_fn and coverage": partial(
        anneal.ensemble_optimize, grad_fn=jac, coverage_radius=0.25
    ),
}
WITH_GRADIENT = [name for name, run in DRIVERS.items() if "grad_fn" in run.keywords]
# Both drivers run four replicas unless told otherwise.
REPLICAS = {name: run.keywords.get("replicas", 4) for name, run in DRIVERS.items()}

# What each callback raises when it is made to raise an ordinary exception.
ERRORS = {
    "fun": (ZeroDivisionError, "division by zero"),
    "grad_fn": (FloatingPointError, "overflow in the gradient"),
}


def _run(driver, fun, **options):
    return DRIVERS[driver](fun, LOW, HIGH, 200, **options)


def _logged(callback, kind, log, raises=None):
    """`callback`, recording each call in `log` as `(kind, raised)`.

    `raises(n)` is the exception the n-th call raises instead of returning,
    or None. The objective and the gradient share a log to keep one order.
    """
    numbers = itertools.count(1)

    def call(x):
        error = raises(next(numbers)) if raises else None
        log.append((kind, error is not None))
        if error is not None:
            raise error
        return callback(x)

    return call


def _every_third(kind):
    error, message = ERRORS[kind]
    return lambda n: error(message) if n % 3 == 0 else None


def _text(kind):
    error, message = ERRORS[kind]
    return f"{error.__name__}: {message}"


def _runtime_warnings(run):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    return result, [str(w.message) for w in caught if w.category is RuntimeWarning]


def _assert_warned_once(warned, count, first):
    assert len(warned) == 1
    assert warned[0].startswith(f"a callback raised {count} exception(s): ")
    assert warned[0].endswith(f"the first was: {first}")


@pytest.mark.parametrize("driver", DRIVERS)
def test_a_raising_fun_is_scored_as_the_worst_value_and_warned_once(driver):
    calls = []

    def fun(x):
        calls.append(1)
        raise ZeroDivisionError("division by zero")

    out, warned = _runtime_warnings(lambda: _run(driver, fun))
    assert out["best_val"] == np.inf
    assert out["n_evals"] == len(calls) > 0
    _assert_warned_once(warned, len(calls), "ZeroDivisionError: division by zero")


@pytest.mark.parametrize(
    "raising", [("grad_fn",), ("fun", "grad_fn")], ids=" and ".join
)
@pytest.mark.parametrize("driver", WITH_GRADIENT)
def test_a_raising_grad_fn_is_scored_as_zero_and_counted_in_the_one_warning(
    driver, raising
):
    log = []
    fun = _logged(
        quadratic, "fun", log, _every_third("fun") if "fun" in raising else None
    )
    grad = _logged(jac, "grad_fn", log, _every_third("grad_fn"))
    out, warned = _runtime_warnings(lambda: _run(driver, fun, grad_fn=grad))
    kinds = [kind for kind, _ in log]
    raised = [kind for kind, error in log if error]
    assert set(raised) == set(raising)
    assert out["n_evals"] == kinds.count("fun")
    assert out["n_grads"] == kinds.count("grad_fn")
    assert np.isfinite(out["best_val"])
    _assert_warned_once(warned, len(raised), _text(raised[0]))


@pytest.mark.parametrize("driver", DRIVERS)
def test_a_fun_raising_on_every_third_call_is_counted_and_the_run_goes_on(driver):
    log = []
    fun = _logged(quadratic, "fun", log, _every_third("fun"))
    out, warned = _runtime_warnings(lambda: _run(driver, fun))
    raised = [kind for kind, error in log if error]
    assert len(raised) > 1
    assert out["n_evals"] == len(log)
    assert np.isfinite(out["best_val"])
    _assert_warned_once(warned, len(raised), _text("fun"))


FATAL = {
    "KeyboardInterrupt": (KeyboardInterrupt, None),
    "a value that is not a number": (
        TypeError,
        "^obj_fn must return a float, got str$",
    ),
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
            _run(driver, fun)

    _, warned = _runtime_warnings(run)
    assert warned == []
    assert len(calls) > 0


INTERRUPTED = [(driver, "fun") for driver in DRIVERS]
INTERRUPTED += [(driver, "grad_fn") for driver in WITH_GRADIENT]


@pytest.mark.parametrize("driver, kind", INTERRUPTED)
def test_after_a_keyboard_interrupt_fewer_calls_follow_than_there_are_replicas(
    driver, kind
):
    log = []

    def interrupt(n):
        return KeyboardInterrupt() if n == 20 else None

    fun = _logged(quadratic, "fun", log, interrupt if kind == "fun" else None)
    grad = _logged(jac, "grad_fn", log, interrupt if kind == "grad_fn" else None)
    options = {"grad_fn": grad} if driver in WITH_GRADIENT else {}

    def run():
        with pytest.raises(KeyboardInterrupt):
            _run(driver, fun, **options)

    _, warned = _runtime_warnings(run)
    assert warned == []
    after = log[log.index((kind, True)) + 1 :]
    assert len(after) < REPLICAS[driver]
