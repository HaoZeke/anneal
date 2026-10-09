"""Both ChemFit session protocols through every bridge, and the first error.

Current ChemFit drives a ``Fitter`` with ``evaluate`` / ``step``; ChemFit 3.1
with ``ask`` / ``tell``. Every bridge must speak whichever pair the fitter
has, and refuse a fitter with neither before starting its session; only
``run_benchmark`` still drives half a pair, as anneal 0.10.0 did. An
exception raised by the fitter, or a loss that is not a real number, must
reach the caller unchanged, with no further fitter call and no ``finish``.
"""

import warnings
from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("anneal")

from chemfit_doubles import (  # noqa: E402
    ENTRIES,
    PROTOCOLS,
    NextFitter,
    Recorder,
    drive,
    protocol_names,
    same_params,
    sum_of_squares,
)

# fit_chemfit steps once every ``tell_every`` portfolio evaluations or once an
# epoch; these settings make it step often enough to reach a failing step.
FREQUENT_STEPS = {"fit_chemfit": {"tell_every": 2, "steps_per_epoch": 10}}


def _problem():
    initial = {
        "positions": np.array([[0.9, -0.6, 0.3], [-1.2, 0.4, 1.5]]),
        "cell": {"a": 1.1},
    }
    bounds = {"positions": (-2.0, 2.0), "cell": {"a": (0.5, 2.0)}}
    return initial, bounds


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("driver", ["portfolio", "boltzmann"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_every_bridge_speaks_both_fitter_protocols(entry, driver, protocol):
    fitter = protocol(*_problem())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = drive(entry, fitter, 120, driver=driver, seed=1)
    evaluate, step = protocol_names(protocol)
    assert fitter.calls[0] == "init"
    assert fitter.calls[-1] == "finish"
    assert set(fitter.calls) == {"init", evaluate, step, "finish"}
    assert fitter.calls.count("init") == fitter.calls.count("finish") == 1
    assert 0 < len(fitter.evaluated) <= 120
    best = fitter.evaluated[int(np.argmin(fitter.losses))]
    assert same_params(fitter.finished_with, best)
    assert out is fitter.finished_with


class _BothProtocols(NextFitter):
    def ask(self, parameters, context_index=0):
        return self._evaluate("ask", parameters)

    def tell(self, step=None):
        self._record("tell")


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_fitter_with_both_protocols_is_driven_through_evaluate_and_step(entry):
    fitter = _BothProtocols(*_problem())
    drive(entry, fitter, 60)
    assert "evaluate" in fitter.calls and "step" in fitter.calls
    assert "ask" not in fitter.calls and "tell" not in fitter.calls


class _EvaluateWithoutStep(Recorder):
    def evaluate(self, parameters, context_index=0):
        return self._evaluate("evaluate", parameters)


class _AskWithoutTell(Recorder):
    def ask(self, parameters, context_index=0):
        return self._evaluate("ask", parameters)


# run_benchmark still drives half a pair; test_chemfit_compat.py covers it.
REFUSED = [
    (entry, fitter_type)
    for entry in ENTRIES
    for fitter_type in (Recorder, _EvaluateWithoutStep, _AskWithoutTell)
    if entry != "run_benchmark" or fitter_type is Recorder
]


@pytest.mark.parametrize(
    "entry, fitter_type",
    REFUSED,
    ids=[f"{entry}-{fitter_type.__name__}" for entry, fitter_type in REFUSED],
)
def test_a_fitter_without_a_whole_protocol_is_refused_before_init(entry, fitter_type):
    fitter = fitter_type(*_problem())
    with pytest.raises(TypeError, match="evaluate"):
        drive(entry, fitter, 50)
    assert fitter.calls == []


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_finish_that_takes_no_parameters_is_called_bare(entry, protocol):
    class BareFinish(protocol):
        def finish(self):
            self.calls.append("finish")
            return {"picked": "by the fitter"}

    fitter = BareFinish(*_problem())
    assert drive(entry, fitter, 40) == {"picked": "by the fitter"}
    assert fitter.calls.count("finish") == 1


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_an_error_inside_finish_reaches_the_caller_after_one_call(entry, protocol):
    error = TypeError("a fitter callback failed")

    class FailingFinish(protocol):
        def finish(self, opt_params=None):
            self.calls.append("finish")
            raise error

    fitter = FailingFinish(*_problem())
    with pytest.raises(TypeError) as caught:
        drive(entry, fitter, 40)
    assert caught.value is error
    assert fitter.calls.count("finish") == 1


@pytest.mark.parametrize("stage", ["evaluation", "step"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("driver", ["portfolio", "boltzmann"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_first_fitter_error_reaches_the_caller_and_ends_the_drive(
    entry, driver, protocol, stage
):
    evaluate, step = protocol_names(protocol)
    failing = evaluate if stage == "evaluation" else step
    error = RuntimeError(f"{failing} failed")
    fitter = protocol(*_problem(), fail={failing: (3, error)})
    with pytest.raises(RuntimeError) as caught:
        drive(entry, fitter, 200, driver=driver, **FREQUENT_STEPS.get(entry, {}))
    assert caught.value is error
    assert fitter.calls[-1] == failing
    assert fitter.calls.count(failing) == 3
    assert "finish" not in fitter.calls


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_keyboard_interrupt_from_the_fitter_reaches_the_caller(entry, protocol):
    evaluate, _ = protocol_names(protocol)
    interrupt = KeyboardInterrupt()
    fitter = protocol(*_problem(), fail={evaluate: (2, interrupt)})
    with pytest.raises(KeyboardInterrupt) as caught:
        drive(entry, fitter, 100)
    assert caught.value is interrupt
    assert fitter.calls[-1] == evaluate
    assert "finish" not in fitter.calls


def _matrix(rows, dtype=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix(rows, dtype=dtype)


class _DeviceScalar:
    """A 0-d array NumPy cannot read, as on a GPU or under autograd.

    Only ``float()`` reads it, as ``float()`` reads a CuPy or torch scalar.
    """

    def __init__(self, value, dtype=np.float64, refusal=TypeError):
        self.value, self.dtype, self.refusal = value, np.dtype(dtype), refusal

    def __array__(self, dtype=None, copy=None):
        raise self.refusal("this array cannot be read as a NumPy array")

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"_DeviceScalar({self.value!r}, {self.dtype}, {self.refusal.__name__})"


NOT_REAL = [
    None,
    "1.5",
    "nan",
    complex(1.0, 0.0),
    [0.5, 0.25],
    {"loss": 1.0},
    True,
    np.array([1.0, 2.0]),
    np.array([True]),
    np.array(["1.5"]),
    np.array([0.5 + 0j]),
    np.complex128(0.5),
    Decimal("sNaN"),
    _matrix([[0.5, 0.25]]),
    np.array(["1.5"], dtype=object),
    np.array([None], dtype=object),
    _DeviceScalar(1.0, bool),
    _DeviceScalar(0.5, complex),
]


@pytest.mark.parametrize("loss", NOT_REAL, ids=repr)
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_loss_that_is_not_one_real_number_is_an_error(entry, protocol, loss):
    evaluate, _ = protocol_names(protocol)
    fitter = protocol(*_problem(), loss=lambda params: loss)
    with pytest.raises(TypeError, match="loss"):
        drive(entry, fitter, 60)
    assert fitter.calls.count(evaluate) == 1
    assert "finish" not in fitter.calls


REAL_FORMS = {
    "int": lambda v: int(round(1000 * v)),
    "float": float,
    "float32": np.float32,
    "float64": np.float64,
    "int64": lambda v: np.int64(round(1000 * v)),
    "0-d array": np.array,
    "one-element list": lambda v: [v],
    "one-element tuple": lambda v: (v,),
    "one-element array": lambda v: np.array([v]),
    "1x1 float32 array": lambda v: np.array([[v]], dtype=np.float32),
    "list of a one-element array": lambda v: [np.array([v])],
    "Decimal": lambda v: Decimal(repr(v)),
    "list of a Decimal": lambda v: [Decimal(repr(v))],
    "0-d object array of a Decimal": lambda v: np.array(Decimal(repr(v)), dtype=object),
    "object array of a Decimal": lambda v: np.array([Decimal(repr(v))], dtype=object),
    "1x1 matrix": lambda v: _matrix([[v]]),
    "1x1 float32 matrix": lambda v: _matrix([[v]], dtype=np.float32),
    "list of a 1x1 matrix": lambda v: [_matrix([[v]])],
}


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("form", list(REAL_FORMS))
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_every_form_of_a_real_loss_is_used(entry, protocol, form):
    wrap = REAL_FORMS[form]
    fitter = protocol(*_problem(), loss=lambda params: wrap(sum_of_squares(params)))
    drive(entry, fitter, 30)
    assert fitter.calls.count("finish") == 1
    assert fitter.calls[-1] == "finish"
    seen = [float(np.asarray(loss).reshape(-1)[0]) for loss in fitter.losses]
    assert same_params(fitter.finished_with, fitter.evaluated[int(np.argmin(seen))])


ARRAY_API_REAL = {
    "0-d float64": lambda xp, v: xp.asarray(v),
    "0-d float32": lambda xp, v: xp.asarray(v, dtype=xp.float32),
    "0-d int64": lambda xp, v: xp.asarray(round(1000 * v)),
    "one-element array": lambda xp, v: xp.asarray([v]),
    "1x1 array": lambda xp, v: xp.asarray([[v]]),
    "list of a 0-d array": lambda xp, v: [xp.asarray(v)],
}


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("form", list(ARRAY_API_REAL))
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_real_loss_from_an_array_api_library_is_used(entry, protocol, form):
    xp = pytest.importorskip("array_api_strict")
    wrap = ARRAY_API_REAL[form]
    fitter = protocol(*_problem(), loss=lambda params: wrap(xp, sum_of_squares(params)))
    drive(entry, fitter, 30)
    assert fitter.calls[-1] == "finish"
    seen = [float(np.asarray(loss).reshape(-1)[0]) for loss in fitter.losses]
    assert same_params(fitter.finished_with, fitter.evaluated[int(np.argmin(seen))])


ARRAY_API_NOT_REAL = {
    "0-d bool": lambda xp: xp.asarray(True),
    "0-d complex": lambda xp: xp.asarray(0.5 + 0j),
    "two elements": lambda xp: xp.asarray([0.5, 0.25]),
}


@pytest.mark.parametrize("form", list(ARRAY_API_NOT_REAL))
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_an_array_api_loss_that_is_not_one_real_number_is_an_error(
    entry, protocol, form
):
    xp = pytest.importorskip("array_api_strict")
    loss = ARRAY_API_NOT_REAL[form](xp)
    evaluate, _ = protocol_names(protocol)
    fitter = protocol(*_problem(), loss=lambda params: loss)
    with pytest.raises(TypeError, match="loss"):
        drive(entry, fitter, 60)
    assert fitter.calls.count(evaluate) == 1
    assert "finish" not in fitter.calls


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("refusal", [TypeError, RuntimeError])
@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_loss_numpy_cannot_read_is_read_with_float(entry, protocol, refusal):
    def loss(params):
        return _DeviceScalar(sum_of_squares(params), refusal=refusal)

    fitter = protocol(*_problem(), loss=loss)
    drive(entry, fitter, 30)
    assert fitter.calls[-1] == "finish"
    seen = [loss.value for loss in fitter.losses]
    assert same_params(fitter.finished_with, fitter.evaluated[int(np.argmin(seen))])


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_nan_loss_scores_as_the_worst_value(entry, protocol):
    def loss(params):
        if params["cell"]["a"] > 1.0:
            return float("nan")
        return sum_of_squares(params)

    fitter = protocol(*_problem(), loss=loss)
    drive(entry, fitter, 80)
    assert fitter.calls[-1] == "finish"
    assert fitter.finished_with["cell"]["a"] <= 1.0


def _chemfit_fitter_type():
    return pytest.importorskip("chemfit.fitter").Fitter


@pytest.mark.parametrize("driver", ["portfolio", "boltzmann"])
@pytest.mark.parametrize("entry", ENTRIES)
def test_the_installed_chemfit_fitter_is_driven_through_its_protocol(entry, driver):
    fitter_type = _chemfit_fitter_type()
    calls = []

    def objective(params):
        x = np.asarray(params["x"])
        calls.append(x.copy())
        return float(np.sum((x - 0.3) ** 2))

    fitter = fitter_type(
        objective,
        initial_params={"x": np.array([0.9, -0.6, 0.2])},
        bounds={"x": (-1.0, 1.0)},
    )
    steps = []
    fitter.register_callback(lambda step, contexts: steps.append(step), 1)
    out = drive(entry, fitter, 80, driver=driver)
    assert 0 < len(calls) <= 80
    assert fitter.contexts[0].n_evals == len(calls)
    assert steps
    assert np.array_equal(out["x"], fitter.contexts[0].opt_params["x"])


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_objective_error_inside_chemfit_reaches_the_caller(entry):
    fitter_type = _chemfit_fitter_type()
    error = RuntimeError("objective failed")
    calls = []

    def objective(params):
        calls.append(1)
        if len(calls) == 3:
            raise error
        return float(np.sum(np.asarray(params["x"]) ** 2))

    fitter = fitter_type(
        objective,
        initial_params={"x": np.array([0.4, -0.3])},
        bounds={"x": (-1.0, 1.0)},
        log_exceptions=False,
    )
    with pytest.raises(RuntimeError) as caught:
        drive(entry, fitter, 60)
    assert caught.value is error
    assert len(calls) == 3
