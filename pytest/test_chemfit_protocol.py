"""Both ChemFit session protocols through every bridge.

Current ChemFit drives a ``Fitter`` with ``evaluate`` / ``step``; ChemFit 3.1
with ``ask`` / ``tell``. Every bridge must speak whichever pair the fitter
has, and refuse a fitter with neither before starting its session.
"""

import warnings

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
)


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


@pytest.mark.parametrize("fitter_type", [Recorder, _EvaluateWithoutStep])
@pytest.mark.parametrize("entry", ENTRIES)
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
    assert calls
    assert fitter.contexts[0].n_evals == len(calls)
    assert steps
    assert np.array_equal(out["x"], fitter.contexts[0].opt_params["x"])
