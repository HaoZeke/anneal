"""Calls anneal 0.10.0 accepted, and what each of them does now.

An argument 0.10.0 took and ignored still runs. It gives one FutureWarning,
attributed to the caller, that says what to pass instead, and the fit is the
one the call that passes that gives. ``run_benchmark`` still drives a fitter
with half a session protocol, as 0.10.0 did, with a FutureWarning too.

An argument 0.10.0 coerced, or dropped without a word, now raises an error
that names it before the fitter's session starts.
"""

import re
import warnings

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Boltzmann, Fast  # noqa: E402
from anneal.chemfit import (  # noqa: E402
    ChemFitVector,
    chemfit_box,
    fit_anneal,
    fit_chemfit,
    flatten_parameters,
    resolve_bounds,
    run_benchmark,
    run_fitter,
)
from chemfit_doubles import (  # noqa: E402
    NextFitter,
    Recorder,
    ReleasedFitter,
    protocol_names,
    same_params,
)


def _fitter(kind=ReleasedFitter):
    return kind(
        {"positions": np.array([[0.5, -0.5, 0.25], [1.0, -1.0, 0.0]]), "eps": 0.5},
        {"positions": (-2.0, 2.0), "eps": (0.0, 1.0)},
    )


def _tall_fitter():
    return ReleasedFitter(
        {"positions": np.random.default_rng(0).uniform(-1, 1, (4, 3)), "eps": 0.5},
        {"positions": (-3.0, 3.0), "eps": (0.0, 1.0)},
    )


def _context(fitter, budget=60):
    return {
        "fitter": fitter,
        "budget": budget,
        "initial_params": fitter.initial_parameters,
    }


def _shape_warning(name, given, shape):
    return re.escape(
        f"x0 {name} has shape {given}, but the parameter has shape {shape}"
    )


_START = np.linspace(-0.9, 0.9, 12)
_SWEEP = {"t_init": 2.0, "sigma": 0.4, "gamma": 0.6, "q_v": 2.5, "q_a": 1.5}
_OWN = {
    "boltzmann": ("t_init", "sigma"),
    "fast": ("t_init", "gamma"),
    "gsa": ("t_init", "q_v", "q_a"),
}

# (id, fitter, the call 0.10.0 took, the call that passes what it meant, warning)
WARNS = [
    (
        "fit_anneal preset_kwargs under the portfolio",
        _fitter,
        lambda f: fit_anneal(f, 60, preset_kwargs={"t_init": 2.0}),
        lambda f: fit_anneal(f, 60),
        "fit_anneal ignores preset_kwargs under the portfolio driver",
    ),
    (
        "fit_chemfit t_init under the portfolio",
        _fitter,
        lambda f: fit_chemfit(f, 60, t_init=2.0),
        lambda f: fit_chemfit(f, 60),
        "fit_chemfit ignores t_init under method 'portfolio'",
    ),
    (
        "fit_chemfit gamma under boltzmann",
        _fitter,
        lambda f: fit_chemfit(f, 60, method="boltzmann", gamma=0.5),
        lambda f: fit_chemfit(f, 60, method="boltzmann"),
        (
            "fit_chemfit ignores gamma, which method 'boltzmann' does not take; "
            "it takes t_init, sigma"
        ),
    ),
    (
        "fit_chemfit sweep keywords under the portfolio",
        _fitter,
        lambda f: fit_chemfit(f, 60, **_SWEEP),
        lambda f: fit_chemfit(f, 60),
        "fit_chemfit ignores t_init, sigma, gamma, q_v, q_a under method 'portfolio'",
    ),
    *[
        (
            f"fit_chemfit sweep keywords under {method}",
            _fitter,
            lambda f, m=method: fit_chemfit(
                f, 60, method=m, steps_per_epoch=10, **_SWEEP
            ),
            lambda f, m=method: fit_chemfit(
                f, 60, method=m, steps_per_epoch=10, **{k: _SWEEP[k] for k in _OWN[m]}
            ),
            (
                f"which method '{method}' does not take; "
                f"it takes {', '.join(_OWN[method])}"
            ),
        )
        for method in _OWN
    ],
    (
        "fit_anneal x0 leaf (3, 2) for (2, 3)",
        _fitter,
        lambda f: fit_anneal(
            f, 60, x0={"positions": _START[:6].reshape(3, 2), "eps": 0.5}
        ),
        lambda f: fit_anneal(
            f, 60, x0={"positions": _START[:6].reshape(2, 3), "eps": 0.5}
        ),
        _shape_warning("positions", (3, 2), (2, 3)),
    ),
    (
        "fit_anneal x0 leaf (12,) for (4, 3)",
        _tall_fitter,
        lambda f: fit_anneal(f, 60, x0={"positions": _START, "eps": 0.5}),
        lambda f: fit_anneal(f, 60, x0={"positions": _START.reshape(4, 3), "eps": 0.5}),
        _shape_warning("positions", (12,), (4, 3)),
    ),
    (
        "fit_anneal x0 leaf (3, 4) for (4, 3)",
        _tall_fitter,
        lambda f: fit_anneal(f, 60, x0={"positions": _START.reshape(3, 4), "eps": 0.5}),
        lambda f: fit_anneal(f, 60, x0={"positions": _START.reshape(4, 3), "eps": 0.5}),
        _shape_warning("positions", (3, 4), (4, 3)),
    ),
    (
        "fit_anneal x0 leaf [0.3] for a scalar",
        _tall_fitter,
        lambda f: fit_anneal(
            f, 60, x0={"positions": _START.reshape(4, 3), "eps": np.array([0.3])}
        ),
        lambda f: fit_anneal(f, 60, x0={"positions": _START.reshape(4, 3), "eps": 0.3}),
        _shape_warning("eps", (1,), ()),
    ),
    (
        "run_benchmark method sa with Boltzmann()",
        _fitter,
        lambda f: run_benchmark(_context(f), method="sa", preset=Boltzmann()),
        lambda f: run_benchmark(_context(f), method="boltzmann", preset=Boltzmann()),
        (
            "run_benchmark has no method 'sa'; it runs the Boltzmann preset as "
            "method 'boltzmann'"
        ),
    ),
    (
        "run_benchmark boltzmann with Fast()",
        _fitter,
        lambda f: run_benchmark(_context(f), method="boltzmann", preset=Fast()),
        lambda f: run_benchmark(_context(f), method="fast", preset=Fast()),
        (
            "run_benchmark runs the Fast preset it was given, not method "
            "'boltzmann'; pass method='fast' with it"
        ),
    ),
    (
        "run_benchmark portfolio with Boltzmann()",
        _fitter,
        lambda f: run_benchmark(_context(f), preset=Boltzmann()),
        lambda f: run_benchmark(_context(f)),
        "run_benchmark ignores the Boltzmann preset under method 'portfolio'",
    ),
    (
        "run_fitter global_optimize with Boltzmann()",
        _fitter,
        lambda f: run_fitter(f, 60, preset=Boltzmann()),
        lambda f: run_fitter(f, 60),
        (
            "run_fitter ignores the Boltzmann preset under method "
            "'global_optimize', which runs the portfolio"
        ),
    ),
    (
        "run_fitter fast with Boltzmann()",
        _fitter,
        lambda f: run_fitter(f, 60, method="fast", preset=Boltzmann()),
        lambda f: run_fitter(f, 60, method="fast"),
        (
            "run_fitter ignores the Boltzmann preset under method 'fast', which "
            "runs its own fast preset"
        ),
    ),
    (
        "run_fitter global_optimize with preset_kwargs",
        _fitter,
        lambda f: run_fitter(f, 60, preset_kwargs={"t_init": 2.0}),
        lambda f: run_fitter(f, 60),
        "run_fitter ignores preset_kwargs under method 'global_optimize'",
    ),
]


@pytest.mark.parametrize(
    "make, deprecated, supported, warning",
    [case[1:] for case in WARNS],
    ids=[case[0] for case in WARNS],
)
def test_an_argument_0_10_0_ignored_still_runs_with_a_future_warning(
    make, deprecated, supported, warning
):
    fitter = make()
    with pytest.warns(FutureWarning, match=warning) as record:
        out = deprecated(fitter)
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert future[0].filename == __file__
    assert "will raise in a future release" in str(future[0].message)
    assert fitter.calls[0] == "init"
    assert fitter.calls[-1] == "finish" and fitter.calls.count("finish") == 1

    plain = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        expected = supported(plain)
    assert len(fitter.evaluated) == len(plain.evaluated) > 1
    for got, want in zip(fitter.evaluated, plain.evaluated):
        assert same_params(got, want)
    assert same_params(out, expected)


class _EvaluateOnly(Recorder):
    def evaluate(self, parameters, context_index=0):
        return self._evaluate("evaluate", parameters)


class _AskOnly(Recorder):
    def ask(self, parameters, context_index=0):
        return self._evaluate("ask", parameters)


@pytest.mark.parametrize(
    "half, whole", [(_EvaluateOnly, NextFitter), (_AskOnly, ReleasedFitter)]
)
def test_run_benchmark_still_drives_half_a_protocol_with_a_future_warning(half, whole):
    evaluate, step = protocol_names(whole)
    fitter = _fitter(half)
    warning = (
        f"run_benchmark drives {half.__name__} through {evaluate} with no step "
        f"notices, since it has no {step}; give it a {step} method"
    )
    with pytest.warns(FutureWarning, match=warning) as record:
        out = run_benchmark(_context(fitter))
    assert record[0].filename == __file__
    assert fitter.calls[0] == "init" and fitter.calls[-1] == "finish"
    assert set(fitter.calls) == {"init", evaluate, "finish"}

    full = _fitter(whole)
    run_benchmark(_context(full))
    assert len(fitter.evaluated) == len(full.evaluated) > 1
    for got, want in zip(fitter.evaluated, full.evaluated):
        assert same_params(got, want)
    assert same_params(out, full.finished_with)


def _with(initial, bounds):
    return lambda: ReleasedFitter(initial, bounds)


def _box(fitter):
    return chemfit_box(fitter, ChemFitVector(fitter.initial_parameters))


_WHOLE = "{} must be a whole number, got {}"
_NOT_A_PRESET = "preset must be Boltzmann(), Fast() or Gsa(), got str"
_PASSES = (
    "it passes only x0, low, high, bound_span, steps_per_epoch and preset_kwargs "
    "on to fit_anneal"
)
_NOT_A_PAIR = "the bounds of x must be a (lower, upper) pair"
_THREE_PAIRS = [(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]


def _raises(label, call, error, message, make=_fitter):
    """A call 0.10.0 took, with the error it raises now and the message."""
    return pytest.param(make, call, error, message, id=label)


RAISES = [
    # 0.10.0 passed budget and seed through int().
    _raises(
        "fit_anneal budget=60.5",
        lambda f: fit_anneal(f, 60.5),
        ValueError,
        _WHOLE.format("budget", 60.5),
    ),
    _raises(
        "fit_anneal budget='60'",
        lambda f: fit_anneal(f, "60"),
        TypeError,
        _WHOLE.format("budget", "'60'"),
    ),
    _raises(
        "fit_chemfit budget=60.5",
        lambda f: fit_chemfit(f, 60.5),
        ValueError,
        _WHOLE.format("budget", 60.5),
    ),
    _raises(
        "run_benchmark budget=60.5",
        lambda f: run_benchmark(_context(f, 60.5)),
        ValueError,
        _WHOLE.format("budget", 60.5),
    ),
    _raises(
        "run_fitter budget=60.5",
        lambda f: run_fitter(f, 60.5),
        ValueError,
        _WHOLE.format("budget", 60.5),
    ),
    _raises(
        "fit_anneal seed=1.5",
        lambda f: fit_anneal(f, 60, seed=1.5),
        ValueError,
        _WHOLE.format("seed", 1.5),
    ),
    _raises(
        "fit_anneal seed='7'",
        lambda f: fit_anneal(f, 60, seed="7"),
        TypeError,
        _WHOLE.format("seed", "'7'"),
    ),
    _raises(
        "run_fitter seed=1.5",
        lambda f: run_fitter(f, 60, seed=1.5),
        ValueError,
        _WHOLE.format("seed", 1.5),
    ),
    # 0.10.0 raised OverflowError after init.
    _raises(
        "fit_anneal seed=-1",
        lambda f: fit_anneal(f, 60, seed=-1),
        ValueError,
        "seed must be at least 0, got -1",
    ),
    _raises(
        "fit_chemfit seed=2**64",
        lambda f: fit_chemfit(f, 60, seed=2**64),
        ValueError,
        f"seed must be at most {2**64 - 1}",
    ),
    # 0.10.0 took max(1, int(value)).
    _raises(
        "fit_chemfit tell_every=0",
        lambda f: fit_chemfit(f, 60, tell_every=0),
        ValueError,
        "tell_every must be positive, got 0",
    ),
    _raises(
        "fit_chemfit tell_every=2.5",
        lambda f: fit_chemfit(f, 60, tell_every=2.5),
        ValueError,
        _WHOLE.format("tell_every", 2.5),
    ),
    _raises(
        "fit_chemfit fast steps_per_epoch=0",
        lambda f: fit_chemfit(f, 60, method="fast", steps_per_epoch=0),
        ValueError,
        "steps_per_epoch must be positive, got 0",
    ),
    _raises(
        "fit_chemfit boltzmann steps_per_epoch=0",
        lambda f: fit_chemfit(f, 60, method="boltzmann", steps_per_epoch=0),
        ValueError,
        "steps_per_epoch must be positive, got 0",
    ),
    _raises(
        "fit_anneal boltzmann steps_per_epoch=0",
        lambda f: fit_anneal(f, 60, driver="boltzmann", steps_per_epoch=0),
        ValueError,
        "steps_per_epoch must be positive, got 0",
    ),
    _raises(
        "fit_anneal portfolio steps_per_epoch=0",
        lambda f: fit_anneal(f, 60, steps_per_epoch=0),
        ValueError,
        "steps_per_epoch must be positive, got 0",
    ),
    _raises(
        "fit_anneal boltzmann steps_per_epoch=2.5",
        lambda f: fit_anneal(f, 60, driver="boltzmann", steps_per_epoch=2.5),
        ValueError,
        _WHOLE.format("steps_per_epoch", 2.5),
    ),
    # 0.10.0 ignored these, or passed them through float().
    _raises(
        "fit_chemfit t_inti=5",
        lambda f: fit_chemfit(f, 60, method="boltzmann", t_inti=5.0),
        TypeError,
        "fit_chemfit() got an unexpected keyword argument 't_inti'; "
        "its preset keywords are t_init, sigma, gamma, q_v and q_a",
    ),
    _raises(
        "fit_chemfit tell_evry=5",
        lambda f: fit_chemfit(f, 60, tell_evry=5),
        TypeError,
        "fit_chemfit() got an unexpected keyword argument 'tell_evry'",
    ),
    _raises(
        "fit_chemfit t_init='2.0'",
        lambda f: fit_chemfit(f, 60, method="boltzmann", t_init="2.0"),
        TypeError,
        "t_init must be a number, got '2.0'",
    ),
    _raises(
        "fit_anneal preset_kwargs='t_init=2'",
        lambda f: fit_anneal(f, 60, preset_kwargs="t_init=2"),
        TypeError,
        "preset_kwargs must be a dict, got str",
    ),
    _raises(
        "run_benchmark preset='hot'",
        lambda f: run_benchmark(_context(f), preset="hot"),
        TypeError,
        _NOT_A_PRESET,
    ),
    _raises(
        "run_fitter preset='hot'",
        lambda f: run_fitter(f, 60, preset="hot"),
        TypeError,
        _NOT_A_PRESET,
    ),
    # 0.10.0 dropped every keyword run_fitter was given.
    _raises(
        "run_fitter bogus=1",
        lambda f: run_fitter(f, 60, bogus=1),
        TypeError,
        f"run_fitter() got an unexpected keyword argument 'bogus'; {_PASSES}",
    ),
    _raises(
        "run_fitter tell_every=5",
        lambda f: run_fitter(f, 60, tell_every=5),
        TypeError,
        "run_fitter() got an unexpected keyword argument 'tell_every'; "
        f"{_PASSES} (tell_every is an option of fit_chemfit)",
    ),
    _raises(
        "run_fitter n_epochs=3",
        lambda f: run_fitter(f, 60, n_epochs=3),
        TypeError,
        f"run_fitter() got an unexpected keyword argument 'n_epochs'; {_PASSES}",
    ),
    _raises(
        "run_fitter x0 of length 5",
        lambda f: run_fitter(f, 60, x0=np.zeros(5)),
        ValueError,
        "x0 has length 5 but the parameters flatten to 7",
    ),
    # 0.10.0 ran the preset, or raised KeyError after init.
    _raises(
        "run_benchmark method='xyz' with Boltzmann()",
        lambda f: run_benchmark(_context(f), method="xyz", preset=Boltzmann()),
        ValueError,
        "unknown method 'xyz'",
    ),
    _raises(
        "run_benchmark method='sa' with no preset",
        lambda f: run_benchmark(_context(f), method="sa"),
        ValueError,
        "unknown method 'sa'",
    ),
    # 0.10.0 read these bounds and parameters without a check.
    _raises(
        "fit_chemfit bounds entry of three items",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        f"{_NOT_A_PAIR}, got (0.0, 1.0, 'extra')",
        make=_with({"x": 0.5, "y": 0.1}, {"x": (0.0, 1.0, "extra"), "y": (-1.0, 1.0)}),
    ),
    _raises(
        "fit_chemfit three per-element pairs",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        _NOT_A_PAIR,
        make=_with({"x": np.zeros(3)}, {"x": _THREE_PAIRS}),
    ),
    _raises(
        "chemfit_box three per-element pairs",
        _box,
        ValueError,
        _NOT_A_PAIR,
        make=_with({"x": np.zeros(3)}, {"x": _THREE_PAIRS}),
    ),
    _raises(
        "chemfit_box two per-element pairs",
        _box,
        ValueError,
        "the bounds of x[0] are empty, the lower above the upper: "
        "lower=-1.0, upper=-2.0",
        make=_with({"x": np.zeros(2)}, {"x": [(-1, 1), (-2, 2)]}),
    ),
    _raises(
        "fit_chemfit NaN parameter",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "parameter a must be finite",
        make=_with({"a": float("nan"), "b": 0.5}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}),
    ),
    _raises(
        "ChemFitVector NaN parameter",
        lambda f: ChemFitVector({"a": float("nan")}),
        ValueError,
        "parameter a must be finite",
    ),
    _raises(
        "resolve_bounds low > high",
        lambda f: resolve_bounds({"x": np.zeros(2)}, low=1.0, high=-1.0),
        ValueError,
        "the bounds of x[0] are empty, the lower above the upper",
    ),
    _raises(
        "flatten_parameters complex leaf",
        lambda f: flatten_parameters({"x": np.array([1 + 1j])}),
        ValueError,
        "parameter x is not real-numeric",
    ),
    _raises(
        "fit_anneal complex leaf",
        lambda f: fit_anneal(f, 60),
        ValueError,
        "parameter x is not real-numeric",
        make=_with({"x": np.array([1 + 1j, 0.5])}, {"x": (-2.0, 2.0)}),
    ),
]


@pytest.mark.parametrize("make, call, error, message", RAISES)
def test_an_argument_0_10_0_coerced_or_dropped_now_raises_before_init(
    make, call, error, message
):
    fitter = make()
    with pytest.raises(error, match=re.escape(message)):
        call(fitter)
    assert fitter.calls == []


# Calls 0.10.0 refused, each with the call it now matches.
NOW_ACCEPTED = [
    pytest.param(
        lambda f: fit_chemfit(f, 60, method="Portfolio"),
        lambda f: fit_chemfit(f, 60),
        id="fit_chemfit method='Portfolio'",
    ),
    pytest.param(
        lambda f: fit_chemfit(f, 60, method="GSA"),
        lambda f: fit_chemfit(f, 60, method="gsa"),
        id="fit_chemfit method='GSA'",
    ),
    pytest.param(
        lambda f: run_fitter(f, 60, method="SA"),
        lambda f: run_fitter(f, 60, method="sa"),
        id="run_fitter method='SA'",
    ),
    pytest.param(
        lambda f: run_fitter(f, 60, method="Global_Optimize"),
        lambda f: run_fitter(f, 60),
        id="run_fitter method='Global_Optimize'",
    ),
]


@pytest.mark.parametrize("call, same_as", NOW_ACCEPTED)
def test_fit_chemfit_and_run_fitter_read_method_names_in_any_case(call, same_as):
    fitter, plain = _fitter(), _fitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = call(fitter)
        expected = same_as(plain)
    assert len(fitter.evaluated) == len(plain.evaluated) > 1
    for got, want in zip(fitter.evaluated, plain.evaluated):
        assert same_params(got, want)
    assert same_params(out, expected)
