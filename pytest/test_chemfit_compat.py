"""Calls anneal 0.10.0 accepted, and what each of them does now.

An argument 0.10.0 took and ignored still runs. It gives one FutureWarning,
attributed to the caller, that says what to pass instead, and the fit is the
one the call that passes that gives. So does a numeric string where 0.10.0
read a parameter value or a bound as a number. ``run_benchmark`` still
drives a fitter with half a session protocol, as 0.10.0 did, with a
FutureWarning too.

An argument 0.10.0 coerced, or dropped without a word, now raises an error
that names it before the fitter's session starts. Bounds 0.10.0 ignored in
favour of others, such as a NumPy pair in ``run_benchmark``'s context
bounds, are now read.
"""

import re
import warnings
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Boltzmann, Fast  # noqa: E402
from anneal.chemfit import (  # noqa: E402
    ChemFitVector,
    bounds_from_fitter,
    chemfit_box,
    fit_anneal,
    fit_chemfit,
    flatten_parameters,
    resolve_bounds,
    run_benchmark,
    run_fitter,
    unflatten_parameters,
)
from chemfit_doubles import (  # noqa: E402
    ENTRIES,
    NextFitter,
    Recorder,
    ReleasedFitter,
    drive,
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


def _with(initial, bounds):
    return lambda: ReleasedFitter(initial, bounds)


_ONLY = _with({"x": np.array([0.9, -0.6, 0.2])}, {"x": (-1.0, 1.0)})


def _context(fitter, budget=60):
    return {
        "fitter": fitter,
        "budget": budget,
        "initial_params": fitter.initial_parameters,
    }


def _as(fitter, bounds=None, **leaves):
    """``fitter`` with some initial parameters, or its bounds, replaced."""
    fitter.initial_parameters = {**fitter.initial_parameters, **leaves}
    if bounds is not None:
        fitter.bounds = bounds
    return fitter


def _paired(fitter, kind=list):
    """``fitter`` with its initial parameters given as (key, value) pairs."""
    fitter.initial_parameters = kind(fitter.initial_parameters.items())
    return fitter


def _reads(caller, what, one=False):
    """The start of the warning for the numeric strings ``caller`` reads."""
    source, number = ("a string", "a number") if one else ("strings", "numbers")
    return re.escape(f"{caller} reads {what} from {source}; pass {number} instead")


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
_TAKES = {
    "boltzmann": "t_init and sigma",
    "fast": "t_init and gamma",
    "gsa": "t_init, q_v and q_a",
}
# PyYAML reads [-2e0, 2e0] as two strings, and [0, 1e0] as 0 and a string.
_YAML = {"positions": ["-2e0", "2e0"], "eps": [0, "1e0"]}
_YAML_READS = (
    "the lower bound of positions, the upper bound of positions and the upper "
    "bound of eps"
)
_POSITIONS = np.array([[0.5, -0.5, 0.25], [1.0, -1.0, 0.0]])
_POSITIONS_AS_TEXT = _POSITIONS.astype(str)


def _objects(values, item=float):
    """An object array of ``values``' shape holding ``item(value)`` for each."""
    out = np.empty(np.shape(values), dtype=object)
    for index, value in np.ndenumerate(np.asarray(values)):
        out[index] = item(value)
    return out


def _decimal(value):
    return Decimal(repr(float(value)))


def _matrix(rows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix(rows)


def _xp():
    return pytest.importorskip("array_api_strict")


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
        (
            "fit_chemfit ignores t_init under method 'portfolio', which takes no "
            "preset; leave it out"
        ),
    ),
    (
        "fit_chemfit gamma under boltzmann",
        _fitter,
        lambda f: fit_chemfit(f, 60, method="boltzmann", gamma=0.5),
        lambda f: fit_chemfit(f, 60, method="boltzmann"),
        (
            "fit_chemfit ignores gamma, which method 'boltzmann' does not take; "
            "it takes t_init and sigma"
        ),
    ),
    (
        "fit_chemfit sweep keywords under the portfolio",
        _fitter,
        lambda f: fit_chemfit(f, 60, **_SWEEP),
        lambda f: fit_chemfit(f, 60),
        (
            "fit_chemfit ignores t_init, sigma, gamma, q_v and q_a under method "
            "'portfolio', which takes no preset; leave them out"
        ),
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
                f"it takes {_TAKES[method]}. Pass only those"
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
    # 0.10.0 filled the only parameter with one x0 value where the fitter
    # bounds it on both sides.
    (
        "fit_anneal x0 one value for the only parameter (3,)",
        _ONLY,
        lambda f: fit_anneal(f, 60, x0={"x": 0.4}),
        lambda f: fit_anneal(f, 60, x0={"x": np.full(3, 0.4)}),
        _shape_warning("x", (), (3,)) + ".*its one value fills the parameter",
    ),
    (
        "fit_anneal x0 [0.4] for the only parameter (2, 3)",
        _with({"x": np.zeros((2, 3))}, {"x": (-1.0, 1.0)}),
        lambda f: fit_anneal(f, 60, driver="fast", x0={"x": [0.4]}),
        lambda f: fit_anneal(f, 60, driver="fast", x0={"x": np.full((2, 3), 0.4)}),
        _shape_warning("x", (1,), (2, 3)) + ".*its one value fills the parameter",
    ),
    (
        "run_fitter x0 one value for the only parameter (3,)",
        _ONLY,
        lambda f: run_fitter(f, 60, x0={"x": 0.4}),
        lambda f: run_fitter(f, 60, x0={"x": np.full(3, 0.4)}),
        _shape_warning("x", (), (3,)) + ".*its one value fills the parameter",
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
    # 0.10.0 read a numeric string wherever it read a parameter value or a
    # bound.
    (
        "fit_anneal string parameter",
        _fitter,
        lambda f: fit_anneal(_as(f, eps="0.5"), 60),
        lambda f: fit_anneal(f, 60),
        _reads("fit_anneal", "parameter eps", one=True),
    ),
    (
        "fit_chemfit string parameter",
        _fitter,
        lambda f: fit_chemfit(_as(f, eps="0.5"), 60),
        lambda f: fit_chemfit(f, 60),
        _reads("fit_chemfit", "parameter eps", one=True),
    ),
    (
        "run_benchmark string parameters",
        _fitter,
        lambda f: run_benchmark(
            {
                **_context(f),
                "initial_params": {"positions": _POSITIONS_AS_TEXT, "eps": "0.5"},
            }
        ),
        lambda f: run_benchmark(_context(f)),
        _reads("run_benchmark", "parameter positions and parameter eps"),
    ),
    (
        "run_fitter string parameter",
        _fitter,
        lambda f: run_fitter(_as(f, eps="0.5"), 60),
        lambda f: run_fitter(f, 60),
        _reads("run_fitter", "parameter eps", one=True),
    ),
    (
        "fit_chemfit bounds read from YAML",
        _fitter,
        lambda f: fit_chemfit(_as(f, bounds=_YAML), 60),
        lambda f: fit_chemfit(f, 60),
        _reads("fit_chemfit", _YAML_READS),
    ),
    (
        "fit_anneal bounds read from YAML",
        _fitter,
        lambda f: fit_anneal(_as(f, bounds=_YAML), 60),
        lambda f: fit_anneal(f, 60),
        _reads("fit_anneal", _YAML_READS),
    ),
    # 0.10.0's fit_chemfit read a tuple of numeric strings but searched the
    # start +/- default_span in place of a NumPy pair of them.
    (
        "fit_chemfit bounds entry that is a NumPy pair of strings",
        _fitter,
        lambda f: fit_chemfit(
            _as(f, {**_NARROW, "positions": np.array(["-0.5", "5e-1"])}), 60
        ),
        lambda f: fit_chemfit(_as(f, _NARROW), 60),
        _reads(
            "fit_chemfit",
            "the lower bound of positions and the upper bound of positions",
        ),
    ),
    (
        "run_benchmark context bounds read from YAML",
        _fitter,
        lambda f: run_benchmark({**_context(f), "bounds": _YAML}),
        lambda f: run_benchmark(_context(f)),
        _reads("run_benchmark", _YAML_READS),
    ),
    (
        "run_benchmark context low and high strings",
        _fitter,
        lambda f: run_benchmark({**_context(f), "bounds": {"low": "-1", "high": "1"}}),
        lambda f: run_benchmark({**_context(f), "bounds": {"low": -1.0, "high": 1.0}}),
        _reads("run_benchmark", 'bounds["low"] and bounds["high"]'),
    ),
    (
        "run_benchmark low and high strings",
        _fitter,
        lambda f: run_benchmark(_context(f), low="-1", high="1"),
        lambda f: run_benchmark(_context(f), low=-1.0, high=1.0),
        _reads("run_benchmark", "low and high"),
    ),
    (
        "fit_anneal low and high strings",
        _fitter,
        lambda f: fit_anneal(f, 60, low=_LOW.astype(str), high=_HIGH.astype(str)),
        lambda f: fit_anneal(f, 60, low=_LOW, high=_HIGH),
        _reads("fit_anneal", "low and high"),
    ),
    (
        "fit_anneal x0 strings",
        _fitter,
        lambda f: fit_anneal(f, 60, x0=[str(v) for v in _START[:7]]),
        lambda f: fit_anneal(f, 60, x0=_START[:7]),
        _reads("fit_anneal", "x0"),
    ),
    # It read the numeric strings in an object array too, item by item.
    (
        "fit_chemfit object array of strings parameter",
        _fitter,
        lambda f: fit_chemfit(_as(f, positions=_objects(_POSITIONS, str)), 60),
        lambda f: fit_chemfit(f, 60),
        _reads("fit_chemfit", "parameter positions"),
    ),
    (
        "fit_anneal object array of strings parameter",
        _fitter,
        lambda f: fit_anneal(_as(f, positions=_objects(_POSITIONS, str)), 60),
        lambda f: fit_anneal(f, 60),
        _reads("fit_anneal", "parameter positions"),
    ),
    (
        "run_benchmark object array of strings parameter",
        _fitter,
        lambda f: run_benchmark(
            {
                **_context(f),
                "initial_params": {"positions": _objects(_POSITIONS, str), "eps": 0.5},
            }
        ),
        lambda f: run_benchmark(_context(f)),
        _reads("run_benchmark", "parameter positions"),
    ),
    (
        "run_fitter object array of strings parameter",
        _fitter,
        lambda f: run_fitter(_as(f, positions=_objects(_POSITIONS, str)), 60),
        lambda f: run_fitter(f, 60),
        _reads("run_fitter", "parameter positions"),
    ),
    (
        "fit_anneal x0 object array of strings",
        _fitter,
        lambda f: fit_anneal(f, 60, x0=_objects(_START[:7], str)),
        lambda f: fit_anneal(f, 60, x0=_START[:7]),
        _reads("fit_anneal", "x0"),
    ),
    (
        "fit_anneal low and high object arrays of strings",
        _fitter,
        lambda f: fit_anneal(f, 60, low=_objects(_LOW, str), high=_objects(_HIGH, str)),
        lambda f: fit_anneal(f, 60, low=_LOW, high=_HIGH),
        _reads("fit_anneal", "low and high"),
    ),
    (
        "fit_anneal bounds sides that are object arrays of strings",
        _fitter,
        lambda f: fit_anneal(
            _as(f, {**_NARROW, "positions": tuple(_objects(s, str) for s in _ROWS)}),
            60,
        ),
        lambda f: fit_anneal(_as(f, _NARROW), 60),
        _reads(
            "fit_anneal",
            "the lower bound of positions and the upper bound of positions",
        ),
    ),
    # 0.10.0's fit_chemfit read initial_parameters given as (key, value)
    # pairs, which ChemFit 3.1's Fitter keeps, as the dict they make.
    (
        "fit_chemfit initial_parameters as a list of pairs",
        _fitter,
        lambda f: fit_chemfit(_paired(f), 60),
        lambda f: fit_chemfit(f, 60),
        re.escape(
            "fit_chemfit reads fitter.initial_parameters, a list of (key, value) "
            "pairs, as a dict; give the fitter a dict"
        ),
    ),
    (
        "fit_chemfit boltzmann initial_parameters as a tuple of pairs",
        _fitter,
        lambda f: fit_chemfit(
            _paired(f, tuple), 60, method="boltzmann", steps_per_epoch=10
        ),
        lambda f: fit_chemfit(f, 60, method="boltzmann", steps_per_epoch=10),
        re.escape("initial_parameters, a tuple of (key, value) pairs, as a dict"),
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


class _EvaluateTell(Recorder):
    def evaluate(self, parameters, context_index=0):
        return self._evaluate("evaluate", parameters)

    def tell(self, step=None):
        self._record("tell")


class _AskStep(Recorder):
    def ask(self, parameters, context_index=0):
        return self._evaluate("ask", parameters)

    def step(self, step=None):
        self._record("step")


@pytest.mark.parametrize("method", ["portfolio", "boltzmann"])
@pytest.mark.parametrize(
    "mixed, whole", [(_EvaluateTell, NextFitter), (_AskStep, ReleasedFitter)]
)
def test_run_benchmark_still_sends_a_mixed_fitter_its_other_step_notice(
    mixed, whole, method
):
    evaluate, step = protocol_names(whole)
    notice = "tell" if step == "step" else "step"
    fitter = _fitter(mixed)
    warning = (
        f"run_benchmark drives {mixed.__name__} through {evaluate} with {notice} "
        f"notices, since it has no {step}; give it a {step} method"
    )
    with pytest.warns(FutureWarning, match=warning) as record:
        out = run_benchmark(_context(fitter), method=method)
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert len(future) == 1 and future[0].filename == __file__

    full = _fitter(whole)
    run_benchmark(_context(full), method=method)
    assert fitter.calls == [notice if call == step else call for call in full.calls]
    assert fitter.calls.count(notice) == len(fitter.evaluated) > 1
    for got, want in zip(fitter.evaluated, full.evaluated):
        assert same_params(got, want)
    assert same_params(out, full.finished_with)


def _same(got, want):
    """Whether two helper results hold the same values, dtype and bits alike."""
    if isinstance(want, (tuple, list)):
        return len(got) == len(want) and all(map(_same, got, want))
    if isinstance(want, np.ndarray):
        return got.dtype == want.dtype and np.array_equal(got, want)
    return got == want


_X = {"x": np.zeros(2)}

# (id, the call with numeric strings, the call with numbers, what it reads)
HELPERS_READ = [
    (
        "flatten_parameters",
        lambda: flatten_parameters({"a": "0.5", "b": ["1", "2e0"]}),
        lambda: flatten_parameters({"a": 0.5, "b": [1.0, 2.0]}),
        _reads("flatten_parameters", "parameter a and parameter b"),
    ),
    (
        "flatten_parameters object array of strings",
        lambda: flatten_parameters({"a": _objects([0.5, 1.0], str)}),
        lambda: flatten_parameters({"a": np.array([0.5, 1.0])}),
        _reads("flatten_parameters", "parameter a"),
    ),
    (
        "ChemFitVector",
        lambda: ChemFitVector({"a": "0.5"}).x0,
        lambda: ChemFitVector({"a": 0.5}).x0,
        _reads("ChemFitVector", "parameter a", one=True),
    ),
    (
        "chemfit_box",
        lambda: _box(ReleasedFitter({"x": 0.5}, {"x": ["0", "1e0"]})),
        lambda: _box(ReleasedFitter({"x": 0.5}, {"x": (0.0, 1.0)})),
        _reads("chemfit_box", "the lower bound of x and the upper bound of x"),
    ),
    (
        "chemfit_box NumPy pair of strings",
        lambda: _box(ReleasedFitter({"x": 0.5}, {"x": np.array(["0", "1e0"])})),
        lambda: _box(ReleasedFitter({"x": 0.5}, {"x": (0.0, 1.0)})),
        _reads("chemfit_box", "the lower bound of x and the upper bound of x"),
    ),
    (
        "resolve_bounds",
        lambda: resolve_bounds(_X, low="-1", high="1"),
        lambda: resolve_bounds(_X, low=-1.0, high=1.0),
        _reads("resolve_bounds", "low and high"),
    ),
    (
        "resolve_bounds object arrays of strings",
        lambda: resolve_bounds(
            _X, low=_objects([-1, -1], str), high=_objects([1, 1], str)
        ),
        lambda: resolve_bounds(_X, low=-1.0, high=1.0),
        _reads("resolve_bounds", "low and high"),
    ),
    (
        "bounds_from_fitter",
        lambda: bounds_from_fitter(_X, {"x": ["-1e0", "1e0"]}, 2),
        lambda: bounds_from_fitter(_X, {"x": (-1.0, 1.0)}, 2),
        _reads("bounds_from_fitter", "the lower bound of x and the upper bound of x"),
    ),
]


@pytest.mark.parametrize(
    "deprecated, supported, warning",
    [case[1:] for case in HELPERS_READ],
    ids=[case[0] for case in HELPERS_READ],
)
def test_the_helpers_read_numeric_strings_with_a_future_warning(
    deprecated, supported, warning
):
    with pytest.warns(FutureWarning, match=warning) as record:
        got = deprecated()
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert future[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        want = supported()
    assert _same(got, want)


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_installed_chemfit_fitter_runs_on_bounds_read_from_yaml(entry):
    fitter_type = pytest.importorskip("chemfit.fitter").Fitter

    def run(bounds):
        calls = []

        def objective(params):
            x = np.asarray(params["x"])
            calls.append(x.copy())
            return float(np.sum((x - 0.3) ** 2))

        fitter = fitter_type(
            objective,
            initial_params={"x": np.array([0.9, -0.6, 0.2])},
            bounds={"x": bounds},
        )
        return drive(entry, fitter, 60), calls

    # PyYAML reads x: [-1e0, 1e0] as two strings, and the Fitter keeps them.
    with pytest.warns(
        FutureWarning, match="from strings; pass numbers instead"
    ) as record:
        out, calls = run(["-1e0", "1e0"])
    assert len([w for w in record if issubclass(w.category, FutureWarning)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        want, want_calls = run((-1.0, 1.0))
    assert len(calls) == len(want_calls) > 1
    for got, expected in zip(calls, want_calls):
        assert np.array_equal(got, expected)
    assert np.array_equal(out["x"], want["x"])


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_installed_chemfit_fitter_runs_on_decimal_and_fraction_parameters(entry):
    fitter_type = pytest.importorskip("chemfit.fitter").Fitter

    def run(initial):
        calls = []

        def objective(params):
            calls.append((params["a"], params["b"]))
            return (params["a"] - 0.3) ** 2 + (params["b"] - 0.2) ** 2

        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        fitter = fitter_type(objective, initial_params=initial, bounds=bounds)
        return drive(entry, fitter, 60), calls

    # The Fitter keeps these as given, and 0.10.0 read them as floats.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out, calls = run({"a": Decimal("0.5"), "b": Fraction(1, 10)})
        want, want_calls = run({"a": 0.5, "b": 0.1})
    assert calls == want_calls and len(calls) > 1
    assert out == want


def test_fit_chemfit_reads_the_pairs_a_chemfit_3_1_fitter_keeps():
    fitter_type = pytest.importorskip("chemfit.fitter").Fitter
    pairs = [("a", 0.5), ("b", 0.1)]
    bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
    try:
        fitter_type(lambda params: 0.0, initial_params=pairs, bounds=bounds)
    except TypeError:
        pytest.skip("this ChemFit refuses initial_params that is not a mapping")

    def run(initial):
        calls = []

        def objective(params):
            calls.append((params["a"], params["b"]))
            return (params["a"] - 0.3) ** 2 + (params["b"] - 0.2) ** 2

        fitter = fitter_type(objective, initial_params=initial, bounds=bounds)
        return fit_chemfit(fitter, 60), calls

    with pytest.warns(
        FutureWarning, match=re.escape("a list of (key, value) pairs, as a dict")
    ) as record:
        out, calls = run(pairs)
    assert len([w for w in record if issubclass(w.category, FutureWarning)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        want, want_calls = run(dict(pairs))
    assert calls == want_calls and len(calls) > 1
    assert out == want


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
_SPAN = "{} must be positive and finite, got {}"
_LOW, _HIGH = np.full(7, -1.0), np.full(7, 1.0)
_UNBOUNDED = _with({"x": 0.5, "y": 0.1}, {})
_NOT_FINITE = "the bounds of x[0] must be finite: lower={}, upper={}"


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
    _raises(
        "fit_anneal budget=Decimal('60.5')",
        lambda f: fit_anneal(f, Decimal("60.5")),
        ValueError,
        _WHOLE.format("budget", "Decimal('60.5')"),
    ),
    _raises(
        "fit_chemfit seed in a 0-d array of 7.5",
        lambda f: fit_chemfit(f, 60, seed=np.array(7.5)),
        ValueError,
        _WHOLE.format("seed", "array(7.5)"),
    ),
    # 0.10.0 read a bool as 1 wherever it read a number.
    _raises(
        "fit_anneal budget=True",
        lambda f: fit_anneal(f, True),
        TypeError,
        _WHOLE.format("budget", True),
    ),
    _raises(
        "run_benchmark budget=True",
        lambda f: run_benchmark(_context(f, True)),
        TypeError,
        _WHOLE.format("budget", True),
    ),
    _raises(
        "fit_anneal seed=True",
        lambda f: fit_anneal(f, 60, seed=True),
        TypeError,
        _WHOLE.format("seed", True),
    ),
    _raises(
        "fit_chemfit seed=True",
        lambda f: fit_chemfit(f, 60, seed=True),
        TypeError,
        _WHOLE.format("seed", True),
    ),
    _raises(
        "run_fitter seed=True",
        lambda f: run_fitter(f, 60, seed=True),
        TypeError,
        _WHOLE.format("seed", True),
    ),
    _raises(
        "fit_anneal boltzmann steps_per_epoch=True",
        lambda f: fit_anneal(f, 60, driver="boltzmann", steps_per_epoch=True),
        TypeError,
        _WHOLE.format("steps_per_epoch", True),
    ),
    _raises(
        "fit_chemfit tell_every=True",
        lambda f: fit_chemfit(f, 60, tell_every=True),
        TypeError,
        _WHOLE.format("tell_every", True),
    ),
    _raises(
        "fit_anneal bound_span=True",
        lambda f: fit_anneal(f, 60, bound_span=True),
        TypeError,
        "bound_span must be a number, got True",
        make=_UNBOUNDED,
    ),
    _raises(
        "fit_chemfit default_span=True",
        lambda f: fit_chemfit(f, 60, default_span=True),
        TypeError,
        "default_span must be a number, got True",
        make=_UNBOUNDED,
    ),
    _raises(
        "fit_chemfit boltzmann t_init=True",
        lambda f: fit_chemfit(f, 60, method="boltzmann", t_init=True),
        TypeError,
        "t_init must be a number, got True",
    ),
    _raises(
        "fit_chemfit budget in a 0-d bool array",
        lambda f: fit_chemfit(f, np.array(True)),
        TypeError,
        _WHOLE.format("budget", "array(True)"),
    ),
    _raises(
        "fit_anneal bound_span in a 0-d bool array",
        lambda f: fit_anneal(f, 60, bound_span=np.array(True)),
        TypeError,
        "bound_span must be a number, got array(True)",
        make=_UNBOUNDED,
    ),
    _raises(
        "run_benchmark seed in an array_api_strict bool",
        lambda f: run_benchmark(_context(f), seed=_xp().asarray(True)),
        TypeError,
        "seed must be a whole number, got Array(True",
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
    _raises(
        "fit_chemfit seed=Decimal('Infinity')",
        lambda f: fit_chemfit(f, 60, seed=Decimal("Infinity")),
        ValueError,
        _WHOLE.format("seed", "Decimal('Infinity')"),
    ),
    # 0.10.0 raised ValueError after init.
    _raises(
        "fit_anneal seed=Decimal('NaN')",
        lambda f: fit_anneal(f, 60, seed=Decimal("NaN")),
        ValueError,
        _WHOLE.format("seed", "Decimal('NaN')"),
    ),
    _raises(
        "fit_chemfit boltzmann t_init=Decimal('NaN')",
        lambda f: fit_chemfit(f, 60, method="boltzmann", t_init=Decimal("NaN")),
        ValueError,
        "t_init must be positive and finite, got nan",
    ),
    _raises(
        "fit_chemfit boltzmann t_init=Decimal('sNaN')",
        lambda f: fit_chemfit(f, 60, method="boltzmann", t_init=Decimal("sNaN")),
        TypeError,
        "t_init must be a number, got Decimal('sNaN')",
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
    # 0.10.0 never read these: tell_every under a classical method, which
    # steps the fitter once an epoch, steps_per_epoch under the portfolio,
    # bound_span beside low and high, and default_span when the fitter
    # bounds every parameter.
    _raises(
        "fit_chemfit boltzmann tell_every=0",
        lambda f: fit_chemfit(f, 60, method="boltzmann", tell_every=0),
        ValueError,
        "tell_every must be positive, got 0",
    ),
    _raises(
        "fit_chemfit fast tell_every=2.5",
        lambda f: fit_chemfit(f, 60, method="fast", tell_every=2.5),
        ValueError,
        _WHOLE.format("tell_every", 2.5),
    ),
    _raises(
        "fit_chemfit gsa tell_every=None",
        lambda f: fit_chemfit(f, 60, method="gsa", tell_every=None),
        TypeError,
        _WHOLE.format("tell_every", None),
    ),
    _raises(
        "fit_chemfit boltzmann tell_every='x'",
        lambda f: fit_chemfit(f, 60, method="boltzmann", tell_every="x"),
        TypeError,
        _WHOLE.format("tell_every", "'x'"),
    ),
    _raises(
        "fit_chemfit portfolio steps_per_epoch=0",
        lambda f: fit_chemfit(f, 60, steps_per_epoch=0),
        ValueError,
        "steps_per_epoch must be positive, got 0",
    ),
    _raises(
        "fit_anneal bound_span=0 with low and high",
        lambda f: fit_anneal(f, 60, low=_LOW, high=_HIGH, bound_span=0),
        ValueError,
        _SPAN.format("bound_span", 0),
    ),
    _raises(
        "fit_anneal bound_span=inf with low and high",
        lambda f: fit_anneal(f, 60, low=_LOW, high=_HIGH, bound_span=np.inf),
        ValueError,
        _SPAN.format("bound_span", "inf"),
    ),
    _raises(
        "run_fitter bound_span=-1.0 with low and high",
        lambda f: run_fitter(f, 60, low=_LOW, high=_HIGH, bound_span=-1.0),
        ValueError,
        _SPAN.format("bound_span", -1.0),
    ),
    _raises(
        "fit_chemfit default_span=0 with every parameter bounded",
        lambda f: fit_chemfit(f, 60, default_span=0),
        ValueError,
        _SPAN.format("default_span", 0),
    ),
    _raises(
        "fit_chemfit boltzmann default_span=-1.0 with every parameter bounded",
        lambda f: fit_chemfit(f, 60, method="boltzmann", default_span=-1.0),
        ValueError,
        _SPAN.format("default_span", -1.0),
    ),
    _raises(
        "fit_chemfit default_span=None with every parameter bounded",
        lambda f: fit_chemfit(f, 60, default_span=None),
        TypeError,
        "default_span must be a number, got None",
    ),
    _raises(
        "chemfit_box default_span=-1.0 with every parameter bounded",
        lambda f: chemfit_box(
            f, ChemFitVector(f.initial_parameters), default_span=-1.0
        ),
        ValueError,
        _SPAN.format("default_span", -1.0),
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
    # 0.10.0 read x0 dict leaves flat, so leaves of the wrong sizes with the
    # right total moved values across parameters: a started at [0.3, 0.6].
    _raises(
        "fit_anneal x0 leaves of the wrong sizes",
        lambda f: fit_anneal(f, 60, x0={"a": 0.3, "b": [0.6, 0.7]}),
        ValueError,
        "x0 a has shape (), but the parameter has shape (2,)",
        make=_with({"a": np.zeros(2), "b": 0.0}, {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}),
    ),
    # 0.10.0 raised after init when one x0 value met an unbounded parameter.
    _raises(
        "fit_anneal x0 one value for an unbounded parameter",
        lambda f: fit_anneal(f, 60, x0={"x": 0.4}),
        ValueError,
        "x0 x has shape (), but the parameter has shape (3,)",
        make=_with({"x": np.zeros(3)}, {}),
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
        "fit_chemfit bounds pair of non-numeric strings",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "the lower bound of x is not real-numeric",
        make=_with({"x": 0.5, "y": 0.1}, {"x": ("lo", "1"), "y": (-1.0, 1.0)}),
    ),
    _raises(
        "chemfit_box bounds pair of non-numeric strings",
        _box,
        ValueError,
        "the upper bound of x is not real-numeric",
        make=_with({"x": 0.5}, {"x": ("0", "hi")}),
    ),
    _raises(
        "fit_chemfit bounds entry that is a NumPy pair of non-numeric strings",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "the lower bound of x is not real-numeric",
        make=_with(
            {"x": 0.5, "y": 0.1}, {"x": np.array(["lo", "1"]), "y": (-1.0, 1.0)}
        ),
    ),
    _raises(
        "chemfit_box bounds entry that is a NumPy pair of non-numeric strings",
        _box,
        ValueError,
        "the upper bound of x is not real-numeric",
        make=_with({"x": 0.5}, {"x": np.array(["0", "hi"])}),
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
    # 0.10.0 fell back to fitter.bounds on a context bounds entry that is not
    # one pair, and bounds_from_fitter returned None.
    _raises(
        "run_benchmark context bounds entry of three items",
        lambda f: run_benchmark(
            {**_context(f), "bounds": {"positions": (-3.0, 3.0, 9), "eps": (0.0, 1.0)}}
        ),
        ValueError,
        "the bounds of positions must be a (lower, upper) pair, got (-3.0, 3.0, 9)",
    ),
    _raises(
        "run_benchmark context bounds entry that is a number",
        lambda f: run_benchmark(
            {**_context(f), "bounds": {"positions": 3.0, "eps": (0.0, 1.0)}}
        ),
        ValueError,
        "the bounds of positions must be a (lower, upper) pair, got 3.0",
    ),
    _raises(
        "resolve_bounds context bounds entry that is a dict",
        lambda f: resolve_bounds(
            _X, context_bounds={"x": {"lo": -1.0}}, fitter_bounds={"x": (-2.0, 2.0)}
        ),
        ValueError,
        f"{_NOT_A_PAIR}, got {{'lo': -1.0}}",
    ),
    _raises(
        "bounds_from_fitter entry of three items",
        lambda f: bounds_from_fitter(_X, {"x": (-1.0, 1.0, 9)}, 2),
        ValueError,
        f"{_NOT_A_PAIR}, got (-1.0, 1.0, 9)",
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
        "fit_chemfit Decimal NaN parameter",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "parameter a must be finite",
        make=_with({"a": Decimal("NaN"), "b": 0.5}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}),
    ),
    # 0.10.0's run_benchmark moved an infinite start to the nearer bound.
    _raises(
        "run_benchmark infinite parameter",
        lambda f: run_benchmark(_context(f)),
        ValueError,
        "parameter a must be finite",
        make=_with({"a": np.inf, "b": 0.5}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}),
    ),
    # 0.10.0 read None as NaN.
    _raises(
        "fit_chemfit parameter holding None",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "parameter a is not real-numeric",
        make=_with({"a": [None, 0.5], "b": 0.5}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}),
    ),
    _raises(
        "ChemFitVector object array holding None",
        lambda f: ChemFitVector({"a": np.array([None, 0.5])}),
        ValueError,
        "parameter a is not real-numeric",
    ),
    _raises(
        "resolve_bounds low > high",
        lambda f: resolve_bounds({"x": np.zeros(2)}, low=1.0, high=-1.0),
        ValueError,
        "the bounds of x[0] are empty, the lower above the upper",
    ),
    # The helpers returned these boxes, which no driver accepts.
    _raises(
        "resolve_bounds low=-inf",
        lambda f: resolve_bounds({"x": np.zeros(2)}, low=-np.inf, high=1.0),
        ValueError,
        _NOT_FINITE.format(-np.inf, 1.0),
    ),
    _raises(
        "resolve_bounds high=nan",
        lambda f: resolve_bounds({"x": np.zeros(2)}, low=-1.0, high=np.nan),
        ValueError,
        _NOT_FINITE.format(-1.0, np.nan),
    ),
    _raises(
        "resolve_bounds context low=-inf",
        lambda f: resolve_bounds(
            {"x": np.zeros(2)}, context_bounds={"low": -np.inf, "high": 1.0}
        ),
        ValueError,
        _NOT_FINITE.format(-np.inf, 1.0),
    ),
    _raises(
        "resolve_bounds fitter bounds (-inf, 1)",
        lambda f: resolve_bounds(
            {"x": np.zeros(2)}, fitter_bounds={"x": (-np.inf, 1.0)}
        ),
        ValueError,
        _NOT_FINITE.format(-np.inf, 1.0),
    ),
    _raises(
        "resolve_bounds too wide for a float",
        lambda f: resolve_bounds({"x": np.zeros(2)}, low=-1e308, high=1e308),
        ValueError,
        "the bounds of x[0] are too wide for a float: lower=-1e+308, upper=1e+308",
    ),
    _raises(
        "bounds_from_fitter (nan, 1)",
        lambda f: bounds_from_fitter({"x": np.zeros(2)}, {"x": (np.nan, 1.0)}, 2),
        ValueError,
        _NOT_FINITE.format(np.nan, 1.0),
    ),
    _raises(
        "bounds_from_fitter (1, -1)",
        lambda f: bounds_from_fitter({"x": np.zeros(2)}, {"x": (1.0, -1.0)}, 2),
        ValueError,
        "the bounds of x[0] are empty, the lower above the upper",
    ),
    _raises(
        "chemfit_box too wide for a float",
        _box,
        ValueError,
        "the bounds of x are too wide for a float: lower=-1e+308, upper=1e+308",
        make=_with({"x": 0.5}, {"x": (-1e308, 1e308)}),
    ),
    _raises(
        "chemfit_box bounds (-inf, 3)",
        _box,
        ValueError,
        "the bounds of x must be finite: lower=-inf, upper=3.0",
        make=_with({"x": 0.5}, {"x": (-np.inf, 3.0)}),
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
    _raises(
        "fit_chemfit object array holding a NumPy complex number",
        lambda f: fit_chemfit(f, 60),
        ValueError,
        "parameter x is not real-numeric",
        make=_with(
            {"x": np.array([np.complex128(0.5), 0.25], dtype=object)},
            {"x": (-2.0, 2.0)},
        ),
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


def _same_fit(call, same_as):
    """``call`` and ``same_as`` fit fresh fitters alike, with no FutureWarning."""
    fitter, plain = _fitter(), _fitter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = call(fitter)
        expected = same_as(plain)
    assert len(fitter.evaluated) == len(plain.evaluated) > 1
    for got, want in zip(fitter.evaluated, plain.evaluated):
        assert same_params(got, want)
    assert same_params(out, expected)


@pytest.mark.parametrize("call, same_as", NOW_ACCEPTED)
def test_fit_chemfit_and_run_fitter_read_method_names_in_any_case(call, same_as):
    _same_fit(call, same_as)


_NARROW = {"positions": (-0.5, 0.5), "eps": (0.0, 1.0)}
_ROWS = np.stack([np.full((2, 3), -0.5), np.full((2, 3), 0.5)])

# Bounds 0.10.0 ignored, using fitter.bounds or the start +/- bound_span or
# default_span instead, each with the call it now matches.
NOW_READ = [
    pytest.param(
        lambda f: fit_chemfit(
            _as(f, {**_NARROW, "positions": np.array([-0.5, 0.5], dtype=object)}), 60
        ),
        lambda f: fit_chemfit(_as(f, _NARROW), 60),
        id="fit_chemfit bounds entry that is a NumPy pair of objects",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            _as(f, {**_NARROW, "positions": np.array([False, True])}), 60
        ),
        lambda f: fit_chemfit(_as(f, {**_NARROW, "positions": (0.0, 1.0)}), 60),
        id="fit_chemfit bounds entry that is a NumPy pair of bools",
    ),
    pytest.param(
        lambda f: fit_chemfit(_as(f, {**_NARROW, "positions": _ROWS}), 60),
        lambda f: fit_chemfit(_as(f, _NARROW), 60),
        id="fit_chemfit per-element bounds in a NumPy array of two rows",
    ),
    pytest.param(
        lambda f: fit_chemfit(_as(f, {**_NARROW, "positions": list(_ROWS)}), 60),
        lambda f: fit_chemfit(_as(f, _NARROW), 60),
        id="fit_chemfit per-element bounds in a list of two arrays",
    ),
    pytest.param(
        lambda f: run_benchmark(
            {**_context(f), "bounds": {**_NARROW, "positions": np.array([-0.5, 0.5])}}
        ),
        lambda f: run_benchmark({**_context(f), "bounds": _NARROW}),
        id="run_benchmark context bounds entry that is a NumPy pair",
    ),
    pytest.param(
        lambda f: run_benchmark({**_context(f), "bounds": MappingProxyType(_NARROW)}),
        lambda f: run_benchmark({**_context(f), "bounds": _NARROW}),
        id="run_benchmark context bounds in a mapping proxy",
    ),
    pytest.param(
        lambda f: run_benchmark(
            {**_context(f), "bounds": MappingProxyType({"low": -0.5, "high": 0.5})}
        ),
        lambda f: run_benchmark({**_context(f), "bounds": {"low": -0.5, "high": 0.5}}),
        id="run_benchmark context low and high in a mapping proxy",
    ),
    pytest.param(
        lambda f: fit_anneal(_as(f, MappingProxyType(_NARROW)), 60),
        lambda f: fit_anneal(_as(f, _NARROW), 60),
        id="fit_anneal fitter bounds in a mapping proxy",
    ),
    pytest.param(
        lambda f: run_fitter(_as(f, MappingProxyType(_NARROW)), 60, method="fast"),
        lambda f: run_fitter(_as(f, _NARROW), 60, method="fast"),
        id="run_fitter fitter bounds in a mapping proxy",
    ),
]


@pytest.mark.parametrize("call, same_as", NOW_READ)
def test_bounds_0_10_0_ignored_are_now_read(call, same_as):
    _same_fit(call, same_as)


_PAIR = {"x": np.array([-1.0, 1.0])}
_PROXY = MappingProxyType({"x": (-1.0, 1.0)})
_WIDE = {"x": (-2.0, 2.0)}

HELPERS_NOW_READ = [
    pytest.param(
        lambda: resolve_bounds(_X, context_bounds=_PAIR, fitter_bounds=_WIDE),
        id="resolve_bounds context bounds entry that is a NumPy pair",
    ),
    pytest.param(
        lambda: resolve_bounds(_X, context_bounds=_PROXY, fitter_bounds=_WIDE),
        id="resolve_bounds context bounds in a mapping proxy",
    ),
    pytest.param(
        lambda: bounds_from_fitter(_X, _PAIR, 2),
        id="bounds_from_fitter entry that is a NumPy pair",
    ),
    pytest.param(
        lambda: bounds_from_fitter(_X, _PROXY, 2),
        id="bounds_from_fitter bounds in a mapping proxy",
    ),
    pytest.param(
        lambda: _box(ReleasedFitter(_X, {"x": np.array([-1.0, 1.0], dtype=object)})),
        id="chemfit_box bounds entry that is a NumPy pair of objects",
    ),
    pytest.param(
        lambda: _box(ReleasedFitter(_X, {"x": np.array([[-1.0, -1.0], [1.0, 1.0]])})),
        id="chemfit_box per-element bounds in a NumPy array of two rows",
    ),
]


@pytest.mark.parametrize("call", HELPERS_NOW_READ)
def test_the_helpers_read_bounds_0_10_0_ignored(call):
    assert _same(call(), (np.full(2, -1.0), np.full(2, 1.0)))


# Numbers 0.10.0 read through NumPy as the floats they hold, though NumPy
# holds them only as objects, each with the call it matches.
OBJECTS_READ = [
    pytest.param(
        lambda f: fit_chemfit(_as(f, eps=Decimal("0.5")), 60),
        lambda f: fit_chemfit(f, 60),
        id="fit_chemfit Decimal parameter",
    ),
    pytest.param(
        lambda f: fit_anneal(_as(f, eps=Fraction(1, 2)), 60),
        lambda f: fit_anneal(f, 60),
        id="fit_anneal Fraction parameter",
    ),
    pytest.param(
        lambda f: run_benchmark(
            {
                **_context(f),
                "initial_params": {
                    "positions": _objects(_POSITIONS, _decimal).tolist(),
                    "eps": Fraction(1, 2),
                },
            }
        ),
        lambda f: run_benchmark(
            {
                **_context(f),
                "initial_params": {"positions": _POSITIONS.tolist(), "eps": 0.5},
            }
        ),
        id="run_benchmark lists of Decimals and a Fraction",
    ),
    pytest.param(
        lambda f: run_fitter(_as(f, positions=_objects(_POSITIONS, _decimal)), 60),
        lambda f: run_fitter(f, 60),
        id="run_fitter object array of Decimals",
    ),
    pytest.param(
        lambda f: fit_chemfit(_as(f, positions=_objects(_POSITIONS)), 60),
        lambda f: fit_chemfit(f, 60),
        id="fit_chemfit object array of floats",
    ),
    pytest.param(
        lambda f: fit_anneal(
            _as(f, positions=_objects(_POSITIONS, lambda v: np.array([v]))), 60
        ),
        lambda f: fit_anneal(f, 60),
        id="fit_anneal object array of one-element arrays",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            _as(f, {**_NARROW, "eps": (Decimal(0), Fraction(1))}), 60
        ),
        lambda f: fit_chemfit(_as(f, _NARROW), 60),
        id="fit_chemfit Decimal and Fraction bounds",
    ),
    pytest.param(
        lambda f: fit_anneal(_as(f, {**_NARROW, "eps": (Fraction(0), Decimal(1))}), 60),
        lambda f: fit_anneal(_as(f, _NARROW), 60),
        id="fit_anneal Fraction and Decimal bounds",
    ),
    pytest.param(
        lambda f: fit_anneal(f, 60, x0=[_decimal(v) for v in _START[:7]]),
        lambda f: fit_anneal(f, 60, x0=_START[:7]),
        id="fit_anneal x0 of Decimals",
    ),
    pytest.param(
        lambda f: fit_anneal(
            f, 60, low=[Fraction(-1)] * 7, high=_objects(_HIGH, _decimal)
        ),
        lambda f: fit_anneal(f, 60, low=_LOW, high=_HIGH),
        id="fit_anneal low of Fractions and high of Decimals",
    ),
    pytest.param(
        lambda f: run_benchmark(_context(f), low=Decimal(-1), high=Fraction(1)),
        lambda f: run_benchmark(_context(f), low=-1.0, high=1.0),
        id="run_benchmark Decimal low and Fraction high",
    ),
    pytest.param(
        lambda f: run_benchmark(
            {**_context(f), "bounds": {**_NARROW, "eps": (Decimal(0), Decimal(1))}}
        ),
        lambda f: run_benchmark({**_context(f), "bounds": _NARROW}),
        id="run_benchmark Decimal context bounds",
    ),
]


@pytest.mark.parametrize("call, same_as", OBJECTS_READ)
def test_numbers_numpy_holds_as_objects_are_read_as_the_floats_they_hold(call, same_as):
    _same_fit(call, same_as)


def test_the_helpers_read_numbers_numpy_holds_as_objects():
    template = {
        "a": Decimal("0.5"),
        "b": [Fraction(1, 4), Decimal(1)],
        "c": _objects([0.5, 0.25]),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        vector = ChemFitVector(template)
        x0, spec = flatten_parameters(template)
        back = unflatten_parameters(x0, spec, template)
        box = resolve_bounds(template, low=Decimal(-1), high=Fraction(1))
    assert _same(vector.x0, np.array([0.5, 0.25, 1.0, 0.5, 0.25]))
    assert _same(x0, vector.x0)
    expected = {"a": 0.5, "b": [0.25, 1.0], "c": np.array([0.5, 0.25])}
    assert same_params(back, expected)
    assert same_params(vector.unpack(vector.x0), expected)
    assert _same(box, (np.full(5, -1.0), np.full(5, 1.0)))


_EPS_UNBOUNDED = {"positions": (-2.0, 2.0)}

# Settings 0.10.0 passed through int() or float(), which read each of these
# as the number it holds, each with the call that passes that number.
SETTINGS_READ = [
    pytest.param(
        lambda f: fit_anneal(f, Decimal(60), seed=Decimal(3)),
        lambda f: fit_anneal(f, 60, seed=3),
        id="fit_anneal Decimal budget and seed",
    ),
    pytest.param(
        lambda f: fit_chemfit(f, Decimal(60), seed=Decimal(3), tell_every=Decimal(5)),
        lambda f: fit_chemfit(f, 60, seed=3, tell_every=5),
        id="fit_chemfit Decimal budget, seed and tell_every",
    ),
    pytest.param(
        lambda f: run_benchmark(_context(f, Decimal(60)), seed=Decimal(3)),
        lambda f: run_benchmark(_context(f), seed=3),
        id="run_benchmark Decimal budget and seed",
    ),
    pytest.param(
        lambda f: run_fitter(f, Decimal(60), seed=Decimal(3)),
        lambda f: run_fitter(f, 60, seed=3),
        id="run_fitter Decimal budget and seed",
    ),
    pytest.param(
        lambda f: fit_anneal(f, 60, driver="boltzmann", steps_per_epoch=Decimal(10)),
        lambda f: fit_anneal(f, 60, driver="boltzmann", steps_per_epoch=10),
        id="fit_anneal Decimal steps_per_epoch",
    ),
    pytest.param(
        lambda f: fit_anneal(_as(f, _EPS_UNBOUNDED), 60, bound_span=Decimal("0.5")),
        lambda f: fit_anneal(_as(f, _EPS_UNBOUNDED), 60, bound_span=0.5),
        id="fit_anneal Decimal bound_span",
    ),
    pytest.param(
        lambda f: fit_chemfit(_as(f, _EPS_UNBOUNDED), 60, default_span=Decimal("0.5")),
        lambda f: fit_chemfit(_as(f, _EPS_UNBOUNDED), 60, default_span=0.5),
        id="fit_chemfit Decimal default_span",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            f,
            60,
            method="boltzmann",
            steps_per_epoch=Decimal(10),
            t_init=Decimal(2),
            sigma=Decimal("0.4"),
        ),
        lambda f: fit_chemfit(
            f, 60, method="boltzmann", steps_per_epoch=10, t_init=2.0, sigma=0.4
        ),
        id="fit_chemfit boltzmann Decimal steps_per_epoch, t_init and sigma",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            f, 60, method="fast", steps_per_epoch=10, gamma=Decimal("0.6")
        ),
        lambda f: fit_chemfit(f, 60, method="fast", steps_per_epoch=10, gamma=0.6),
        id="fit_chemfit fast Decimal gamma",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            f,
            60,
            method="gsa",
            steps_per_epoch=10,
            q_v=Decimal("2.5"),
            q_a=Decimal("1.5"),
        ),
        lambda f: fit_chemfit(
            f, 60, method="gsa", steps_per_epoch=10, q_v=2.5, q_a=1.5
        ),
        id="fit_chemfit gsa Decimal q_v and q_a",
    ),
    pytest.param(
        lambda f: fit_chemfit(f, np.array(60), seed=np.array(3.0)),
        lambda f: fit_chemfit(f, 60, seed=3),
        id="fit_chemfit budget and seed in 0-d arrays",
    ),
    pytest.param(
        lambda f: fit_anneal(
            _as(f, _EPS_UNBOUNDED),
            60,
            driver="fast",
            steps_per_epoch=np.array(10),
            bound_span=np.array(0.5),
        ),
        lambda f: fit_anneal(
            _as(f, _EPS_UNBOUNDED),
            60,
            driver="fast",
            steps_per_epoch=10,
            bound_span=0.5,
        ),
        id="fit_anneal steps_per_epoch and bound_span in 0-d arrays",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            f,
            60,
            method="boltzmann",
            steps_per_epoch=10,
            t_init=np.array(2),
            sigma=np.array(0.4, dtype=np.float32),
        ),
        lambda f: fit_chemfit(
            f,
            60,
            method="boltzmann",
            steps_per_epoch=10,
            t_init=2.0,
            sigma=float(np.float32(0.4)),
        ),
        id="fit_chemfit boltzmann t_init and sigma in 0-d arrays",
    ),
    pytest.param(
        lambda f: fit_chemfit(f, np.array(Decimal(60), dtype=object)),
        lambda f: fit_chemfit(f, 60),
        id="fit_chemfit budget in a 0-d object array of a Decimal",
    ),
    pytest.param(
        lambda f: run_benchmark(
            _context(f, np.array([60])),
            method="boltzmann",
            seed=np.array([[3]]),
            steps_per_epoch=np.array([10]),
        ),
        lambda f: run_benchmark(
            _context(f), method="boltzmann", seed=3, steps_per_epoch=10
        ),
        id="run_benchmark budget, seed and steps_per_epoch in one-element arrays",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            _as(f, _EPS_UNBOUNDED),
            60,
            default_span=np.array([0.5]),
            tell_every=np.array([5]),
        ),
        lambda f: fit_chemfit(
            _as(f, _EPS_UNBOUNDED), 60, default_span=0.5, tell_every=5
        ),
        id="fit_chemfit default_span and tell_every in one-element arrays",
    ),
    pytest.param(
        lambda f: run_fitter(
            _as(f, _EPS_UNBOUNDED), _matrix([[60]]), bound_span=np.array([[0.5]])
        ),
        lambda f: run_fitter(_as(f, _EPS_UNBOUNDED), 60, bound_span=0.5),
        id="run_fitter budget in a 1x1 matrix and bound_span in a 1x1 array",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            f, 60, method="gsa", steps_per_epoch=10, q_v=_matrix([[2.5]])
        ),
        lambda f: fit_chemfit(f, 60, method="gsa", steps_per_epoch=10, q_v=2.5),
        id="fit_chemfit gsa q_v in a 1x1 matrix",
    ),
    pytest.param(
        lambda f: fit_anneal(f, _xp().asarray(60), seed=_xp().asarray(3)),
        lambda f: fit_anneal(f, 60, seed=3),
        id="fit_anneal budget and seed in array_api_strict arrays",
    ),
    pytest.param(
        lambda f: run_benchmark(_context(f, _xp().asarray(60)), seed=_xp().asarray(3)),
        lambda f: run_benchmark(_context(f), seed=3),
        id="run_benchmark budget and seed in array_api_strict arrays",
    ),
    pytest.param(
        lambda f: run_fitter(_as(f, _EPS_UNBOUNDED), 60, bound_span=_xp().asarray(0.5)),
        lambda f: run_fitter(_as(f, _EPS_UNBOUNDED), 60, bound_span=0.5),
        id="run_fitter bound_span in an array_api_strict array",
    ),
    pytest.param(
        lambda f: fit_chemfit(
            _as(f, _EPS_UNBOUNDED),
            60,
            method="boltzmann",
            default_span=_xp().asarray(0.5),
            steps_per_epoch=_xp().asarray(10),
            t_init=_xp().asarray(2.0, dtype=_xp().float32),
        ),
        lambda f: fit_chemfit(
            _as(f, _EPS_UNBOUNDED),
            60,
            method="boltzmann",
            default_span=0.5,
            steps_per_epoch=10,
            t_init=2.0,
        ),
        id="fit_chemfit default_span, steps_per_epoch and t_init in array_api_strict arrays",
    ),
]


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("call, same_as", SETTINGS_READ)
def test_settings_0_10_0_read_as_numbers_are_read_as_those_numbers(call, same_as):
    _same_fit(call, same_as)


# 0.10.0 refused these, whose own int() and float() need a 0-d array.
SETTINGS_NOW_READ = [
    pytest.param(
        lambda f: fit_chemfit(f, _xp().asarray([60]), seed=_xp().asarray([[3]])),
        lambda f: fit_chemfit(f, 60, seed=3),
        id="fit_chemfit budget and seed in one-element array_api_strict arrays",
    ),
    pytest.param(
        lambda f: fit_anneal(
            _as(f, _EPS_UNBOUNDED), 60, bound_span=_xp().asarray([0.5])
        ),
        lambda f: fit_anneal(_as(f, _EPS_UNBOUNDED), 60, bound_span=0.5),
        id="fit_anneal bound_span in a one-element array_api_strict array",
    ),
]


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("call, same_as", SETTINGS_NOW_READ)
def test_settings_in_one_element_arrays_0_10_0_refused_are_read(call, same_as):
    _same_fit(call, same_as)


def test_chemfit_box_reads_default_span_as_the_number_it_holds():
    fitter = ReleasedFitter(_X, {})
    vector = ChemFitVector(fitter.initial_parameters)
    want = chemfit_box(fitter, vector, default_span=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for span in (Decimal("0.5"), np.array(0.5), np.array([0.5]), _matrix([[0.5]])):
            assert _same(chemfit_box(fitter, vector, default_span=span), want)
