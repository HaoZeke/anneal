"""Calls anneal 0.10.0 accepted, and what each of them does now.

An argument 0.10.0 took and ignored still runs. It gives one FutureWarning,
attributed to the caller, that says what to pass instead, and the fit is the
one the call that passes that gives. ``run_benchmark`` still drives a fitter
with half a session protocol, as 0.10.0 did, with a FutureWarning too.
"""

import re
import warnings

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Boltzmann, Fast  # noqa: E402
from anneal.chemfit import (  # noqa: E402
    fit_anneal,
    fit_chemfit,
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
    return {"fitter": fitter, "budget": budget, "initial_params": fitter.initial_parameters}


def _shape_warning(name, given, shape):
    return re.escape(f"x0 {name} has shape {given}, but the parameter has shape {shape}")


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
            lambda f, m=method: fit_chemfit(f, 60, method=m, steps_per_epoch=10, **_SWEEP),
            lambda f, m=method: fit_chemfit(
                f, 60, method=m, steps_per_epoch=10, **{k: _SWEEP[k] for k in _OWN[m]}
            ),
            f"which method '{method}' does not take; it takes {', '.join(_OWN[method])}",
        )
        for method in _OWN
    ],
    (
        "fit_anneal x0 leaf (3, 2) for (2, 3)",
        _fitter,
        lambda f: fit_anneal(f, 60, x0={"positions": _START[:6].reshape(3, 2), "eps": 0.5}),
        lambda f: fit_anneal(f, 60, x0={"positions": _START[:6].reshape(2, 3), "eps": 0.5}),
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
