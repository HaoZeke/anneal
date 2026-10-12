"""``run_fitter``: the options it passes on to ``fit_anneal``, and no others.

A fitter with its own ``fit_anneal`` is handed to it under the method names
of ``run_benchmark``, which a ``fit_anneal`` that runs ``run_benchmark``
takes.
"""

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Fast  # noqa: E402
from anneal.chemfit import fit_anneal, run_benchmark, run_fitter  # noqa: E402
from chemfit_doubles import NextFitter, ReleasedFitter, same_params  # noqa: E402


class BenchmarkFitter(NextFitter):
    """A fitter whose own ``fit_anneal`` runs ``run_benchmark`` and takes a preset.

    ``handed`` logs the method, preset and seed each call received.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handed = []

    def fit_anneal(
        self,
        budget,
        *,
        method="portfolio",
        preset=None,
        seed=0,
        batch_size=1,
        contexts=None,
        steps_per_epoch=100,
    ):
        self.handed.append((method, preset, seed))
        context = {"fitter": self, "budget": budget, "initial_params": self.initial_parameters}
        return run_benchmark(
            context, method=method, preset=preset, seed=seed, steps_per_epoch=steps_per_epoch
        )


def _fitter(kind=ReleasedFitter, bounds=None):
    return kind(
        {"x": np.array([0.2, -0.1]), "eps": 1.0},
        {"x": (-1.0, 1.0), "eps": (0.5, 2.0)} if bounds is None else bounds,
    )


def _same_run(left, right):
    return len(left.evaluated) == len(right.evaluated) and all(
        same_params(a, b) for a, b in zip(left.evaluated, right.evaluated)
    )


def test_run_fitter_forwards_the_fit_anneal_options():
    fitter = _fitter(NextFitter)
    low = np.array([0.0, 0.0, 1.0])
    high = np.array([1.0, 1.0, 2.0])
    run_fitter(fitter, 40, x0={"x": np.array([0.5, 0.25]), "eps": 1.5}, low=low, high=high)
    seen = np.array([np.append(p["x"], p["eps"]) for p in fitter.evaluated])
    assert 1 < len(seen) <= 40
    assert np.array_equal(seen[0], [0.5, 0.25, 1.5])
    assert np.all(seen >= low) and np.all(seen <= high)


def test_run_fitter_runs_the_preset_it_is_given():
    given = _fitter()
    run_fitter(given, 60, method="sa", preset=Fast(t_init=2.0), seed=4, steps_per_epoch=10)
    named = _fitter()
    fit_anneal(
        named, 60, driver="fast", seed=4, steps_per_epoch=10, preset_kwargs={"t_init": 2.0}
    )
    assert len(given.evaluated) > 1
    assert _same_run(given, named)


@pytest.mark.parametrize(
    "method, option",
    [
        ("global_optimize", {"bound_span": 0.25}),
        ("boltzmann", {"steps_per_epoch": 7}),
        ("fast", {"preset_kwargs": {"t_init": 0.5, "gamma": 2.0}}),
    ],
    ids=["bound_span", "steps_per_epoch", "preset_kwargs"],
)
def test_each_forwarded_option_takes_effect(method, option):
    driver = "portfolio" if method == "global_optimize" else method
    forwarded, direct, plain = (_fitter(bounds={}) for _ in range(3))
    run_fitter(forwarded, 60, method=method, **option)
    fit_anneal(direct, 60, driver=driver, seed=42, **option)
    run_fitter(plain, 60, method=method)
    assert len(forwarded.evaluated) > 1
    assert _same_run(forwarded, direct)
    assert not _same_run(forwarded, plain)


@pytest.mark.parametrize(
    "keyword, hint",
    [
        ("n_epochs", "on to fit_anneal"),
        ("bogus", "on to fit_anneal"),
        ("tell_every", "tell_every is an option of fit_chemfit"),
        ("default_span", "default_span is an option of fit_chemfit"),
        ("t_init", "pass t_init in preset_kwargs"),
    ],
)
def test_run_fitter_names_itself_for_a_keyword_it_does_not_pass_on(keyword, hint):
    fitter = _fitter()
    with pytest.raises(TypeError) as caught:
        run_fitter(fitter, 60, **{keyword: 5})
    message = str(caught.value)
    assert message.startswith(
        f"run_fitter() got an unexpected keyword argument {keyword!r}; it passes only "
        "x0, low, high, bound_span, steps_per_epoch and preset_kwargs on to fit_anneal"
    )
    assert hint in message
    assert "_fit_anneal" not in message
    assert fitter.calls == []


def test_a_fitter_with_its_own_fit_anneal_gets_every_keyword():
    class OwnFitAnneal:
        def fit_anneal(self, **kwargs):
            return kwargs

    got = run_fitter(OwnFitAnneal(), 60, method="sa", tell_every=5, n_epochs=3)
    assert got == {
        "budget": 60,
        "method": "boltzmann",
        "preset": None,
        "seed": 42,
        "tell_every": 5,
        "n_epochs": 3,
    }


SLOW_FAST = Fast(t_init=2.0)


@pytest.mark.parametrize(
    "options, handed",
    [
        ({}, ("portfolio", None, 42)),
        ({"method": "portfolio", "seed": 3}, ("portfolio", None, 3)),
        ({"method": "sa"}, ("boltzmann", None, 42)),
        ({"method": "gsa", "steps_per_epoch": 10}, ("gsa", None, 42)),
        ({"method": "sa", "preset": SLOW_FAST, "steps_per_epoch": 10}, ("fast", SLOW_FAST, 42)),
    ],
    ids=["defaults", "portfolio", "sa", "gsa", "sa-preset"],
)
def test_a_fit_anneal_over_run_benchmark_runs_each_method_as_the_bridges_do(options, handed):
    own, plain = _fitter(BenchmarkFitter), _fitter(NextFitter)
    run_fitter(own, 60, **options)
    run_fitter(plain, 60, **options)
    assert len(own.handed) == 1
    assert own.handed[0][0] == handed[0] and own.handed[0][2] == handed[2]
    assert own.handed[0][1] is handed[1]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, plain)


def test_a_preset_another_driver_runs_is_ignored_for_a_fit_anneal_too():
    own = _fitter(BenchmarkFitter)
    with pytest.warns(FutureWarning, match="ignores the Fast preset under method 'boltzmann'"):
        run_fitter(own, 60, method="boltzmann", preset=Fast())
    assert [(method, preset) for method, preset, _ in own.handed] == [("boltzmann", None)]
    assert 1 < len(own.evaluated) <= 60


def test_an_unknown_method_is_refused_before_a_fit_anneal_is_called():
    own = _fitter(BenchmarkFitter)
    with pytest.raises(ValueError, match="method must be one of 'global_optimize', 'sa'"):
        run_fitter(own, 60, method="anneal")
    assert own.handed == [] and own.calls == []
