"""``run_fitter``: the options it passes on to ``fit_anneal``, and no others.

A fitter with its own ``fit_anneal`` is handed to it with what the signature
of that method takes, the method under the names of ``run_benchmark``, which
a ``fit_anneal`` that runs ``run_benchmark`` takes. The first three doubles
below have the signatures such methods have had: one that runs
``run_benchmark`` with a ``preset`` parameter, one without it, and one with
neither ``method`` nor ``preset`` that runs ``anneal.minimize`` itself. The
others try what else a signature can do: take the budget alone, leave out
the seed, or take parameters only by position.
"""

import functools
import warnings

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Boltzmann, Fast, minimize  # noqa: E402
from anneal.chemfit import fit_anneal, run_benchmark, run_fitter  # noqa: E402
from chemfit_doubles import NextFitter, ReleasedFitter, same_params  # noqa: E402


class Handed(NextFitter):
    """A fitter whose own ``fit_anneal`` logs what each call received in ``handed``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handed = []

    def _benchmark(self, budget, **options):
        context = {"fitter": self, "budget": budget, "initial_params": self.initial_parameters}
        return run_benchmark(context, **options)

    def _minimize(self, budget, seed, replicas=1, history="shared"):
        self.init()
        start = np.append(self.initial_parameters["x"], self.initial_parameters["eps"])
        bounds = [self.bounds["x"], self.bounds["x"], self.bounds["eps"]]

        def loss(v):
            return self.evaluate({"x": v[:2], "eps": float(v[2])})

        res = minimize(
            loss,
            start,
            bounds=bounds,
            jac=None,
            budget=budget,
            seed=seed,
            replicas=replicas,
            history=history,
        )
        return self.finish({"x": res.x[:2], "eps": float(res.x[2])})


class BenchmarkFitter(Handed):
    """Runs ``run_benchmark`` and takes a preset."""

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
        self.handed.append({"method": method, "preset": preset, "seed": seed})
        return self._benchmark(
            budget, method=method, preset=preset, seed=seed, steps_per_epoch=steps_per_epoch
        )


class NoPresetFitter(Handed):
    """Runs ``run_benchmark`` but takes no preset."""

    def fit_anneal(
        self,
        budget,
        *,
        method="portfolio",
        seed=0,
        batch_size=1,
        contexts=None,
        steps_per_epoch=100,
    ):
        self.handed.append({"method": method, "seed": seed})
        return self._benchmark(budget, method=method, seed=seed, steps_per_epoch=steps_per_epoch)


class OwnLoopFitter(Handed):
    """Takes neither method nor preset, and runs ``anneal.minimize`` itself."""

    def fit_anneal(
        self, budget, replicas=4, store=None, seed=0, history="shared", jac=None, ctx=None
    ):
        self.handed.append({"budget": budget, "replicas": replicas, "seed": seed})
        return self._minimize(budget, seed, replicas, history)


class BudgetOnlyFitter(Handed):
    """Takes the budget alone."""

    def fit_anneal(self, budget):
        self.handed.append({"budget": budget})
        return self._minimize(budget, seed=0)


class NoSeedFitter(Handed):
    """Runs ``run_benchmark`` with the method it is given, and has no seed parameter."""

    def fit_anneal(self, budget, method="portfolio"):
        self.handed.append({"method": method})
        return self._benchmark(budget, method=method)


class PositionalBudgetFitter(Handed):
    """Runs ``run_benchmark``, taking the budget only by position."""

    def fit_anneal(self, budget, /, *, method="portfolio", seed=0):
        self.handed.append({"budget": budget, "method": method, "seed": seed})
        return self._benchmark(budget, method=method, seed=seed)


class PositionalBudgetKwargsFitter(Handed):
    """Runs ``run_benchmark`` with what ``**options`` gets, taking the budget only by position."""

    def fit_anneal(self, budget, /, **options):
        self.handed.append({"budget": budget, **options})
        return self._benchmark(budget, **options)


class PositionalSeedFitter(Handed):
    """Takes the budget and the seed only by position, and anything else as ``**options``."""

    def fit_anneal(self, budget, seed=0, /, **options):
        self.handed.append({"budget": budget, "seed": seed, **options})
        return self._minimize(budget, seed)


class PositionalPresetFitter(Handed):
    """Runs ``run_benchmark``, taking the preset only by position."""

    def fit_anneal(self, budget, preset=None, /, *, method="portfolio", seed=0):
        self.handed.append({"method": method, "preset": preset, "seed": seed})
        return self._benchmark(budget, method=method, preset=preset, seed=seed)


class LateBudgetFitter(Handed):
    """Takes the budget only by position, after a scale."""

    def fit_anneal(self, scale, budget, /):
        self.handed.append({"scale": scale, "budget": budget})
        return self._minimize(budget, seed=0)


def _fitter(kind=ReleasedFitter, bounds=None):
    return kind(
        {"x": np.array([0.2, -0.1]), "eps": 1.0},
        {"x": (-1.0, 1.0), "eps": (0.5, 2.0)} if bounds is None else bounds,
    )


def _same_run(left, right):
    return len(left.evaluated) == len(right.evaluated) and all(
        same_params(a, b) for a, b in zip(left.evaluated, right.evaluated)
    )


def _ignored_fast(method, runs, why_not=None):
    """The warning for a Fast preset that ``method`` does not run.

    ``why_not`` says why a fitter's own ``fit_anneal`` cannot take the preset.
    """
    if why_not is None:
        return (
            f"run_fitter ignores the Fast preset under method {method!r}, which runs {runs}; "
            "leave it out, or pass method='sa' to run it. This will raise in a future release."
        )
    return (
        f"run_fitter ignores the Fast preset under method {method!r}, which runs {runs}, "
        f"and {why_not}; leave it out. This will raise in a future release."
    )


def _warned(call):
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        call()
    return [(w.category, str(w.message)) for w in warned]


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


def test_a_fit_anneal_that_takes_kwargs_gets_what_the_caller_gave():
    class OwnFitAnneal:
        def fit_anneal(self, **kwargs):
            return kwargs

    assert run_fitter(OwnFitAnneal(), 60) == {"budget": 60}
    got = run_fitter(OwnFitAnneal(), 60, method="sa", seed=7, tell_every=5, n_epochs=3)
    assert got == {
        "budget": 60,
        "method": "boltzmann",
        "seed": 7,
        "tell_every": 5,
        "n_epochs": 3,
    }
    with pytest.warns(FutureWarning) as warned:
        got = run_fitter(OwnFitAnneal(), 60, method="boltzmann", preset=Fast())
    assert got == {"budget": 60, "method": "boltzmann"}
    assert [str(w.message) for w in warned] == [
        _ignored_fast("boltzmann", "its own boltzmann preset")
    ]


SLOW_FAST = Fast(t_init=2.0)


@pytest.mark.parametrize("kind", [BenchmarkFitter, NoPresetFitter], ids=["preset", "no-preset"])
@pytest.mark.parametrize(
    "options, method, seed",
    [
        ({}, "portfolio", 42),
        ({"method": "portfolio", "seed": 3}, "portfolio", 3),
        ({"method": "sa"}, "boltzmann", 42),
        ({"method": "gsa", "steps_per_epoch": 10}, "gsa", 42),
    ],
    ids=["defaults", "portfolio", "sa", "gsa"],
)
def test_a_fit_anneal_over_run_benchmark_runs_each_method_as_the_bridges_do(
    kind, options, method, seed
):
    own, plain = _fitter(kind), _fitter(NextFitter)
    run_fitter(own, 60, **options)
    run_fitter(plain, 60, **options)
    assert [(h["method"], h["seed"], h.get("preset")) for h in own.handed] == [
        (method, seed, None)
    ]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, plain)


def test_a_fit_anneal_that_takes_a_preset_runs_the_one_given():
    own, plain = _fitter(BenchmarkFitter), _fitter(NextFitter)
    run_fitter(own, 60, method="sa", preset=SLOW_FAST, steps_per_epoch=10)
    run_fitter(plain, 60, method="sa", preset=SLOW_FAST, steps_per_epoch=10)
    assert own.handed == [{"method": "fast", "preset": SLOW_FAST, "seed": 42}]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, plain)


@pytest.mark.parametrize(
    "kind, why_not",
    [
        (BenchmarkFitter, None),
        (NoPresetFitter, "NoPresetFitter.fit_anneal() takes no 'preset'"),
        (
            PositionalPresetFitter,
            "PositionalPresetFitter.fit_anneal() takes 'preset' only by position",
        ),
    ],
    ids=["preset", "no-preset", "positional-only-preset"],
)
@pytest.mark.parametrize(
    "method, driver, runs",
    [
        ("global_optimize", "portfolio", "the portfolio"),
        ("boltzmann", "boltzmann", "its own boltzmann preset"),
    ],
    ids=["default", "boltzmann"],
)
def test_a_preset_the_method_does_not_run_is_ignored_for_a_fit_anneal_too(
    kind, why_not, method, driver, runs
):
    own, plain = _fitter(kind), _fitter(NextFitter)
    for fitter, why in ((own, why_not), (plain, None)):
        warned = _warned(lambda: run_fitter(fitter, 60, method=method, preset=Fast()))
        assert warned == [(FutureWarning, _ignored_fast(method, runs, why))]
    assert [(h["method"], h.get("preset")) for h in own.handed] == [(driver, None)]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, plain)


def test_a_fit_anneal_without_method_or_preset_runs_as_it_stands():
    fitter = _fitter(OwnLoopFitter)
    run_fitter(fitter, 60)
    assert fitter.handed == [{"budget": 60, "replicas": 4, "seed": 42}]
    assert 1 < len(fitter.evaluated) <= 60
    other = _fitter(OwnLoopFitter)
    run_fitter(other, 60, seed=5, replicas=1)
    assert other.handed == [{"budget": 60, "replicas": 1, "seed": 5}]
    assert 1 < len(other.evaluated) <= 60


def test_a_fit_anneal_without_seed_runs_on_the_defaults():
    fitter = _fitter(BudgetOnlyFitter)
    run_fitter(fitter, 60)
    assert fitter.handed == [{"budget": 60}]
    assert 1 < len(fitter.evaluated) <= 60


def _benchmarked(budget, **options):
    """A fitter without ``fit_anneal``, run by ``run_benchmark`` with ``options``."""
    fitter = _fitter(NextFitter)
    context = {"fitter": fitter, "budget": budget, "initial_params": fitter.initial_parameters}
    run_benchmark(context, **options)
    return fitter


@pytest.mark.parametrize(
    "kind, options, handed",
    [
        (PositionalBudgetFitter, {}, {"method": "portfolio", "seed": 42}),
        (PositionalBudgetFitter, {"method": "sa", "seed": 3}, {"method": "boltzmann", "seed": 3}),
        (PositionalBudgetKwargsFitter, {}, {}),
        (PositionalBudgetKwargsFitter, {"method": "gsa", "seed": 3}, {"method": "gsa", "seed": 3}),
    ],
    ids=["keyword-only-defaults", "keyword-only-sa", "kwargs-defaults", "kwargs-gsa"],
)
def test_a_positional_only_budget_goes_by_position(kind, options, handed):
    own = _fitter(kind)
    run_fitter(own, 60, **options)
    assert own.handed == [{"budget": 60, **handed}]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, _benchmarked(60, **handed))


def test_a_fit_anneal_over_run_benchmark_without_seed_runs_its_own_seed():
    own, plain = _fitter(NoSeedFitter), _fitter(NextFitter)
    run_fitter(own, 60)
    run_fitter(plain, 60)
    assert own.handed == [{"method": "portfolio"}]
    assert 1 < len(own.evaluated) <= 60
    assert _same_run(own, _benchmarked(60, method="portfolio", seed=0))
    assert not _same_run(own, plain)


def test_a_positional_only_seed_gets_nothing_by_default():
    fitter = _fitter(PositionalSeedFitter)
    run_fitter(fitter, 60)
    assert fitter.handed == [{"budget": 60, "seed": 0}]
    assert 1 < len(fitter.evaluated) <= 60


@pytest.mark.parametrize(
    "kind, options, message",
    [
        (
            NoPresetFitter,
            {"method": "sa", "preset": SLOW_FAST},
            (
                "run_fitter() cannot pass the Fast preset to NoPresetFitter.fit_anneal(), "
                "which takes no 'preset'"
            ),
        ),
        (
            BenchmarkFitter,
            {"x0": {"x": np.zeros(2), "eps": 1.0}, "bound_span": 0.5},
            (
                "run_fitter() cannot pass x0 and bound_span to BenchmarkFitter.fit_anneal(), "
                "which takes no 'x0' or 'bound_span'"
            ),
        ),
        (
            OwnLoopFitter,
            {"method": "sa"},
            (
                "run_fitter() cannot pass method 'sa' to OwnLoopFitter.fit_anneal(), which "
                "takes no 'method'; leave method out to run it as it stands"
            ),
        ),
        (
            OwnLoopFitter,
            {"method": "portfolio", "preset": SLOW_FAST, "steps_per_epoch": 10},
            (
                "run_fitter() cannot pass method 'portfolio', the Fast preset and "
                "steps_per_epoch to OwnLoopFitter.fit_anneal(), which takes no 'method', "
                "'preset' or 'steps_per_epoch'; leave method out to run it as it stands"
            ),
        ),
        (
            OwnLoopFitter,
            {"preset": SLOW_FAST},
            (
                "run_fitter() cannot pass the Fast preset to OwnLoopFitter.fit_anneal(), "
                "which takes no 'preset'"
            ),
        ),
        (
            BudgetOnlyFitter,
            {"seed": 7},
            (
                "run_fitter() cannot pass seed 7 to BudgetOnlyFitter.fit_anneal(), which "
                "takes no 'seed'"
            ),
        ),
        (
            BenchmarkFitter,
            {"preset": Boltzmann(), "x0": 1.0},
            "run_fitter() cannot pass x0 to BenchmarkFitter.fit_anneal(), which takes no 'x0'",
        ),
        (
            NoPresetFitter,
            {"method": "boltzmann", "preset": Fast(), "x0": 1.0},
            "run_fitter() cannot pass x0 to NoPresetFitter.fit_anneal(), which takes no 'x0'",
        ),
        (
            LateBudgetFitter,
            {},
            (
                "run_fitter() cannot pass budget to LateBudgetFitter.fit_anneal(), which "
                "takes 'budget' only by position"
            ),
        ),
        (
            LateBudgetFitter,
            {"x0": 1.0},
            (
                "run_fitter() cannot pass budget and x0 to LateBudgetFitter.fit_anneal(), "
                "which takes no 'x0' and takes 'budget' only by position"
            ),
        ),
        (
            PositionalSeedFitter,
            {"seed": 7},
            (
                "run_fitter() cannot pass seed 7 to PositionalSeedFitter.fit_anneal(), which "
                "takes 'seed' only by position"
            ),
        ),
        (
            PositionalPresetFitter,
            {"method": "sa", "preset": SLOW_FAST},
            (
                "run_fitter() cannot pass the Fast preset to "
                "PositionalPresetFitter.fit_anneal(), which takes 'preset' only by position"
            ),
        ),
    ],
    ids=[
        "preset",
        "keywords",
        "method",
        "all-three",
        "preset-default-method",
        "seed",
        "x0-preset-the-portfolio-ignores",
        "x0-preset-boltzmann-ignores",
        "budget-after-a-positional-parameter",
        "budget-after-a-positional-parameter-and-x0",
        "positional-only-seed-beside-kwargs",
        "positional-only-preset",
    ],
)
def test_an_option_a_fit_anneal_cannot_take_is_refused_before_anything_is_evaluated(
    kind, options, message
):
    fitter = _fitter(kind)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        with pytest.raises(TypeError) as caught:
            run_fitter(fitter, 60, **options)
    assert str(caught.value) == message
    assert [str(w.message) for w in warned] == []
    assert fitter.handed == [] and fitter.calls == [] and fitter.evaluated == []


def _logged(method):
    def wrapper(self, *args, **kwargs):
        self.calls.append("wrapper")
        return method(self, *args, **kwargs)

    return wrapper


def test_a_wrapper_made_with_functools_wraps_is_read_through():
    class Wrapped(NoPresetFitter):
        fit_anneal = functools.wraps(NoPresetFitter.fit_anneal)(_logged(NoPresetFitter.fit_anneal))

    fitter = _fitter(Wrapped)
    with pytest.raises(TypeError) as caught:
        run_fitter(fitter, 60, method="sa", preset=SLOW_FAST)
    assert str(caught.value) == (
        "run_fitter() cannot pass the Fast preset to Wrapped.fit_anneal(), which takes no 'preset'"
    )
    assert fitter.calls == [] and fitter.handed == []
    run_fitter(fitter, 60)
    assert fitter.handed == [{"method": "portfolio", "seed": 42}]
    assert fitter.calls[0] == "wrapper" and 1 < len(fitter.evaluated) <= 60


def test_a_wrapper_without_functools_wraps_gets_what_the_caller_gave():
    class Unwrapped(NoPresetFitter):
        fit_anneal = _logged(NoPresetFitter.fit_anneal)

    fitter = _fitter(Unwrapped)
    run_fitter(fitter, 60)
    assert fitter.handed == [{"method": "portfolio", "seed": 0}]
    assert fitter.calls[0] == "wrapper" and 1 < len(fitter.evaluated) <= 60


def test_a_fit_anneal_whose_signature_cannot_be_read_gets_what_the_caller_gave():
    class Unreadable:
        __signature__ = 0

        def __call__(self, **kwargs):
            return kwargs

    class Opaque:
        fit_anneal = Unreadable()

    assert run_fitter(Opaque(), 60) == {"budget": 60}
    assert run_fitter(Opaque(), 60, method="sa", preset=SLOW_FAST, x0=1.0) == {
        "budget": 60,
        "method": "fast",
        "preset": SLOW_FAST,
        "x0": 1.0,
    }


def test_an_unknown_method_is_refused_before_a_fit_anneal_is_called():
    own = _fitter(BenchmarkFitter)
    with pytest.raises(ValueError, match="method must be one of 'global_optimize', 'sa'"):
        run_fitter(own, 60, method="anneal")
    assert own.handed == [] and own.calls == []
