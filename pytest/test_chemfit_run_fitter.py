"""``run_fitter``: the options it passes on to ``fit_anneal``, and no others.

A fitter with its own ``fit_anneal`` is handed to it with what the signature
of that method takes, the method under the names of ``run_benchmark``, which
a ``fit_anneal`` that runs ``run_benchmark`` takes. The first three doubles
below have the signatures such methods have had: one that runs
``run_benchmark`` with a ``preset`` parameter, one without it, and one with
neither ``method`` nor ``preset`` that runs ``anneal.minimize`` itself. The
others try what else a signature can do: take the budget alone, leave out
the seed or the budget, or take parameters only by position. The next has
neither session pair, which the bridges that drive the session themselves
refuse before ``init``, while ``run_fitter`` hands it to its ``fit_anneal``
with the budget and the seed unchecked. The last two run ``run_benchmark``
again after it refuses a budget, and on a worker thread.
"""

import concurrent.futures
import functools
import inspect
import json
import os
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest

pytest.importorskip("anneal")

from anneal import Boltzmann, Fast, minimize  # noqa: E402
from anneal.chemfit import fit_anneal, run_benchmark, run_fitter  # noqa: E402
from chemfit_doubles import (  # noqa: E402
    ENTRIES,
    NextFitter,
    Recorder,
    ReleasedFitter,
    drive,
    same_params,
)


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


class LateBudgetKwargsFitter(Handed):
    """Takes a scale, a shift, the budget and the seed only by position, and ``**options``."""

    def fit_anneal(self, scale, shift, budget, seed=0, /, **options):
        self.handed.append({"scale": scale, "shift": shift, "budget": budget, "seed": seed})
        return self._minimize(budget, seed)


class NoBudgetFitter(Handed):
    """Takes a number of steps in place of a budget."""

    def fit_anneal(self, steps=60):
        self.handed.append({"steps": steps})
        return self._minimize(steps, seed=0)


class NoPairFitter(Recorder):
    """Has neither session pair, and a ``fit_anneal`` that calls ``init`` and drives nothing."""

    def fit_anneal(self, budget, seed=0):
        self.init()
        return {"budget": budget, "seed": seed}


class RetryFitter(Handed):
    """Runs ``run_benchmark``, and again with a budget of 60 when it refuses the one given."""

    def fit_anneal(self, budget, *, method="portfolio", seed=0):
        self.handed.append({"budget": budget, "method": method, "seed": seed})
        try:
            return self._benchmark(budget, method=method, seed=seed)
        except ValueError:
            return self._benchmark(60, method=method, seed=seed)


class ThreadFitter(Handed):
    """Runs ``run_benchmark`` on a worker thread; ``threads`` logs where ``init`` ran."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads = []

    def init(self):
        self.threads.append(threading.get_ident())
        super().init()

    def fit_anneal(self, budget, *, method="portfolio", seed=0):
        self.handed.append({"budget": budget, "method": method, "seed": seed})
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(self._benchmark, budget, method=method, seed=seed).result()


def _fitter(kind=ReleasedFitter, bounds=None, fail=None):
    return kind(
        {"x": np.array([0.2, -0.1]), "eps": 1.0},
        {"x": (-1.0, 1.0), "eps": (0.5, 2.0)} if bounds is None else bounds,
        fail=fail,
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


@pytest.mark.parametrize(
    "kind, why_not",
    [(BenchmarkFitter, None), (NoPresetFitter, "NoPresetFitter.fit_anneal() takes no 'preset'")],
    ids=["preset", "no-preset"],
)
@pytest.mark.parametrize(
    "budget, options, bounds, message",
    [
        (0, {}, None, "budget must be positive, got 0"),
        (60, {"seed": -1}, None, "seed must be at least 0, got -1"),
        (60, {"steps_per_epoch": 0}, None, "steps_per_epoch must be positive, got 0"),
        (
            60,
            {},
            {"x": (1.0, -1.0), "eps": (0.5, 2.0)},
            "the bounds of x[0] are empty, the lower above the upper: lower=1.0, upper=-1.0",
        ),
    ],
    ids=["budget-0", "seed-minus-1", "steps-per-epoch-0", "empty-bounds"],
)
def test_a_refusal_inside_a_fit_anneal_comes_without_the_warning(
    kind, why_not, budget, options, bounds, message
):
    own, plain = _fitter(kind, bounds), _fitter(NextFitter, bounds)
    for fitter in (own, plain):
        for action in ("always", "error"):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter(action)
                with pytest.raises(ValueError) as caught:
                    run_fitter(fitter, budget, preset=Fast(), **options)
            assert str(caught.value) == message
            assert record == []
        assert fitter.calls == [] and fitter.evaluated == []
    assert [h["method"] for h in own.handed] == ["portfolio", "portfolio"]


@pytest.mark.parametrize(
    "kind, why_not",
    [(BenchmarkFitter, None), (NoPresetFitter, "NoPresetFitter.fit_anneal() takes no 'preset'")],
    ids=["preset", "no-preset"],
)
def test_the_warning_for_a_fit_anneal_comes_once_its_bridge_checks_pass(kind, why_not):
    warning = _ignored_fast("global_optimize", "the portfolio", why_not)
    fitter = _fitter(kind)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        run_fitter(fitter, 60, preset=Fast())
    assert [(w.category, str(w.message)) for w in record] == [(FutureWarning, warning)]
    assert (record[0].filename, record[0].lineno) == (__file__, line)
    assert fitter.calls[0] == "init" and 1 < len(fitter.evaluated) <= 60
    first = _fitter(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FutureWarning) as caught:
            run_fitter(first, 60, preset=Fast())
    assert str(caught.value) == warning
    assert len(first.handed) == 1 and first.calls == []


_CALLER = '''\
import numpy as np
from anneal import Fast
from anneal.chemfit import run_benchmark, run_fitter


class Own:
    def __init__(self):
        self.initial_parameters = {"x": [0.2, -0.1], "eps": 1.0}
        self.bounds = {"x": (-1.0, 1.0), "eps": (0.5, 2.0)}
        self.evaluations = 0

    def init(self, batch_size=1):
        pass

    def evaluate(self, params):
        self.evaluations += 1
        return float(np.sum((np.asarray(params["x"]) - 0.3) ** 2) + (params["eps"] - 1.2) ** 2)

    def step(self):
        pass

    def finish(self, params=None):
        return params

    def fit_anneal(self, budget, *, method="portfolio", seed=0):
        context = {"fitter": self, "budget": budget, "initial_params": self.initial_parameters}
        return run_benchmark(context, method=method, seed=seed)


def call(fitter):
    return run_fitter(fitter, 40, preset=Fast())
'''

# IPython and Jupyter run a cell in a fresh module's namespace, whose
# __loader__ and __spec__ are None; a script's __main__ has a loader but no
# spec.
_DRIVE = '''\
import importlib.util
import json
import sys
import types
import warnings

where, path = sys.argv[1], sys.argv[2]
with open(path) as fh:
    source = fh.read()
if where == "script":
    namespace, filename = globals(), path
    exec(compile(source, filename, "exec"), namespace)
elif where == "cell":
    cell = types.ModuleType("__main__")
    cell.__dict__["__builtins__"] = __builtins__
    namespace, filename = cell.__dict__, "<ipython-input-1>"
    exec(compile(source, filename, "exec"), namespace)
else:
    spec = importlib.util.spec_from_file_location("caller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = module.__dict__
out = {"names": [namespace.get(name) is not None for name in ("__loader__", "__spec__")]}
fitter = namespace["Own"]()
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always", FutureWarning)
    best = namespace["call"](fitter)
out["record"] = [[w.category.__name__, str(w.message), w.filename, w.lineno] for w in record]
out["ran"] = [sorted(best), fitter.evaluations]
fitter = namespace["Own"]()
try:
    namespace["call"](fitter)
except Exception as error:
    out["raised"] = [type(error).__name__, str(error), fitter.evaluations]
print(json.dumps(out))
'''


@pytest.mark.parametrize(
    "where, names",
    [("cell", [False, False]), ("script", [True, False]), ("module", [True, True])],
)
def test_the_warning_for_a_fit_anneal_reaches_its_caller_from_any_namespace(
    tmp_path, where, names
):
    caller, drive = tmp_path / "caller.py", tmp_path / "drive.py"
    caller.write_text(_CALLER)
    drive.write_text(_DRIVE)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    done = subprocess.run(
        [sys.executable, "-W", "error", str(drive), where, str(caller)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout)
    assert out["names"] == names
    warning = _ignored_fast(
        "global_optimize", "the portfolio", "Own.fit_anneal() takes no 'preset'"
    )
    filename = "<ipython-input-1>" if where == "cell" else str(caller)
    line = _CALLER.splitlines().index("    return run_fitter(fitter, 40, preset=Fast())") + 1
    assert out["record"] == [["FutureWarning", warning, filename, line]]
    keys, evaluations = out["ran"]
    assert keys == ["eps", "x"] and 1 < evaluations <= 40
    assert out.get("raised") == ["FutureWarning", warning, 0]


def test_a_call_after_a_refused_one_has_the_warning_once_its_checks_pass():
    warning = _ignored_fast(
        "global_optimize", "the portfolio", "RetryFitter.fit_anneal() takes no 'preset'"
    )
    fitter = _fitter(RetryFitter)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, category, filename, lineno, *rest: (
            fitter.calls.append((category, str(message), filename, lineno))
        )
        line = inspect.currentframe().f_lineno + 1
        run_fitter(fitter, 0, preset=Fast())
    assert fitter.calls[:2] == [(FutureWarning, warning, __file__, line), "init"]
    assert sum(isinstance(call, tuple) for call in fitter.calls) == 1
    assert len(fitter.handed) == 1 and 1 < len(fitter.evaluated) <= 60
    fitter = _fitter(RetryFitter)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FutureWarning) as caught:
            run_fitter(fitter, 0, preset=Fast())
    assert str(caught.value) == warning and fitter.calls == []


def test_a_bridge_on_another_thread_gives_the_warning_there_before_init():
    warning = _ignored_fast(
        "global_optimize", "the portfolio", "ThreadFitter.fit_anneal() takes no 'preset'"
    )
    fitter = _fitter(ThreadFitter)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, category, filename, lineno, *rest: (
            fitter.calls.append((category, str(message), filename, lineno, threading.get_ident()))
        )
        line = inspect.currentframe().f_lineno + 1
        run_fitter(fitter, 60, preset=Fast())
    worker = fitter.threads[0]
    assert worker != threading.get_ident()
    assert fitter.calls[:2] == [(FutureWarning, warning, __file__, line, worker), "init"]
    assert sum(isinstance(call, tuple) for call in fitter.calls) == 1
    assert 1 < len(fitter.evaluated) <= 60
    fitter = _fitter(ThreadFitter)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(FutureWarning) as caught:
            run_fitter(fitter, 60, preset=Fast())
    assert str(caught.value) == warning and fitter.calls == []


def test_a_fit_anneal_that_calls_no_bridge_has_the_warning_once_it_returns():
    events = []

    class OwnLoop:
        def fit_anneal(self, budget, method="portfolio"):
            if budget < 1:
                raise ValueError(f"budget must be at least 1, got {budget}")
            events.append("ran")
            return {"budget": budget, "method": method}

    warning = _ignored_fast(
        "global_optimize", "the portfolio", "OwnLoop.fit_anneal() takes no 'preset'"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda message, *rest, **kw: events.append(str(message))
        assert run_fitter(OwnLoop(), 60, preset=Fast()) == {"budget": 60, "method": "portfolio"}
        assert events == ["ran", warning]
        events.clear()
        with pytest.raises(ValueError, match="^budget must be at least 1, got 0$"):
            run_fitter(OwnLoop(), 0, preset=Fast())
        assert events == []


@pytest.mark.parametrize("entry", ENTRIES)
def test_run_fitter_hands_a_fitter_with_neither_pair_to_its_own_fit_anneal(entry):
    plain, own = _fitter(Recorder), _fitter(NoPairFitter)
    with pytest.raises(TypeError, match="; Recorder has neither pair$"):
        drive(entry, plain, 60)
    assert plain.calls == []
    if entry == "run_fitter":
        assert drive(entry, own, 60) == {"budget": 60, "seed": 42}
        assert own.calls == ["init"]
    else:
        with pytest.raises(TypeError, match="; NoPairFitter has neither pair$"):
            drive(entry, own, 60)
        assert own.calls == []


def test_a_fit_anneal_decides_whether_a_budget_of_0_is_refused_and_when():
    unchecked = _fitter(NoPairFitter)
    assert run_fitter(unchecked, 0, seed=-1) == {"budget": 0, "seed": -1}
    assert unchecked.calls == ["init"]
    late = _fitter(OwnLoopFitter)
    with pytest.raises(ValueError) as caught:
        run_fitter(late, 0)
    assert str(caught.value) == "budget must be a whole number of at least 1, got 0"
    assert late.handed == [{"budget": 0, "replicas": 4, "seed": 42}]
    assert late.calls == ["init"] and late.evaluated == []
    plain = _fitter(NextFitter)
    with pytest.raises(ValueError) as caught:
        run_fitter(plain, 0)
    assert str(caught.value) == "budget must be positive, got 0"
    assert plain.calls == []


def test_a_fit_anneal_decides_what_becomes_of_an_exception_the_fitter_raises():
    error = RuntimeError("evaluate failed")
    raised = _fitter(BenchmarkFitter, fail={"evaluate": (3, error)})
    plain = _fitter(NextFitter, fail={"evaluate": (3, error)})
    for fitter in (raised, plain):
        with pytest.raises(RuntimeError) as caught:
            run_fitter(fitter, 60)
        assert caught.value is error
        assert fitter.calls.count("evaluate") == 3 and "finish" not in fitter.calls
    assert [h["method"] for h in raised.handed] == ["portfolio"]
    scored = _fitter(OwnLoopFitter, fail={"evaluate": (3, error)})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        best = run_fitter(scored, 60)
    warning = (
        "a callback raised 1 exception(s): an objective call that raised was scored "
        "as the worst value and a gradient call as zero; the first was: "
        "RuntimeError: evaluate failed"
    )
    assert [(w.category, str(w.message)) for w in record] == [(RuntimeWarning, warning)]
    assert scored.handed == [{"budget": 60, "replicas": 4, "seed": 42}]
    assert scored.calls[0] == "init" and scored.calls[-1] == "finish"
    assert 3 < scored.calls.count("evaluate") <= 60
    assert best is scored.finished_with


def test_run_fitter_passes_on_an_exception_from_init_inside_a_fit_anneal():
    error = RuntimeError("init failed")
    fitter = _fitter(OwnLoopFitter)

    def init():
        fitter.calls.append("init")
        raise error

    fitter.init = init
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError) as caught:
            run_fitter(fitter, 60)
    assert caught.value is error and record == []
    assert fitter.handed == [{"budget": 60, "replicas": 4, "seed": 42}]
    assert fitter.calls == ["init"] and fitter.evaluated == []


def test_a_fit_anneal_decides_what_becomes_of_a_loss_that_is_not_a_real_number():
    refusal = "the fitter returned a loss of type bool; a loss must be one real number"
    stopped = _fitter(BenchmarkFitter)
    plain = _fitter(NextFitter)
    read = _fitter(OwnLoopFitter)
    for fitter in (stopped, plain, read):
        fitter.loss = lambda params: True
    for fitter in (stopped, plain):
        with pytest.raises(TypeError) as caught:
            run_fitter(fitter, 60)
        assert str(caught.value) == refusal
        assert fitter.calls.count("evaluate") == 1 and "finish" not in fitter.calls
    assert [h["method"] for h in stopped.handed] == ["portfolio"]
    best = run_fitter(read, 60)
    assert read.handed == [{"budget": 60, "replicas": 4, "seed": 42}]
    assert read.calls[0] == "init" and read.calls[-1] == "finish"
    assert 1 < read.calls.count("evaluate") <= 60
    assert best is read.finished_with


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
                "takes 'budget' only by position; 'budget' comes after 'scale', so a "
                "budget passed by position would fill 'scale'"
            ),
        ),
        (
            LateBudgetFitter,
            {"x0": 1.0},
            (
                "run_fitter() cannot pass budget and x0 to LateBudgetFitter.fit_anneal(), "
                "which takes no 'x0' and takes 'budget' only by position; 'budget' comes "
                "after 'scale', so a budget passed by position would fill 'scale'"
            ),
        ),
        (
            LateBudgetKwargsFitter,
            {"seed": 7},
            (
                "run_fitter() cannot pass budget and seed 7 to "
                "LateBudgetKwargsFitter.fit_anneal(), which takes 'budget' and 'seed' only "
                "by position; 'budget' comes after 'scale' and 'shift', so a budget passed "
                "by position would fill 'scale'"
            ),
        ),
        (
            LateBudgetFitter,
            {"method": "sa"},
            (
                "run_fitter() cannot pass budget and method 'sa' to "
                "LateBudgetFitter.fit_anneal(), which takes no 'method' and takes 'budget' "
                "only by position; 'budget' comes after 'scale', so a budget passed by "
                "position would fill 'scale'"
            ),
        ),
        (
            NoBudgetFitter,
            {"method": "sa"},
            (
                "run_fitter() cannot pass budget and method 'sa' to "
                "NoBudgetFitter.fit_anneal(), which takes no 'budget' or 'method'"
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
        "budget-and-seed-after-two-positional-parameters",
        "method-with-a-late-budget",
        "method-without-a-budget",
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
