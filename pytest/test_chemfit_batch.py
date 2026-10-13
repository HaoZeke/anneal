"""Batch replicas through the ChemFit bridges.

With ``batch_size`` above one, ``fit_anneal``, ``fit_chemfit`` and
``run_benchmark`` run the portfolio as that many batch replicas of
``anneal.minimize``: the fitter is initialised with ``init(batch_size=...)``
and each batch is one ``evaluate`` of a list and one ``step``. That they take
batches shows in ``anneal.chemfit.FEATURES`` and in their signatures. When
every other check passes, a batch under a classical driver is refused before
``init`` by a message naming the driver, whatever the fitter has of the
three. Under the portfolio, a fitter without ``evaluate``, ``step`` or
``init``, or whose ``init`` has a signature that can be read and cannot take
``batch_size``, has its batch refused before ``init`` by a message naming
what it lacks, as ChemFit 3.1.1's lacks all three and the one of ChemFit's
source from commit c57d6f5 until 3b65fe0 lacks an ``init`` that takes
``batch_size``, and ``batch_size=1`` still fits. Any other ``init``, such as
one whose signature cannot be read or one behind a wrapper of
``(*args, **kwargs)`` written without ``functools.wraps``, is not named there
and meets ``batch_size`` only at its call, where what it raises stops the
fit. An ``evaluate`` that takes a single candidate shows only when it is
called: it gets the start as a list of one, after ``init``, and the fit stops
there.
"""

import functools
import inspect
import re
import warnings

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from anneal.chemfit import fit_anneal, fit_chemfit  # noqa: E402
from chemfit_doubles import (  # noqa: E402
    CLASSICAL,
    BatchFitter,
    NextFitter,
    ReleasedFitter,
    WorkersFitter,
    drive,
    installed_fitter_type,
    same_params,
    sum_of_squares,
)

BATCH_ENTRIES = ("fit_anneal", "fit_chemfit", "run_benchmark")


def _listed(names):
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _refusal(name, lacks):
    return (
        "batch_size above one needs a fitter with evaluate, step and an init "
        "that takes batch_size, which ChemFit's source has from commit 3b65fe0 "
        f"and no release has yet; {name} lacks {lacks}"
    )


_UNEXPECTED_BATCH_SIZE = "init() got an unexpected keyword argument 'batch_size'"


def _installed_fitter():
    """The installed ChemFit ``Fitter`` and what it lacks of a batch.

    ChemFit 3.1.1 has ``ask`` and ``tell`` and an ``init`` that takes
    ``num_workers``. ChemFit's source has ``evaluate`` and ``step`` from
    commit c57d6f5 and an ``init`` that takes ``batch_size`` from 3b65fe0.
    """
    fitter_type = installed_fitter_type()
    init = getattr(fitter_type, "init", None)
    takes = callable(init) and "batch_size" in inspect.signature(init).parameters
    lacks = [name for name in ("evaluate", "step") if not callable(getattr(fitter_type, name, None))]
    lacks += [] if takes else ["an init that takes batch_size"]
    return fitter_type, lacks


def _fitter(kind=BatchFitter, **kwargs):
    return kind(
        {"positions": np.array([[0.1, -0.4, 0.3], [0.9, -1.25, 0.2]]), "eps": 0.7},
        {"positions": (-3.0, 3.0), "eps": (0.0, 1.0)},
        **kwargs,
    )


def test_the_bridges_say_they_take_batches_by_features_and_by_signature():
    assert "FEATURES" in anneal.chemfit.__all__
    assert isinstance(anneal.chemfit.FEATURES, frozenset)
    assert "batch_size" in anneal.chemfit.FEATURES
    for name in BATCH_ENTRIES:
        assert "batch_size" in inspect.signature(getattr(anneal.chemfit, name)).parameters


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_each_batch_is_one_evaluate_of_a_list_and_one_step(entry):
    fitter = _fitter()
    drive(entry, fitter, 200, batch_size=4)
    assert fitter.batch_size == 4
    assert fitter.calls[0] == "init" and fitter.calls[-1] == "finish"
    assert fitter.calls[1:-1] == ["evaluate", "step"] * len(fitter.batches)
    assert fitter.batches[0] == 1
    assert set(fitter.batches) <= {1, 2, 3, 4} and 4 in fitter.batches
    assert sum(fitter.batches) == len(fitter.evaluated) <= 200


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_the_first_evaluation_is_the_initial_parameters_bit_for_bit(entry, batch_size):
    initial = {
        "positions": np.array([[0.1, -0.0, 1 / 3], [2.0**-30, -1.25, 5e-324]]),
        "eps": 0.7,
        "f32": np.float32(1 / 3),
        "edge": 3.0,
    }
    bounds = {
        "positions": (-3.0, 3.0),
        "eps": (0.0, 1.0),
        "f32": (0.0, 1.0),
        "edge": (-3.0, 3.0),
    }
    fitter = BatchFitter(initial, bounds)
    drive(entry, fitter, 40, batch_size=batch_size)
    first = fitter.evaluated[0]
    assert first["positions"].tobytes() == initial["positions"].tobytes()
    assert np.float64(first["eps"]).tobytes() == np.float64(0.7).tobytes()
    assert np.float32(first["f32"]).tobytes() == initial["f32"].tobytes()
    assert first["edge"] == 3.0
    assert fitter.batches[:1] == ([1] if batch_size > 1 else [])


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_batches_stay_in_the_box_and_hold_a_fixed_parameter(entry):
    fitter = BatchFitter(
        {"x": np.array([0.5, -0.25, 2.0]), "eps": 0.3},
        {"x": (np.array([-1.0, -1.0, 2.0]), np.array([1.0, 1.0, 2.0])), "eps": (0.0, 1.0)},
    )
    drive(entry, fitter, 120, batch_size=3)
    x = np.array([params["x"] for params in fitter.evaluated])
    eps = np.array([params["eps"] for params in fitter.evaluated])
    assert np.all((x[:, :2] >= -1.0) & (x[:, :2] <= 1.0))
    assert np.all(x[:, 2] == 2.0)
    assert np.all((eps >= 0.0) & (eps <= 1.0))
    assert max(fitter.batches) == 3


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_the_batch_fit_finishes_with_the_best_evaluated_parameters(entry):
    fitter = _fitter()
    result = drive(entry, fitter, 150, batch_size=4)
    best = fitter.evaluated[int(np.argmin(fitter.losses))]
    assert same_params(result, best)
    assert fitter.finished_with is result


def test_a_seed_replays_the_batches():
    runs = [_fitter(), _fitter()]
    for fitter in runs:
        fit_anneal(fitter, 150, seed=7, batch_size=4)
    first, second = runs
    assert first.batches == second.batches
    assert len(first.evaluated) == len(second.evaluated)
    assert all(same_params(a, b) for a, b in zip(first.evaluated, second.evaluated))


def test_the_batches_are_those_of_minimize_with_as_many_replicas():
    start = np.array([0.4, -0.3, 0.8])
    fitter = BatchFitter({"x": start.copy()}, {"x": (-1.0, 1.0)})
    fit_anneal(fitter, 120, seed=3, batch_size=4)

    batches = []

    class Objective:
        def __call__(self, x):
            return sum_of_squares({"x": np.asarray(x)})

        def eval_batch(self, rows):
            batches.append(np.array(rows))
            return [self(row) for row in rows]

    bounds = np.column_stack([np.full(3, -1.0), np.full(3, 1.0)])
    anneal.minimize(Objective(), start, bounds, budget=120, seed=3, replicas=4)
    assert [len(rows) for rows in batches] == fitter.batches
    sent = np.array([params["x"] for params in fitter.evaluated])
    assert np.array_equal(np.concatenate(batches), sent)


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_batch_size_one_keeps_the_single_candidate_path(entry):
    batch, single = _fitter(), _fitter(NextFitter)
    drive(entry, batch, 80, batch_size=1)
    drive(entry, single, 80)
    assert batch.batch_size == 1 and batch.batches == []
    assert batch.calls == single.calls
    assert len(batch.evaluated) == len(single.evaluated)
    assert all(same_params(a, b) for a, b in zip(batch.evaluated, single.evaluated))


def test_a_fit_with_every_parameter_fixed_is_one_batch_of_one():
    fitter = BatchFitter({"x": np.array([0.5, -0.5])}, {"x": (np.array([0.5, -0.5]),) * 2})
    result = fit_chemfit(fitter, 30, batch_size=4)
    assert fitter.calls == ["init", "evaluate", "step", "finish"]
    assert fitter.batches == [1]
    assert result["x"].tobytes() == np.array([0.5, -0.5]).tobytes()


@pytest.mark.parametrize("kind", [BatchFitter, ReleasedFitter])
@pytest.mark.parametrize("driver", CLASSICAL)
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_batch_under_a_classical_driver_is_refused_before_init_naming_the_driver(
    entry, driver, kind
):
    fitter = _fitter(kind)
    with pytest.raises(ValueError) as refused:
        drive(entry, fitter, 50, driver=driver, batch_size=2)
    assert str(refused.value) == (
        "batch_size above one needs the portfolio, the one driver that runs "
        f"batch replicas; {driver} runs one chain"
    )
    assert fitter.calls == []


@pytest.mark.parametrize(
    "kind, lacks",
    [
        (ReleasedFitter, "evaluate, step and an init that takes batch_size"),
        (NextFitter, "an init that takes batch_size"),
        (WorkersFitter, "an init that takes batch_size"),
    ],
)
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_batch_is_refused_before_init_naming_what_the_fitter_lacks(entry, kind, lacks):
    fitter = _fitter(kind)
    with pytest.raises(TypeError, match=re.escape(_refusal(kind.__name__, lacks))):
        drive(entry, fitter, 50, batch_size=2)
    assert fitter.calls == []


def test_run_benchmark_refuses_a_batch_from_a_fitter_without_step():
    class EvaluateOnly(BatchFitter):
        step = None

    fitter = _fitter(EvaluateOnly)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        with pytest.raises(TypeError) as refused:
            drive("run_benchmark", fitter, 50, batch_size=2)
    assert str(refused.value) == _refusal("EvaluateOnly", "step")
    assert [str(w.message) for w in warned] == []
    assert fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_fitter_without_init_cannot_take_a_batch(entry):
    class NoInit(BatchFitter):
        init = None

    fitter = _fitter(NoInit)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        with pytest.raises(TypeError) as refused:
            drive(entry, fitter, 50, batch_size=2)
    assert str(refused.value) == _refusal("NoInit", "an init that takes batch_size")
    assert [str(w.message) for w in warned] == []
    assert fitter.calls == [] and fitter.evaluated == []


def _recorded_init(fitter, wraps):
    """Put ``fitter.init`` behind a wrapper of ``(*args, **kwargs)``.

    The wrapper is written with ``functools.wraps`` or without it, and logs
    the keywords of each call in the list returned.
    """
    inits = []
    init = fitter.init

    def recorded_init(*args, **kwargs):
        inits.append(kwargs)
        return init(*args, **kwargs)

    fitter.init = functools.wraps(init)(recorded_init) if wraps else recorded_init
    return inits


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_the_batch_check_reads_an_init_through_functools_wraps(entry):
    fitter = _fitter(WorkersFitter)
    inits = _recorded_init(fitter, wraps=True)
    with pytest.raises(TypeError) as refused:
        drive(entry, fitter, 50, batch_size=2)
    assert str(refused.value) == _refusal("WorkersFitter", "an init that takes batch_size")
    assert inits == [] and fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_an_init_behind_a_wrapper_without_functools_wraps_meets_batch_size_at_its_call(entry):
    fitter = _fitter(WorkersFitter)
    inits = _recorded_init(fitter, wraps=False)
    with pytest.raises(TypeError, match=re.escape(_UNEXPECTED_BATCH_SIZE)):
        drive(entry, fitter, 50, batch_size=2)
    assert inits == [{"batch_size": 2}]
    assert fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_an_init_whose_signature_cannot_be_read_meets_batch_size_at_its_call(entry):
    class UnreadableInit(WorkersFitter):
        def init(self, num_workers=1, contexts=None):
            super().init(num_workers, contexts)

        init.__signature__ = object()

    fitter = _fitter(UnreadableInit)
    with pytest.raises((TypeError, ValueError)):
        inspect.signature(fitter.init)
    with pytest.raises(TypeError, match=re.escape(_UNEXPECTED_BATCH_SIZE)):
        drive(entry, fitter, 50, batch_size=2)
    assert fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_refusal_before_init_does_not_name_an_init_behind_a_wrapper(entry):
    fitter = _fitter(ReleasedFitter)
    inits = _recorded_init(fitter, wraps=False)
    with pytest.raises(TypeError) as refused:
        drive(entry, fitter, 50, batch_size=2)
    assert str(refused.value) == _refusal("ReleasedFitter", "evaluate and step")
    assert inits == [] and fitter.calls == []


def _one_loss(params):
    return 1.0


@pytest.mark.parametrize(
    "loss, error, match",
    [
        (sum_of_squares, AttributeError, "'list' object has no attribute 'values'"),
        (_one_loss, TypeError, "the fitter returned float for a batch of 1"),
    ],
    ids=["raising", "one loss"],
)
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_an_evaluate_of_one_candidate_stops_the_fit_at_its_first_call(entry, loss, error, match):
    class OneCandidate(BatchFitter):
        evaluate = NextFitter.evaluate

    fitter = _fitter(OneCandidate, loss=loss)
    with pytest.raises(error, match=re.escape(match)):
        drive(entry, fitter, 50, batch_size=2)
    assert fitter.batch_size == 2
    assert fitter.calls == ["init", "evaluate"]
    [[start]] = fitter.evaluated
    assert same_params(start, fitter.initial_parameters)


@pytest.mark.parametrize(
    "value, error",
    [(0, ValueError), (-2, ValueError), (2.5, ValueError), ("4", TypeError), (None, TypeError)],
)
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_bad_batch_size_is_refused_before_init(entry, value, error):
    fitter = _fitter()
    with pytest.raises(error, match="batch_size"):
        drive(entry, fitter, 50, batch_size=value)
    assert fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_batch_with_the_wrong_number_of_losses_stops_the_fit(entry):
    class Short(BatchFitter):
        def evaluate(self, parameters, context_index=0):
            losses = super().evaluate(parameters, context_index)
            return losses[:-1] if len(self.batches) == 3 else losses

    fitter = _fitter(Short)
    with pytest.raises(TypeError, match="a batch needs one loss per candidate"):
        drive(entry, fitter, 100, batch_size=4)
    assert fitter.calls.count("evaluate") == 3
    assert fitter.calls[-1] == "evaluate"


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_the_first_fitter_exception_in_a_batch_stops_the_fit(entry):
    fitter = _fitter(fail={"evaluate": (4, RuntimeError("scheduler failed"))})
    with pytest.raises(RuntimeError, match="scheduler failed"):
        drive(entry, fitter, 100, batch_size=4)
    assert fitter.calls.count("evaluate") == 4
    assert fitter.calls[-1] == "evaluate"


def _chemfit_problem(fitter_type):
    """A real ``Fitter`` on a box, its start, and the points its objective saw."""
    start = np.array([0.4, -0.3, 0.8, 0.1])
    seen = []

    def objective(params):
        seen.append(np.array(params["x"], dtype=np.float64))
        return sum_of_squares(params)

    fitter = fitter_type(objective, initial_params={"x": start}, bounds={"x": (-1.0, 1.0)})
    return fitter, start, seen


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_chemfit_fitter_evaluates_each_batch_through_its_schedule(entry):
    fitter_type, lacks = _installed_fitter()
    if lacks:
        pytest.skip(f"the installed ChemFit Fitter lacks {_listed(lacks)}, which batches need")
    fitter, start, seen = _chemfit_problem(fitter_type)
    result = drive(entry, fitter, 120, batch_size=4)
    assert seen[0].tobytes() == start.tobytes()
    assert len(seen) <= 120
    assert sum(context.n_evals for context in fitter.contexts) == len(seen)
    assert sum_of_squares(result) == min(sum_of_squares({"x": x}) for x in seen)


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_chemfit_fitter_that_cannot_take_batches_refuses_one_before_init(entry):
    fitter_type, lacks = _installed_fitter()
    if not lacks:
        pytest.skip("the installed ChemFit Fitter takes batches")
    fitter, _, seen = _chemfit_problem(fitter_type)
    inits = []
    init = fitter.init

    @functools.wraps(init)
    def recorded_init(*args, **kwargs):
        inits.append(kwargs)
        return init(*args, **kwargs)

    fitter.init = recorded_init
    with pytest.raises(TypeError, match=re.escape(_refusal(fitter_type.__name__, _listed(lacks)))):
        drive(entry, fitter, 120, batch_size=4)
    assert inits == []
    assert seen == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_batch_size_one_fits_with_the_installed_chemfit_fitter(entry):
    fitter_type, _ = _installed_fitter()
    fitter, start, seen = _chemfit_problem(fitter_type)
    result = drive(entry, fitter, 120, batch_size=1)
    assert seen[0].tobytes() == start.tobytes()
    assert 0 < len(seen) <= 120
    assert sum(context.n_evals for context in fitter.contexts) == len(seen)
    assert sum_of_squares(result) == min(sum_of_squares({"x": x}) for x in seen)
