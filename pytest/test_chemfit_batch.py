"""Batch replicas through the ChemFit bridges.

With ``batch_size`` above one, ``fit_anneal``, ``fit_chemfit`` and
``run_benchmark`` run the portfolio as that many batch replicas of
``anneal.minimize``: the fitter is initialised with ``init(batch_size=...)``
and each batch is one ``evaluate`` of a list and one ``step``.
"""

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from anneal.chemfit import fit_anneal, fit_chemfit  # noqa: E402
from chemfit_doubles import (  # noqa: E402
    CLASSICAL,
    BatchFitter,
    NextFitter,
    ReleasedFitter,
    drive,
    same_params,
    sum_of_squares,
)

BATCH_ENTRIES = ("fit_anneal", "fit_chemfit", "run_benchmark")


def _installed_fitter():
    """The installed ChemFit ``Fitter`` and the batch methods it lacks.

    ChemFit 3.1.1 has ``ask`` and ``tell`` but neither ``evaluate`` nor
    ``step``, so the bridges refuse its batches.
    """
    fitter_type = pytest.importorskip("chemfit.fitter").Fitter
    missing = [name for name in ("evaluate", "step") if not callable(getattr(fitter_type, name, None))]
    return fitter_type, missing


def _fitter(kind=BatchFitter, **kwargs):
    return kind(
        {"positions": np.array([[0.1, -0.4, 0.3], [0.9, -1.25, 0.2]]), "eps": 0.7},
        {"positions": (-3.0, 3.0), "eps": (0.0, 1.0)},
        **kwargs,
    )


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


@pytest.mark.parametrize("driver", CLASSICAL)
@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_batches_need_the_portfolio(entry, driver):
    fitter = _fitter()
    with pytest.raises(ValueError, match="batch_size above one needs the portfolio"):
        drive(entry, fitter, 50, driver=driver, batch_size=2)
    assert fitter.calls == []


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_batches_need_a_fitter_with_evaluate_and_step(entry):
    fitter = _fitter(ReleasedFitter)
    with pytest.raises(TypeError, match="needs a fitter with evaluate and step"):
        drive(entry, fitter, 50, batch_size=2)
    assert fitter.calls == []


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


@pytest.mark.parametrize("entry", BATCH_ENTRIES)
def test_a_chemfit_fitter_evaluates_each_batch_through_its_schedule(entry):
    fitter_type, missing = _installed_fitter()
    if missing:
        pytest.skip(f"the installed ChemFit Fitter lacks {' and '.join(missing)}, which batches need")
    start = np.array([0.4, -0.3, 0.8, 0.1])
    seen = []

    def objective(params):
        seen.append(np.array(params["x"], dtype=np.float64))
        return sum_of_squares(params)

    fitter = fitter_type(objective, initial_params={"x": start}, bounds={"x": (-1.0, 1.0)})
    result = drive(entry, fitter, 120, batch_size=4)
    assert seen[0].tobytes() == start.tobytes()
    assert len(seen) <= 120
    assert sum(context.n_evals for context in fitter.contexts) == len(seen)
    assert sum_of_squares(result) == min(sum_of_squares({"x": x}) for x in seen)
