"""Recording doubles for the two ChemFit ``Fitter`` session protocols.

``NextFitter`` speaks the ``init`` / ``evaluate`` / ``step`` / ``finish`` of
ChemFit's source from commit c57d6f5, which no release has yet;
``ReleasedFitter`` speaks the ``init`` / ``ask`` / ``tell`` / ``finish`` of
ChemFit 3.1.1, the first release with a session. Both log every protocol
call and keep a deep copy of each evaluated parameter dict, so a test can
check what a bridge sent, in which order, and what ``finish`` received.
``BatchFitter`` adds the batches of ChemFit's source from commit 3b65fe0:
``init(batch_size=...)``, and ``evaluate`` of a list of at most that many
candidates returning their losses. ``WorkersFitter`` has the same
``evaluate`` but an ``init`` that takes ``num_workers`` instead.
``installed_fitter_type`` gives the tests that run on the installed ChemFit
its ``Fitter``, and skips them where the bridges refuse it.
"""

import copy

import numpy as np
import pytest

from anneal.chemfit import fit_anneal, fit_chemfit, run_benchmark, run_fitter

ENTRIES = ("fit_anneal", "fit_chemfit", "run_benchmark", "run_fitter")
CLASSICAL = ("boltzmann", "fast", "gsa")
DRIVERS = ("portfolio", *CLASSICAL)


def leaves(params):
    """Every non-dict leaf of a nested dict, depth first."""
    for value in params.values():
        if isinstance(value, dict):
            yield from leaves(value)
        else:
            yield value


def sum_of_squares(params, target=0.25):
    total = 0.0
    for value in leaves(params):
        diff = np.asarray(value, dtype=np.float64) - target
        total += float(np.sum(diff * diff))
    return total


class Recorder:
    """A fitter with ``init`` and ``finish`` but neither evaluation protocol.

    ``fail`` maps a protocol method name to ``(n, exception)``: the ``n``-th
    call of that method raises ``exception``.
    """

    def __init__(self, initial_parameters, bounds=None, loss=sum_of_squares, fail=None):
        self.initial_parameters = initial_parameters
        self.bounds = {} if bounds is None else bounds
        self.loss = loss
        self.fail = dict(fail or {})
        self.calls = []
        self.evaluated = []
        self.losses = []
        self.finished_with = None

    def init(self):
        self.calls.append("init")

    def finish(self, opt_params=None):
        self.finished_with = opt_params
        self._record("finish")
        return opt_params

    def _record(self, name):
        self.calls.append(name)
        due = self.fail.get(name)
        if due is not None and self.calls.count(name) == due[0]:
            raise due[1]

    def _evaluate(self, name, params):
        self._record(name)
        self.evaluated.append(copy.deepcopy(params))
        loss = self.loss(params)
        self.losses.append(loss)
        return loss


class NextFitter(Recorder):
    """ChemFit's source from c57d6f5: ``evaluate`` per candidate, ``step`` per step."""

    def evaluate(self, parameters, context_index=0):
        return self._evaluate("evaluate", parameters)

    def step(self, step=None):
        self._record("step")


class BatchFitter(NextFitter):
    """ChemFit's source from 3b65fe0, with batches; ``batches`` logs their lengths.

    A list is one logged ``evaluate``, as one call to ChemFit's schedule;
    more candidates than ``init`` allowed is a ValueError, as in ChemFit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = None
        self.batches = []

    def init(self, batch_size=1):
        self.calls.append("init")
        self.batch_size = batch_size

    def evaluate(self, parameters, context_index=0):
        if not isinstance(parameters, list):
            return super().evaluate(parameters, context_index)
        if len(parameters) > self.batch_size:
            raise ValueError(
                "a batch cannot contain more candidates than the configured "
                f"batch_size ({self.batch_size})"
            )
        self._record("evaluate")
        self.batches.append(len(parameters))
        losses = []
        for params in parameters:
            self.evaluated.append(copy.deepcopy(params))
            losses.append(self.loss(params))
        self.losses.extend(losses)
        return losses


class WorkersFitter(BatchFitter):
    """ChemFit's source from commit c57d6f5 until 3b65fe0, as at b41faa4.

    ``evaluate`` takes a list and ``step`` follows, but ``init`` takes
    ``num_workers`` and ``contexts``, not ``batch_size``.
    """

    def init(self, num_workers=1, contexts=None):
        self.calls.append("init")
        self.batch_size = num_workers


class ReleasedFitter(Recorder):
    """ChemFit 3.1.1: ``ask`` per candidate, ``tell`` per optimizer step."""

    def ask(self, parameters, context_index=0):
        return self._evaluate("ask", parameters)

    def tell(self, step=None):
        self._record("tell")


PROTOCOLS = (NextFitter, ReleasedFitter)


def installed_fitter_type():
    """The installed ChemFit ``Fitter``; skip without ChemFit or a session pair.

    ChemFit 3.1.0 and earlier have neither ``ask`` and ``tell`` nor
    ``evaluate`` and ``step``, so the bridges refuse their ``Fitter``.
    """
    fitter_type = pytest.importorskip("chemfit.fitter").Fitter
    for pair in (("ask", "tell"), ("evaluate", "step")):
        if all(callable(getattr(fitter_type, name, None)) for name in pair):
            return fitter_type
    pytest.skip(
        "the installed ChemFit Fitter has neither ask and tell, first released in "
        "3.1.1, nor evaluate and step, so the bridges refuse it"
    )


def protocol_names(fitter_type):
    """The ``(evaluate, step)`` method names a fitter type is driven through."""
    if issubclass(fitter_type, NextFitter):
        return "evaluate", "step"
    return "ask", "tell"


def drive(entry, fitter, budget, driver="portfolio", **kwargs):
    """Run the bridge ``entry`` on ``fitter`` with the driver (method) ``driver``."""
    if entry == "fit_anneal":
        return fit_anneal(fitter, budget, driver=driver, **kwargs)
    if entry == "fit_chemfit":
        return fit_chemfit(fitter, budget, method=driver, **kwargs)
    if entry == "run_benchmark":
        context = {
            "fitter": fitter,
            "budget": budget,
            "initial_params": fitter.initial_parameters,
        }
        return run_benchmark(context, method=driver, **kwargs)
    if entry == "run_fitter":
        method = "global_optimize" if driver == "portfolio" else driver
        return run_fitter(fitter, budget, method=method, **kwargs)
    raise ValueError(f"unknown bridge {entry!r}")


def same_params(left, right):
    """Whether two parameter dicts hold the same leaves, type and bits alike."""
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and list(left) == list(right)
            and all(same_params(left[key], right[key]) for key in left)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    return np.asarray(left).tobytes() == np.asarray(right).tobytes()
