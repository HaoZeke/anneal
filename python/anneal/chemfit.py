"""Fit a ChemFit ``Fitter`` with anneal's drivers.

ChemFit fits the parameters of simulation-based objectives, held in nested
mappings. :func:`fit` drives a ``Fitter`` through its session API (``init``,
``evaluate``, ``step`` and ``finish``, on ChemFit's ``next`` branch) with one
of anneal's drivers: the search starts at ``fitter.initial_parameters``, every
candidate lies inside the bounds, at most ``budget`` candidates reach
``fitter.evaluate``, and the result is ``fitter.finish()``, the best candidate
ChemFit evaluated. :func:`objective` returns the flat view of the parameters
that the drivers search.

ChemFit is an optional dependency: this module never imports it and accepts
any object with the same session API.
"""

from __future__ import annotations

import inspect
import math
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import numpy as np

import anneal

if TYPE_CHECKING:
    from chemfit import Fitter

_DRIVERS = {
    "portfolio": "global_optimize",
    "boltzmann": "run",
    "fast": "run",
    "gsa": "run",
    "qmc": "run_qmc",
    "dmc": "dmc_population_optimize",
}
_PRESETS = {"boltzmann": "Boltzmann", "fast": "Fast", "gsa": "Gsa"}

METHODS = tuple(_DRIVERS)
"""Names accepted by ``fit(method=...)``."""


@dataclass(frozen=True)
class _Leaf:
    path: tuple[Any, ...]
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any] | None  # None: a real scalar, restored as a float
    start: int
    stop: int


def _walk(
    tree: Mapping[Any, Any], path: tuple[Any, ...] = ()
) -> Iterator[tuple[tuple[Any, ...], Any]]:
    for key, value in tree.items():
        if isinstance(value, Mapping):
            yield from _walk(value, (*path, key))
        else:
            yield (*path, key), value


def _leaf_dtype(name: str, value: Any) -> np.dtype[Any] | None:
    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
        return value.dtype
    if isinstance(value, Real) and not isinstance(value, bool):
        return None
    got = (
        f"an array of {value.dtype}"
        if isinstance(value, np.ndarray)
        else type(value).__name__
    )
    msg = (
        f"parameter {name!r} must be a real number or a floating-point array, got {got}"
    )
    raise TypeError(msg)


def _leaf_box(
    name: str, pair: Any, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        msg = f"bounds of {name!r} must be a (lower, upper) pair"
        raise ValueError(msg)
    try:
        low, high = (
            np.broadcast_to(np.asarray(b, dtype=np.float64), shape) for b in pair
        )
    except (TypeError, ValueError):
        msg = (
            f"bounds of {name!r} must be numbers or arrays that broadcast to "
            f"its shape {shape}"
        )
        raise ValueError(msg) from None
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        msg = (
            f"parameter {name!r} has a one-sided or infinite bound; anneal "
            "searches a finite box"
        )
        raise ValueError(msg)
    if (low > high).any():
        msg = f"lower bound of {name!r} exceeds its upper bound"
        raise ValueError(msg)
    return low, high


class Objective:
    """Flat view of a ChemFit ``Fitter`` that anneal's drivers search.

    The parameter leaves are laid out in tree order, each raveled in C order,
    as one float64 vector. Calling the view with such a vector evaluates the
    candidate with ``fitter.evaluate`` and then calls ``fitter.step()``.
    Build it with :func:`objective`, and evaluate inside a session opened with
    ``fitter.init``.

    Attributes
    ----------
    fitter : chemfit.Fitter
        The fitter that evaluates the candidates.
    x0 : numpy.ndarray
        ``fitter.initial_parameters`` as a flat vector.
    low, high : numpy.ndarray
        The box from the bounds, aligned with ``x0``.
    n_evals : int
        Candidates sent to ``fitter.evaluate`` so far.
    budget : int or None
        Cap on ``n_evals``; past it the view returns ``inf`` without
        evaluating. ``None`` sets no cap.
    error : BaseException or None
        The first exception raised by ``fitter.evaluate`` or ``fitter.step``.
        It propagates from that call, and afterwards the view returns ``inf``
        without evaluating: anneal's drivers turn an exception raised by the
        objective into ``inf`` and keep proposing, so re-raise ``error`` once
        the driver returns.
    """

    def __init__(
        self, fitter: Fitter[Any], bounds: Mapping[str, Any] | None = None
    ) -> None:
        missing = [
            name
            for name in ("init", "evaluate", "step", "finish")
            if not callable(getattr(fitter, name, None))
        ]
        if missing:
            msg = (
                "fitter needs ChemFit's session API (init, evaluate, step, "
                f"finish), but has no {', '.join(missing)}"
            )
            raise TypeError(msg)
        params = dict(_walk(fitter.initial_parameters))
        override = dict(_walk(bounds or {}))
        for path in override:
            if path not in params:
                name = ".".join(map(str, path))
                msg = f"bounds name {name!r}, which is not a parameter"
                raise ValueError(msg)
        known = {**dict(_walk(fitter.bounds or {})), **override}

        leaves, starts, lows, highs = [], [], [], []
        offset = 0
        for path, value in params.items():
            name = ".".join(map(str, path))
            dtype = _leaf_dtype(name, value)
            if path not in known:
                msg = f"parameter {name!r} has no bound; give it a (lower, upper) pair"
                raise ValueError(msg)
            start = np.asarray(value, dtype=np.float64)
            low, high = _leaf_box(name, known[path], start.shape)
            if not ((low <= start) & (start <= high)).all():
                msg = f"start of {name!r} lies outside its bounds"
                raise ValueError(msg)
            leaves.append(
                _Leaf(path, name, start.shape, dtype, offset, offset + start.size)
            )
            offset += start.size
            starts.append(start.ravel())
            lows.append(low.ravel())
            highs.append(high.ravel())
        if offset == 0:
            msg = "fitter.initial_parameters holds no values to fit"
            raise ValueError(msg)

        self.fitter = fitter
        self.x0 = np.concatenate(starts)
        self.low = np.concatenate(lows)
        self.high = np.concatenate(highs)
        self.n_evals = 0
        self.budget: int | None = None
        self.error: BaseException | None = None
        self._leaves = tuple(leaves)
        self._start_loss: float | None = None

    def _left(self) -> int:
        if self.error is not None:
            return 0
        if self.budget is None:
            return sys.maxsize
        return max(0, self.budget - self.n_evals)

    def __call__(self, x: Any) -> float:
        """Evaluate one flat candidate and step the fitter.

        Returns ``inf`` without evaluating once the budget is spent or an
        evaluation has failed.
        """
        if self._start_loss is not None and np.array_equal(x, self.x0):
            return self._start_loss
        if not self._left():
            return math.inf
        params = self.unflatten(x)
        self.n_evals += 1
        try:
            loss = self.fitter.evaluate(params)
            self.fitter.step()
        except BaseException as exc:
            self.error = exc
            raise
        return float(loss)

    def eval_batch(self, X: Any) -> np.ndarray:
        """Evaluate the rows of ``X`` in candidate batches.

        A batch holds as many rows as the session has candidate slots
        (``len(fitter.contexts)``, the ``batch_size`` given to ``fitter.init``)
        and reaches ChemFit in one ``fitter.evaluate`` call followed by one
        ``fitter.step()``, so a concurrent ChemFit schedule evaluates its
        candidates in parallel. anneal's population drivers call this method
        for their walker batches. Rows past the budget or after a failed
        evaluation get ``inf`` without being evaluated.

        Parameters
        ----------
        X : array_like
            Candidates, one flat vector per row.

        Returns
        -------
        numpy.ndarray
            The loss of each row.
        """
        rows = np.atleast_2d(np.asarray(X, dtype=np.float64))
        losses = np.full(len(rows), math.inf)
        slots = max(1, len(self.fitter.contexts))
        for first in range(0, len(rows), slots):
            size = min(slots, len(rows) - first, self._left())
            if size == 0:
                break
            batch = [self.unflatten(row) for row in rows[first : first + size]]
            self.n_evals += size
            try:
                losses[first : first + size] = self.fitter.evaluate(batch)
                self.fitter.step()
            except BaseException as exc:
                self.error = exc
                raise
        return losses

    def unflatten(self, x: Any) -> dict[str, Any]:
        """Return the parameter tree that the flat vector ``x`` stands for.

        Array leaves keep their shape and dtype; scalar leaves come back as
        floats.
        """
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        if flat.size != self.x0.size:
            msg = f"expected {self.x0.size} values, got {flat.size}"
            raise ValueError(msg)
        params: dict[Any, Any] = {}
        for leaf in self._leaves:
            node = params
            for key in leaf.path[:-1]:
                node = node.setdefault(key, {})
            chunk = flat[leaf.start : leaf.stop]
            node[leaf.path[-1]] = (
                float(chunk[0])
                if leaf.dtype is None
                else chunk.reshape(leaf.shape).astype(leaf.dtype)
            )
        return params

    def flatten(self, params: Mapping[str, Any]) -> np.ndarray:
        """Return the flat vector of a parameter tree shaped like the fitter's."""
        parts = []
        for leaf in self._leaves:
            node: Any = params
            for key in leaf.path:
                if not isinstance(node, Mapping) or key not in node:
                    msg = f"params have no parameter {leaf.name!r}"
                    raise ValueError(msg)
                node = node[key]
            value = np.asarray(node, dtype=np.float64)
            if value.shape != leaf.shape:
                msg = (
                    f"parameter {leaf.name!r} has shape {value.shape}, "
                    f"expected {leaf.shape}"
                )
                raise ValueError(msg)
            parts.append(value.ravel())
        return np.concatenate(parts)


def objective(
    fitter: Fitter[Any], bounds: Mapping[str, Any] | None = None
) -> Objective:
    """Return the flat view of ``fitter`` that anneal's drivers search.

    Parameters
    ----------
    fitter : chemfit.Fitter
        Fitter with the session API (``init``, ``evaluate``, ``step``,
        ``finish``). Every leaf of ``fitter.initial_parameters`` is searched:
        a real number, or a floating-point array of any shape, such as
        ``(n, 3)`` positions.
    bounds : mapping, optional
        Bounds in ChemFit's format, a tree mirroring the parameters with a
        ``(lower, upper)`` pair per leaf. A pair of numbers applies to every
        element of an array leaf, and a pair of arrays bounds each element
        (any arrays that broadcast to the leaf's shape). These pairs override
        ``fitter.bounds`` leaf by leaf.

    Returns
    -------
    Objective
        Callable view with ``x0``, ``low``, ``high``, ``unflatten``,
        ``flatten``, ``eval_batch`` and the evaluation counter ``n_evals``.

    Raises
    ------
    ValueError
        If a leaf has no bound, a one-sided or infinite bound, bounds that do
        not broadcast to its shape or whose lower side exceeds the upper, or a
        start outside its bounds, or if ``bounds`` names a key that is not a
        parameter. The message names the key.
    TypeError
        If a leaf is neither a real number nor a floating-point array, or if
        ``fitter`` lacks the session API.

    Examples
    --------
    Run a driver by hand on the view::

        view = anneal.chemfit.objective(fitter, bounds={"positions": (-3.0, 3.0)})
        fitter.init()
        anneal.run(view, view.low, view.high, anneal.Boltzmann(),
                   n_epochs=20, steps_per_epoch=100, x0=view.x0)
        if view.error is not None:
            raise view.error
        best = fitter.finish()
    """
    return Objective(fitter, bounds)


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _check_open_box(view: Objective, method: str) -> None:
    for leaf in view._leaves:
        span = slice(leaf.start, leaf.stop)
        if (view.low[span] == view.high[span]).any():
            msg = (
                f"method {method!r} needs lower < upper, but the bounds of "
                f"{leaf.name!r} are equal; boltzmann, fast, gsa and qmc hold "
                "such a value fixed"
            )
            raise ValueError(msg)


def _driver_options(
    method: str, budget: int, seed: int, options: dict[str, Any]
) -> dict[str, Any]:
    if method in ("portfolio", "dmc"):
        return {"budget": budget, "seed": seed, **options}
    steps = int(options.pop("steps_per_epoch", 100))
    if steps < 1:
        msg = f"steps_per_epoch must be at least 1, got {steps}"
        raise ValueError(msg)
    if method == "qmc":
        preset = options.pop("preset", None)
        n_starts = int(options.pop("n_starts", 8))
        if n_starts < 1:
            msg = f"n_starts must be at least 1, got {n_starts}"
            raise ValueError(msg)
        n_starts = min(n_starts, budget)
        if options:
            msg = f"qmc takes no option {', '.join(sorted(options))}"
            raise TypeError(msg)
        # Every start gets an equal share, so shorten the epochs to fit it.
        share = max(0, budget // n_starts - 1)
        n_epochs = -(-share // steps)
        return {
            "preset": anneal.Boltzmann() if preset is None else preset,
            "n_starts": n_starts,
            "n_epochs": n_epochs,
            "steps_per_epoch": share // n_epochs if n_epochs else steps,
            "seed": seed,
        }
    return {
        "preset": getattr(anneal, _PRESETS[method])(**options),
        "n_epochs": -(-max(0, budget - 1) // steps),
        "steps_per_epoch": steps,
        "seed": seed,
    }


def fit(
    fitter: Fitter[Any],
    budget: int,
    *,
    method: str = "portfolio",
    bounds: Mapping[str, Any] | None = None,
    seed: int = 0,
    batch_size: int = 1,
    **options: Any,
) -> dict[str, Any]:
    """Fit ``fitter`` with an anneal driver and return its best parameters.

    Opens a session with ``fitter.init(batch_size=batch_size)``, evaluates
    ``fitter.initial_parameters`` first, runs the driver from there over the
    flat view from :func:`objective`, and closes the session with
    ``fitter.finish()``. ``fitter.step()`` follows every evaluation or batch,
    every candidate lies inside the bounds, and at most ``budget`` candidates
    reach ``fitter.evaluate``.

    Parameters
    ----------
    fitter : chemfit.Fitter
        Fitter with the session API (``init``, ``evaluate``, ``step``,
        ``finish``), as on ChemFit's ``next`` branch.
    budget : int
        Hard cap on the candidates sent to ``fitter.evaluate``.
    method : str, optional
        Driver, one of :data:`METHODS`:

        - ``"portfolio"``: ``global_optimize``, the budget-allocating
          portfolio (the default).
        - ``"boltzmann"``, ``"fast"``, ``"gsa"``: ``run`` with that preset.
        - ``"qmc"``: ``run_qmc``, ``run`` from low-discrepancy starts.
        - ``"dmc"``: ``dmc_population_optimize``, whose walker batches go
          through :meth:`Objective.eval_batch`.

        The driver gets the start as ``x0`` when the installed version takes
        one (``global_optimize`` does not yet), and its own call at exactly
        ``x0`` reuses the start's loss instead of evaluating it again.
    bounds : mapping, optional
        ChemFit-format bounds that override ``fitter.bounds`` leaf by leaf;
        see :func:`objective`.
    seed : int, optional
        Seed of the driver.
    batch_size : int, optional
        Candidate slots of the session, the most candidates one
        ``fitter.evaluate`` call receives from :meth:`Objective.eval_batch`.
    **options
        Passed on to the driver. ``"portfolio"`` and ``"dmc"`` forward them
        to their driver (for example ``policy``, ``target_n`` or
        ``steps_per_control``). The ``run`` methods take ``steps_per_epoch``
        (default 100), with as many epochs as the budget allows, and the
        preset's parameters (``t_init``, ``sigma``, ``gamma``, ``q_v``,
        ``q_a``). ``"qmc"`` takes ``preset`` (default ``Boltzmann()``),
        ``n_starts`` (default 8, at most ``budget``) and ``steps_per_epoch``,
        shortening the epochs so every start gets an equal share of the
        budget.

    Returns
    -------
    dict
        ``fitter.finish()``: the best evaluated parameters, with array leaves
        in their own shape and dtype.

    Raises
    ------
    ValueError
        For an unknown ``method``, a ``budget``, ``steps_per_epoch`` or
        ``n_starts`` below one, equal lower and upper bounds with
        ``"portfolio"`` or ``"dmc"`` (the ``run`` methods hold such a value
        fixed), and the bound and start errors of :func:`objective`.
    TypeError
        For the leaf and fitter errors of :func:`objective`, an ``x0`` option
        (the start is ``fitter.initial_parameters``), and options the driver
        or preset does not take.

    Notes
    -----
    anneal's drivers turn an exception raised by the objective into ``inf``
    and keep going. ``fit`` instead sends nothing more after the first
    exception from ``fitter.evaluate``, or from a callback run by
    ``fitter.step``, and re-raises it once the driver returns, so ChemFit's
    exception policy (``swallow_exceptions``) applies as in its own fitters.

    Examples
    --------
    Relax ``(13, 3)`` Lennard-Jones positions inside a box of +-3::

        best = anneal.chemfit.fit(fitter, 2000, bounds={"positions": (-3.0, 3.0)})
        best["positions"].shape  # (13, 3)
    """
    if method not in _DRIVERS:
        msg = f"method must be one of {', '.join(METHODS)}, got {method!r}"
        raise ValueError(msg)
    if "x0" in options:
        msg = "fit starts at fitter.initial_parameters and takes no x0"
        raise TypeError(msg)
    budget = int(budget)
    if budget < 1:
        msg = f"budget must be at least 1, got {budget}"
        raise ValueError(msg)
    view = objective(fitter, bounds)
    if method in ("portfolio", "dmc"):
        _check_open_box(view, method)
    view.budget = budget
    driver = getattr(anneal, _DRIVERS[method])
    kwargs = _driver_options(method, budget, seed, dict(options))
    if _accepts(driver, "x0"):
        kwargs["x0"] = view.x0

    fitter.init(batch_size=batch_size)
    # Not every driver evaluates x0 bit for bit (dmc reflects it into the box,
    # which can move it by an ulp), and older ones take no x0 at all, so the
    # start is evaluated here and a driver's own call at x0 reuses its loss.
    view._start_loss = view(view.x0)
    driver(view, view.low, view.high, **kwargs)
    if view.error is not None:
        raise view.error
    return fitter.finish()
