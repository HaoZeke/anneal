"""ChemFit bridges for the gradient-free drivers.

ChemFit's ``Fitter`` runs a session that the caller drives: ``init``, one
evaluation per candidate, one notice per optimizer step, then ``finish``.
Current ChemFit names the middle two ``evaluate`` and ``step``; ChemFit 3.1
named them ``ask`` and ``tell``. Every bridge here drives ``evaluate`` and
``step`` when the fitter has both, ``ask`` and ``tell`` otherwise, and refuses
a fitter with neither pair before calling ``init``. ``finish`` receives the
best evaluated parameters and its return value is the result; ChemFit returns
the parameters it was given.

Nested parameter dicts are flattened only at the optimizer boundary and
rebuilt on the way back. A leaf keeps its type: a Python number comes back as
a float, a NumPy scalar or array keeps its dtype and shape, and the bounds of
a float32 or float16 leaf are rounded inward so the cast candidate stays
inside them. A parameter whose lower and upper bounds are equal is held
fixed. Every evaluation lies inside the box, the first one is the start, and
the budget counts it. The default driver is the Thompson-allocated portfolio.

The first exception the fitter raises, or a loss that is not a real number,
ends the drive: the fitter is not called again, ``finish`` is skipped, and
the exception reaches the caller.
"""

from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "ChemFitVector",
    "chemfit_box",
    "fit_anneal",
    "fit_chemfit",
    "flatten_parameters",
    "run_benchmark",
    "run_fitter",
    "unflatten_parameters",
]

_CLASSICAL_DRIVERS = ("boltzmann", "fast", "gsa")
_FLOAT64_MANTISSA = np.finfo(np.float64).nmant
_MISSING = object()


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number other than a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _first(mask: np.ndarray) -> int | None:
    """Index of the first true entry of ``mask``, or ``None``."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def _real_array(value: Any, what: str) -> np.ndarray:
    """``value`` as a float64 array; anything not real-numeric is an error."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{what} is not real-numeric")
    return arr.astype(np.float64)


# ---------------------------------------------------------------------------
# The fitter's session protocol.
# ---------------------------------------------------------------------------

_PROTOCOLS = (("evaluate", "step"), ("ask", "tell"))


def _protocol(fitter: Any):
    """The fitter's ``(evaluate, step)`` methods, else its ``(ask, tell)``.

    A fitter with neither whole pair, or without ``finish``, is a TypeError.
    """
    for evaluate_name, step_name in _PROTOCOLS:
        evaluate = getattr(fitter, evaluate_name, None)
        step = getattr(fitter, step_name, None)
        if callable(evaluate) and callable(step):
            break
    else:
        raise TypeError(
            "the fitter needs evaluate and step (ChemFit) or ask and tell "
            f"(ChemFit 3.1); {type(fitter).__name__} has neither pair"
        )
    if not callable(getattr(fitter, "finish", None)):
        raise TypeError(f"the fitter needs finish; {type(fitter).__name__} has none")
    return evaluate, step


def _init(fitter: Any) -> None:
    init = getattr(fitter, "init", None)
    if callable(init):
        init()


def _finish(fitter: Any, params: dict[str, Any]) -> Any:
    """``fitter.finish(params)``, or ``fitter.finish()`` when it takes no argument."""
    finish = fitter.finish
    try:
        signature = inspect.signature(finish)
    except (TypeError, ValueError):
        return finish(params)
    try:
        signature.bind(params)
    except TypeError:
        return finish()
    return finish(params)


def _loss_value(value: Any) -> float:
    """One real loss as a float; anything else is a TypeError.

    A one-element list (a batch of one) and a 0-d array are unwrapped first.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if not _is_number(value):
        raise TypeError(
            f"the fitter returned a loss of type {type(value).__name__}; "
            "a loss must be one real number"
        )
    return float(value)


# ---------------------------------------------------------------------------
# Parameter layout: where each leaf of a nested dict sits in the flat vector.
# ---------------------------------------------------------------------------


def _path_str(path: tuple) -> str:
    return ".".join(str(key) for key in path)


def _iter_leaves(params: Mapping, path: tuple = ()):
    """Yield ``(path, value)`` for every non-mapping leaf of a nested mapping."""
    for key, value in params.items():
        if isinstance(value, Mapping):
            yield from _iter_leaves(value, path + (key,))
        else:
            yield path + (key,), value


def _lookup(params: Any, path: tuple) -> Any:
    node = params
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _tree(params: Mapping) -> dict:
    """The nesting of ``params`` with every leaf replaced by ``None``."""
    return {
        key: _tree(value) if isinstance(value, Mapping) else None
        for key, value in params.items()
    }


def _copy_tree(tree: dict) -> dict:
    return {
        key: _copy_tree(value) if isinstance(value, dict) else None
        for key, value in tree.items()
    }


def _leaf_form(name: str, value: Any) -> tuple[str, np.dtype, tuple[int, ...]]:
    """``(kind, dtype, shape)`` of a parameter leaf.

    ``kind`` is ``"number"`` for a Python number, ``"scalar"`` for a NumPy
    scalar and ``"array"`` for anything array-like. ``dtype`` is the float
    type a candidate value is cast to; integer and bool leaves are optimized
    as float64.
    """
    if isinstance(value, np.generic):
        kind, raw, shape = "scalar", value.dtype, ()
    elif isinstance(value, np.ndarray):
        kind, raw, shape = "array", value.dtype, value.shape
    elif isinstance(value, numbers.Real):
        return "number", np.dtype(np.float64), ()
    else:
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name} is not real-numeric") from error
        kind, raw, shape = "array", arr.dtype, arr.shape
    if raw.kind in "biu":
        return kind, np.dtype(np.float64), shape
    if raw.kind != "f":
        raise ValueError(f"parameter {name} is not real-numeric")
    if np.finfo(raw).nmant > _FLOAT64_MANTISSA:
        raise TypeError(
            f"parameter {name} is {raw.name}, which the float64 drivers cannot "
            "carry exactly; convert it to float64 first"
        )
    return kind, raw, shape


class _Leaf:
    """One parameter leaf: its path, its slice of the flat vector, its type."""

    __slots__ = ("path", "name", "kind", "dtype", "shape", "size", "offset")

    def __init__(self, path: tuple, value: Any, offset: int):
        self.path = path
        self.name = _path_str(path)
        self.kind, self.dtype, self.shape = _leaf_form(self.name, value)
        self.size = math.prod(self.shape)
        self.offset = offset

    @property
    def span(self) -> slice:
        return slice(self.offset, self.offset + self.size)

    def rebuild(self, chunk: np.ndarray) -> Any:
        """A fresh leaf holding ``chunk``, in the leaf's own type."""
        if self.kind == "number":
            return float(chunk[0])
        if self.kind == "scalar":
            return self.dtype.type(chunk[0])
        return chunk.astype(self.dtype).reshape(self.shape)


class _Layout:
    """Where each leaf of a nested parameter mapping sits in the flat vector."""

    def __init__(self, template: Mapping):
        self.leaves: list[_Leaf] = []
        offset = 0
        for path, value in _iter_leaves(template):
            leaf = _Leaf(path, value, offset)
            self.leaves.append(leaf)
            offset += leaf.size
        self.size = offset
        self._tree = _tree(template)

    def name(self, i: int) -> str:
        """Name of coordinate ``i``: the dotted key, indexed inside an array."""
        for leaf in self.leaves:
            if leaf.offset <= i < leaf.offset + leaf.size:
                if leaf.shape == ():
                    return leaf.name
                index = np.unravel_index(i - leaf.offset, leaf.shape)
                return f"{leaf.name}[{','.join(str(int(k)) for k in index)}]"
        raise IndexError(i)

    def names(self) -> list[str]:
        return [self.name(i) for i in range(self.size)]

    def vector(
        self, params: Any, what: str, *, exact: bool = True, finite: bool = True
    ) -> np.ndarray:
        """The flat float64 vector of ``params``, a mapping that mirrors the layout.

        Each leaf must have the layout's shape (only its size when ``exact``
        is false) and, when ``finite``, finite values. Errors name the leaf.
        """
        out = np.empty(self.size)
        for leaf in self.leaves:
            value = _lookup(params, leaf.path)
            if value is _MISSING:
                raise ValueError(f"{what} has no {leaf.name}")
            arr = _real_array(value, f"{what} {leaf.name}")
            if arr.shape != leaf.shape if exact else arr.size != leaf.size:
                raise ValueError(
                    f"{what} {leaf.name} has shape {arr.shape}, but the parameter "
                    f"has shape {leaf.shape}"
                )
            values = arr.reshape(-1)
            if finite and not np.all(np.isfinite(values)):
                raise ValueError(f"{what} {leaf.name} must be finite")
            out[leaf.span] = values
        return out

    def cast(self, vector: np.ndarray) -> np.ndarray:
        """``vector`` with each coordinate rounded to its leaf's dtype."""
        out = np.array(vector, dtype=np.float64)
        for leaf in self.leaves:
            if leaf.dtype != np.float64:
                out[leaf.span] = out[leaf.span].astype(leaf.dtype)
        return out

    def params(self, vector: np.ndarray) -> dict[str, Any]:
        """A fresh nested dict holding ``vector``, each leaf in its own type."""
        out = _copy_tree(self._tree)
        for leaf in self.leaves:
            node = out
            for key in leaf.path[:-1]:
                node = node[key]
            node[leaf.path[-1]] = leaf.rebuild(vector[leaf.span])
        return out


# ---------------------------------------------------------------------------
# Bounds.
# ---------------------------------------------------------------------------


def _bound_side(leaf: _Leaf, value: Any, side: str) -> np.ndarray | None:
    """One side of a leaf's bounds, flat with the leaf's size; ``None`` if unset.

    A side may be a scalar, an array of the leaf's size, or an array that
    broadcasts to the leaf's shape.
    """
    if value is None:
        return None
    arr = _real_array(value, f"the {side} bound of {leaf.name}")
    if arr.size == leaf.size:
        return arr.reshape(-1)
    try:
        return np.broadcast_to(arr, leaf.shape).reshape(-1)
    except ValueError:
        raise ValueError(
            f"the {side} bound of {leaf.name} has shape {arr.shape}, which does "
            f"not fit the parameter's shape {leaf.shape}"
        ) from None


def _bound_pair(leaf: _Leaf, entry: Any):
    """``(lower, upper)`` from one bounds entry; ``None`` marks a missing side."""
    if entry is None or entry is _MISSING:
        return None, None
    if isinstance(entry, np.ndarray) and entry.ndim >= 1 and entry.shape[0] == 2:
        lower, upper = entry[0], entry[1]
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        lower, upper = entry
    else:
        raise ValueError(
            f"the bounds of {leaf.name} must be a (lower, upper) pair, got {entry!r}"
        )
    return _bound_side(leaf, lower, "lower"), _bound_side(leaf, upper, "upper")


def _covers(layout: _Layout, bounds: Any) -> bool:
    """Whether a bounds mapping has an entry for every leaf."""
    if not isinstance(bounds, Mapping):
        return False
    for leaf in layout.leaves:
        entry = _lookup(bounds, leaf.path)
        if entry is _MISSING or entry is None:
            return False
    return True


def _mapped_box(layout: _Layout, bounds: Any, start, span):
    """The raw box from a bounds mapping that mirrors the parameters.

    A missing entry or side falls back to ``start -/+ span``; when ``span``
    is ``None`` every side must be given.
    """
    if bounds is not None and not isinstance(bounds, Mapping):
        raise TypeError(
            "bounds must be a mapping that mirrors the parameters, got "
            f"{type(bounds).__name__}"
        )
    if span is None:
        low = np.full(layout.size, np.nan)
        high = np.full(layout.size, np.nan)
    else:
        with np.errstate(over="ignore"):
            low, high = start - span, start + span
    for leaf in layout.leaves:
        entry = _MISSING if bounds is None else _lookup(bounds, leaf.path)
        lower, upper = _bound_pair(leaf, entry)
        for side, values, target in (("lower", lower, low), ("upper", upper, high)):
            if values is not None:
                target[leaf.span] = values
            elif span is None:
                raise ValueError(
                    f"{leaf.name} has no {side} bound: pass low and high, or bounds "
                    "that mirror initial_params; the chain will not invent a box"
                )
    return low, high


def _inward(low: np.ndarray, high: np.ndarray, dtype: np.dtype):
    """``low`` rounded up and ``high`` rounded down to values ``dtype`` holds."""
    info = np.finfo(dtype)
    up = np.clip(low, info.min, info.max).astype(dtype)
    up = np.where(up < low, np.nextafter(up, dtype.type(np.inf)), up)
    down = np.clip(high, info.min, info.max).astype(dtype)
    down = np.where(down > high, np.nextafter(down, dtype.type(-np.inf)), down)
    return up.astype(np.float64), down.astype(np.float64)


def _settle(layout: _Layout, low, high):
    """Check a raw box coordinate by coordinate and round it to each leaf's dtype.

    Errors name the coordinate. Coordinates whose lower and upper bounds are
    equal stay in the result; the bridges hold them fixed.
    """
    low = np.array(low, dtype=np.float64)
    high = np.array(high, dtype=np.float64)

    def refuse(i: int, problem: str):
        raise ValueError(
            f"the bounds of {layout.name(i)} {problem}: "
            f"lower={float(low[i])!r}, upper={float(high[i])!r}"
        )

    bad = _first(~(np.isfinite(low) & np.isfinite(high)))
    if bad is not None:
        refuse(bad, "must be finite")
    bad = _first(low > high)
    if bad is not None:
        refuse(bad, "are empty, the lower above the upper")
    with np.errstate(over="ignore"):
        bad = _first(~np.isfinite(2.0 * (high - low)))
    if bad is not None:
        refuse(bad, "are too wide for a float")
    total = 0.0
    for width in (high - low).tolist():
        total += width
    if not math.isfinite(total):
        raise ValueError("the box is too wide: the sum of its widths is not finite")
    for leaf in layout.leaves:
        if leaf.dtype == np.float64:
            continue
        up, down = _inward(low[leaf.span], high[leaf.span], leaf.dtype)
        bad = _first(up > down)
        if bad is not None:
            refuse(leaf.offset + bad, f"hold no {leaf.dtype.name} value")
        low[leaf.span], high[leaf.span] = up, down
    return low, high


def _clip(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """``x`` moved onto ``[low, high]``; coordinates inside keep their bits."""
    return np.where(x < low, low, np.where(x > high, high, x))


# ---------------------------------------------------------------------------
# One drive of a fitter.
# ---------------------------------------------------------------------------


class _Problem:
    """A flattened fit: the layout, the box, the start, the free coordinates."""

    def __init__(self, layout: _Layout, low: np.ndarray, high: np.ndarray, start):
        self.layout = layout
        self.low = low
        self.high = high
        self.free = low < high
        self.start = layout.cast(_clip(np.asarray(start, dtype=np.float64), low, high))

    def candidate(self, x: np.ndarray) -> dict[str, Any]:
        """The parameters at the driver's point ``x`` over the free coordinates."""
        full = self.start.copy()
        full[self.free] = _clip(x, self.low[self.free], self.high[self.free])
        return self.layout.params(full)


class _Stop(BaseException):
    """Ends a driver run from inside the objective; the bridge catches it."""


class _Session:
    """The objective a driver calls: one fitter evaluation per call.

    It counts evaluations against the budget, sends a step notice every
    ``step_every`` of them, keeps the best point, and records the first
    exception from the fitter or from reading its loss. That exception, or
    a spent budget, stops the driver.
    """

    def __init__(self, problem: _Problem, evaluate, step, budget: int, step_every: int):
        self.problem = problem
        self._evaluate = evaluate
        self._step = step
        self.budget = budget
        self.step_every = step_every
        self.count = 0
        self.best: np.ndarray | None = None
        self.best_loss = math.inf
        self.error: BaseException | None = None

    def __call__(self, x) -> float:
        if self.error is not None or self.count >= self.budget:
            raise _Stop
        try:
            point = np.array(x, dtype=np.float64)
            loss = _loss_value(self._evaluate(self.problem.candidate(point)))
            self.count += 1
            rank = math.inf if math.isnan(loss) else loss
            if self.best is None or rank < self.best_loss:
                self.best, self.best_loss = point, rank
            if self.count % self.step_every == 0:
                self._step()
        except BaseException as error:
            self.error = error
            raise _Stop from None
        return loss


def _drive(
    session: _Session,
    driver: str,
    seed: int,
    preset: Any = None,
    steps_per_epoch: int = 1,
) -> dict[str, Any]:
    """Run ``driver`` over the free coordinates and return the best parameters.

    The first fitter exception is raised here, after the driver has returned.
    """
    from anneal import global_optimize, run

    problem, budget = session.problem, session.budget
    free = problem.free
    low, high, start = problem.low[free], problem.high[free], problem.start[free]
    try:
        if not free.any():
            session(start)
        elif driver == "portfolio":
            global_optimize(session, low, high, budget, seed=seed, x0=start)
        else:
            run(
                session,
                low,
                high,
                preset,
                n_epochs=max(1, budget // steps_per_epoch),
                steps_per_epoch=steps_per_epoch,
                seed=seed,
                x0=start,
                max_evals=budget,
            )
    except _Stop:
        pass
    if session.error is not None:
        raise session.error
    if session.best is None:
        raise RuntimeError("the driver ended before evaluating the start")
    return problem.candidate(session.best)


def _classical_preset(driver: str, preset_kwargs: dict[str, Any] | None):
    from anneal import Boltzmann, Fast, Gsa

    kwargs = dict(preset_kwargs or {})
    if driver == "boltzmann":
        return Boltzmann(**kwargs)
    if driver == "fast":
        return Fast(**kwargs)
    return Gsa(**kwargs)


# ---------------------------------------------------------------------------
# fit_anneal and its flatten helpers.
# ---------------------------------------------------------------------------


def _deep_copy(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _deep_copy(value) for key, value in node.items()}
    if isinstance(node, np.ndarray):
        return node.copy()
    return node


def _assign_path(params: dict[str, Any], path: tuple, value: Any) -> None:
    for key in path[:-1]:
        params = params[key]
    params[path[-1]] = value


def flatten_parameters(params: dict[str, Any]):
    """Flatten a nested parameter dict to ``(vector, spec)``.

    Scalar leaves contribute one entry; array leaves contribute their
    ravelled entries in C order. ``spec`` records ``(path, shape)`` per
    leaf (``shape == ()`` for scalars) so :func:`unflatten_parameters`
    can rebuild the structure. Only real-numeric leaves are supported;
    non-finite starts are rejected, and so are ``longdouble`` leaves, which a
    float64 vector cannot carry exactly.
    """
    if not isinstance(params, Mapping) or not params:
        raise ValueError("params must be a non-empty dict")
    layout = _Layout(params)
    for leaf in layout.leaves:
        if leaf.size == 0:
            raise ValueError(f"parameter {leaf.name} is empty")
    vector = layout.vector(params, "parameter")
    return vector, [(leaf.path, leaf.shape) for leaf in layout.leaves]


def spec_total(spec) -> int:
    """Flattened dimension of a :func:`flatten_parameters` spec."""
    return sum(int(np.prod(shape)) if shape != () else 1 for _, shape in spec)


def unflatten_parameters(vector: np.ndarray, spec, template: dict[str, Any]):
    """Rebuild a nested parameter dict from a flat vector and a spec.

    Array leaves are reshaped to their original shape, and each leaf takes
    the type of the matching ``template`` leaf: a Python number comes back as
    a float, a NumPy scalar or array keeps its dtype. The returned dict
    mirrors ``template``'s nesting and never mutates the template.
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    total = spec_total(spec)
    if vector.size != total:
        raise ValueError(f"vector has length {vector.size} but the spec needs {total}")
    out = _deep_copy(template)
    offset = 0
    for path, shape in spec:
        size = int(np.prod(shape)) if shape != () else 1
        chunk = vector[offset : offset + size]
        try:
            leaf = _Leaf(path, _lookup(template, path), 0)
        except (TypeError, ValueError):
            leaf = None
        if leaf is not None and leaf.shape == tuple(shape):
            value = leaf.rebuild(chunk)
        else:
            value = chunk[0] if shape == () else chunk.reshape(shape)
        _assign_path(out, path, value)
        offset += size
    return out


def fit_anneal(
    fitter: Any,
    budget: int,
    *,
    driver: str = "portfolio",
    seed: int = 0,
    x0: dict[str, Any] | np.ndarray | None = None,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    bound_span: float = 3.0,
    steps_per_epoch: int = 100,
    preset_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a ChemFit ``Fitter`` with an anneal gradient-free optimizer.

    Args:
      fitter: ChemFit ``Fitter``, duck-typed: ``initial_parameters``,
        ``bounds``, ``init``, ``finish``, and either ``evaluate`` and
        ``step`` (current ChemFit) or ``ask`` and ``tell`` (ChemFit 3.1).
        Each evaluation is followed by one ``step`` (or ``tell``).
      budget: total objective-evaluation budget, the start included; it is
        never exceeded.
      driver: ``"portfolio"`` (default; Thompson-allocated SOTA driver
        over the gradient-free arms — QMC restarts, basin hopping,
        differential evolution, GSA, parallel tempering — with no
        gradient), or one of ``"boltzmann"``, ``"fast"``, ``"gsa"`` for
        single-chain ablation runs.
      seed: RNG seed.
      x0: warm start. ``None`` (default) uses the fitter's
        ``initial_parameters``; a nested dict with the same structure or
        a flat vector of the flattened dimension overrides it. A start
        outside the box is moved onto it.
      low, high: explicit flat bound vectors. When omitted, bounds come
        from the fitter's ``bounds`` dict (``(lower, upper)`` pairs
        mirroring ``initial_params``; each side a scalar, an array of the
        leaf's shape, or ``None``); entries without bounds fall back to
        ``x0 +/- bound_span``. A coordinate whose two bounds are equal is
        held fixed.
      bound_span: half-width of the fallback box around unbounded entries.
      steps_per_epoch: classical-driver epoch width; the chain runs
        ``max(1, budget // steps_per_epoch)`` epochs and stops when the
        budget is spent.
      preset_kwargs: extra kwargs for the preset constructor
        (e.g. ``{"t_init": 5.0}``); classical drivers only.

    Returns what ``fitter.finish`` returns for the best evaluated parameters
    (ChemFit returns them as given); each leaf keeps its type, dtype and
    shape. The first exception raised by the fitter, or a loss that is not a
    real number, stops the fit and is raised without calling ``finish``.
    """
    evaluate, step = _protocol(fitter)
    driver = str(driver).lower()
    if driver not in ("portfolio", *_CLASSICAL_DRIVERS):
        raise ValueError(
            f"driver must be 'portfolio' or one of {', '.join(_CLASSICAL_DRIVERS)}; "
            f"got {driver!r}"
        )
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")

    initial_parameters = getattr(fitter, "initial_parameters", None)
    if not isinstance(initial_parameters, dict) or not initial_parameters:
        raise ValueError("fitter.initial_parameters must be a non-empty dict")

    layout = _Layout(initial_parameters)
    start_vector, spec = flatten_parameters(initial_parameters)
    if x0 is not None:
        if isinstance(x0, dict):
            start_vector, x0_spec = flatten_parameters(x0)
            if [path for path, _ in x0_spec] != [path for path, _ in spec]:
                raise ValueError("x0 dict must mirror fitter.initial_parameters")
        else:
            start_vector = np.asarray(x0, dtype=np.float64).ravel()
            if start_vector.size != spec_total(spec):
                raise ValueError(
                    f"x0 has length {start_vector.size} but the parameters "
                    f"flatten to {spec_total(spec)}"
                )
            if not np.all(np.isfinite(start_vector)):
                raise ValueError("x0 must contain only finite values")

    if (low is None) != (high is None):
        raise ValueError("low and high must be given together")
    if low is None:
        span = float(bound_span)
        if not (math.isfinite(span) and span > 0.0):
            raise ValueError("bound_span must be positive and finite")
        box = _mapped_box(layout, getattr(fitter, "bounds", None), start_vector, span)
    else:
        box = (_real_array(low, "low").ravel(), _real_array(high, "high").ravel())
        if box[0].size != layout.size or box[1].size != layout.size:
            raise ValueError(
                f"low/high have lengths {box[0].size}/{box[1].size} "
                f"but the flattened parameters have dimension {layout.size}"
            )
    # run and global_optimize refuse a start outside the box. A caller vector
    # such as zeros is pulled onto the box before the first evaluation.
    problem = _Problem(layout, *_settle(layout, *box), start_vector)

    # The fitter owns bookkeeping; every evaluation is one optimizer step.
    _init(fitter)
    preset = None if driver == "portfolio" else _classical_preset(driver, preset_kwargs)
    steps = max(1, min(int(steps_per_epoch), budget))
    session = _Session(problem, evaluate, step, budget, step_every=1)
    return _finish(fitter, _drive(session, driver, int(seed), preset, steps))


# ---------------------------------------------------------------------------
# fit_chemfit and its vector view.
# ---------------------------------------------------------------------------


class ChemFitVector:
    """Flattened view of ChemFit (possibly nested, array-valued) parameters.

    Scalar leaves become one coordinate each; array leaves expand element-wise
    in C order under the same dotted key. The vector layout is fixed at
    construction, so :meth:`pack` / :meth:`unpack` round-trip between anneal's
    flat box and ChemFit's nested parameter dicts; :meth:`unpack` gives each
    leaf the template's type, dtype and shape.
    """

    def __init__(self, template: dict[str, Any]):
        self._layout = _Layout(template)
        self.keys: list[str] = self._layout.names()
        self.shapes: dict[str, tuple[int, ...]] = {
            leaf.name: leaf.shape for leaf in self._layout.leaves
        }
        self.x0 = self._layout.vector(template, "parameter")

    @property
    def dim(self) -> int:
        return self._layout.size

    def _base_key(self, key: str) -> str:
        return key.split("[", 1)[0]

    def pack(self, params: dict[str, Any]) -> np.ndarray:
        """Flatten a nested parameter dict into the fixed vector layout."""
        return self._layout.vector(params, "params", exact=False, finite=False)

    def unpack(self, vector: np.ndarray) -> dict[str, Any]:
        """Rebuild the nested parameter dict from a flat vector."""
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"vector has length {vector.size} but the layout needs {self.dim}")
        return self._layout.params(vector)


def chemfit_box(
    fitter: Any,
    vector: ChemFitVector,
    default_span: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite ``(low, high)`` box vectors for a ChemFit fitter.

    Bounds mirror ``fitter.bounds`` (same structure as the initial
    parameters); each side is a scalar, an array of the leaf's shape, or
    ``None``, and a NumPy array of two rows is a pair too. A parameter
    without bounds gets ``init +/- default_span``. The bounds of a float32 or
    float16 leaf are rounded inward to that dtype. A bound that is not
    finite, is inverted, overflows, or has no width is a ValueError naming
    the parameter: the drivers need ``lower < upper``, and
    :func:`fit_chemfit` holds a zero-width parameter fixed instead.
    """
    layout = vector._layout
    span = float(default_span)
    low, high = _settle(
        layout, *_mapped_box(layout, getattr(fitter, "bounds", None), vector.x0, span)
    )
    fixed = _first(low == high)
    if fixed is not None:
        raise ValueError(
            f"ChemFit bound for {layout.name(fixed)!r} is empty: "
            f"lower={float(low[fixed])} upper={float(high[fixed])}; "
            "a driver box needs lower < upper"
        )
    return low, high


def fit_chemfit(
    fitter: Any,
    budget: int,
    method: str = "portfolio",
    seed: int = 0,
    default_span: float = 3.0,
    tell_every: int = 50,
    steps_per_epoch: int = 100,
    **preset_kwargs: Any,
) -> dict[str, Any]:
    """Fit ChemFit parameters with a gradient-free anneal driver.

    Args:
        fitter: a :class:`chemfit.Fitter` with ``initial_parameters`` and
            optional ``bounds``. Only the session protocol is used, so
            gradient-free drivers never need forces: ``init``, ``finish``,
            and ``evaluate`` / ``step`` (current ChemFit) or ``ask`` /
            ``tell`` (ChemFit 3.1).
        budget: total objective evaluations, the start included; it is
            never exceeded.
        method: ``"portfolio"`` (default; Thompson-allocated SOTA including
            parallel-tempering communicating chains), or ``"boltzmann"``,
            ``"fast"``, ``"gsa"`` for the bound-respecting classical chain.
            Every method starts from the initial parameters, moved onto the
            box when they lie outside it.
        seed: RNG seed forwarded to the anneal driver.
        default_span: half-width around the initial value for parameters
            ChemFit leaves unbounded. Bounds read as in :func:`chemfit_box`,
            except that a parameter with equal bounds is held fixed.
        tell_every: portfolio evaluations between step notices (``step`` or
            ``tell``) so registered callbacks still fire.
        steps_per_epoch: classical-chain evaluations per epoch, and per
            step notice; the chain runs ``max(1, budget // steps_per_epoch)``
            epochs and stops when the budget is spent.
        **preset_kwargs: ``t_init`` / ``sigma`` / ``gamma`` / ``q_v`` /
            ``q_a`` forwarded to the classical preset constructors.

    Returns:
        What ``fitter.finish(best_params)`` returns; ChemFit returns
        ``best_params``, whose leaves keep their types. The first exception
        raised by the fitter, or a loss that is not a real number, stops the
        fit and is raised without calling ``finish``.
    """
    from anneal import Boltzmann, Fast, Gsa

    evaluate, step = _protocol(fitter)
    budget = int(budget)
    if budget < 1:
        raise ValueError("budget must be positive")
    vector = ChemFitVector(dict(fitter.initial_parameters))
    if vector.dim == 0:
        raise ValueError("fitter.initial_parameters holds no parameters")
    layout = vector._layout
    span = float(default_span)
    box = _mapped_box(layout, getattr(fitter, "bounds", None), vector.x0, span)
    problem = _Problem(layout, *_settle(layout, *box), vector.x0)
    steps = max(1, min(int(steps_per_epoch), budget))

    _init(fitter)
    if method == "portfolio":
        preset, every = None, max(1, int(tell_every))
    elif method in ("boltzmann", "fast", "gsa"):
        presets = {
            "boltzmann": Boltzmann(
                t_init=float(preset_kwargs.get("t_init", 5.0)),
                sigma=float(preset_kwargs.get("sigma", 0.5)),
            ),
            "fast": Fast(
                t_init=float(preset_kwargs.get("t_init", 3.0)),
                gamma=float(preset_kwargs.get("gamma", 0.5)),
            ),
            "gsa": Gsa(
                t_init=float(preset_kwargs.get("t_init", 3.0)),
                q_v=float(preset_kwargs.get("q_v", 2.62)),
                q_a=float(preset_kwargs.get("q_a", 1.7)),
            ),
        }
        preset, every = presets[method], steps
    else:
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )
    session = _Session(problem, evaluate, step, budget, step_every=every)
    return _finish(fitter, _drive(session, method, int(seed), preset, steps))


# ---------------------------------------------------------------------------
# run_benchmark, run_fitter and their helpers.
# ---------------------------------------------------------------------------


def flatten_params(params: dict[str, Any]) -> tuple[np.ndarray, list[tuple[tuple[str, ...], tuple[int, ...]]]]:
    """Flatten a nested parameter mapping in key order.

    Returns the flat vector and a spec that rebuilds the same nesting and
    shapes. A scalar leaf comes back as a Python float.
    """
    spec: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    chunks: list[np.ndarray] = []

    def walk(path: tuple[str, ...], value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(path + (str(key),), child)
            return
        arr = np.asarray(value, dtype=np.float64)
        spec.append((path, tuple(arr.shape)))
        chunks.append(arr.reshape(-1))

    walk((), params)
    if not chunks:
        raise ValueError("initial_params is empty")
    return np.concatenate(chunks), spec


def unflatten_params(
    vector: np.ndarray,
    spec: list[tuple[tuple[str, ...], tuple[int, ...]]],
) -> dict[str, Any]:
    """Rebuild the parameter mapping flattened by ``flatten_params``."""
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {}
    offset = 0
    for path, shape in spec:
        size = 1
        for dim in shape:
            size *= int(dim)
        chunk = vector[offset : offset + size].reshape(shape)
        offset += size
        leaf: Any = float(chunk) if shape == () else chunk.copy()
        cursor = out
        for key in path[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        if not path:
            raise ValueError("a parameter leaf needs a name")
        cursor[path[-1]] = leaf
    if offset != vector.size:
        raise ValueError("flat vector does not match the parameter spec")
    return out


def _flat_bound(value: Any, size: int) -> np.ndarray:
    """An explicit bound vector; a single value is broadcast."""
    arr = _real_array(value, "bound").reshape(-1)
    if arr.size == 1 and size != 1:
        return np.full(size, arr[0])
    if arr.size != size:
        raise ValueError(f"bound length {arr.size} does not match parameter length {size}")
    return arr


def _benchmark_box(layout: _Layout, low, high, context_bounds, fitter_bounds):
    """The checked box of :func:`run_benchmark` and :func:`resolve_bounds`."""
    if low is not None or high is not None:
        if low is None or high is None:
            raise ValueError("low and high must be passed together")
        return _settle(layout, _flat_bound(low, layout.size), _flat_bound(high, layout.size))
    if (
        isinstance(context_bounds, Mapping)
        and "low" in context_bounds
        and "high" in context_bounds
    ):
        return _settle(
            layout,
            _flat_bound(context_bounds["low"], layout.size),
            _flat_bound(context_bounds["high"], layout.size),
        )
    for bounds in (context_bounds, fitter_bounds):
        if _covers(layout, bounds):
            return _settle(layout, *_mapped_box(layout, bounds, None, None))
    raise ValueError(
        "pass low and high, or bounds that mirror initial_params; "
        "the chain will not invent a box"
    )


def bounds_from_fitter(
    initial: dict[str, Any], fitter_bounds: Any, size: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a ChemFit bounds mapping that mirrors ``initial_params``.

    ``None`` when the mapping does not bound every leaf or the parameters do
    not flatten to ``size``.
    """
    layout = _Layout(initial)
    if layout.size != size or not _covers(layout, fitter_bounds):
        return None
    return _settle(layout, *_mapped_box(layout, fitter_bounds, None, None))


def resolve_bounds(
    initial: dict[str, Any],
    *,
    low: Any = None,
    high: Any = None,
    context_bounds: Any = None,
    fitter_bounds: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Box for a flattened parameter vector.

    Explicit ``low`` and ``high`` win. A length-1 bound is broadcast.
    Otherwise the context or the fitter supplies a mapping that mirrors
    ``initial``; each side of an entry is a scalar or an array of the leaf's
    shape, and a NumPy array of two rows is a pair too. Bounds are checked
    coordinate by coordinate, and errors name the parameter.
    """
    layout = _Layout(initial)
    if layout.size == 0:
        raise ValueError("initial_params is empty")
    return _benchmark_box(layout, low, high, context_bounds, fitter_bounds)


def run_benchmark(
    benchmark_context: dict[str, Any],
    *,
    method: str = "portfolio",
    seed: int = 0,
    steps_per_epoch: int = 100,
    low: Any = None,
    high: Any = None,
    preset: Any = None,
) -> Any:
    """Minimize a ChemFit fitter with a gradient-free anneal driver.

    ``method`` is ``portfolio`` (the budget-only global optimizer), or
    ``boltzmann``, ``fast``, or ``gsa``. The chain starts at
    ``initial_params``, moved onto the box when it lies outside. Every
    coordinate the fitter sees lies in the box, and ``budget`` caps the
    evaluations, the start included. The fitter is driven through
    ``evaluate`` / ``step`` or ``ask`` / ``tell``, with one step notice per
    evaluation, and each parameter leaf keeps its type.

    ``low`` and ``high`` may be vectors or scalars (broadcast). When they
    are omitted, bounds are read from ``benchmark_context["bounds"]`` or
    ``fitter.bounds``. A coordinate whose two bounds are equal is held fixed.
    The first exception raised by the fitter is raised without calling
    ``finish``.
    """
    from anneal import Boltzmann, Fast, Gsa

    fitter = benchmark_context["fitter"]
    evaluate, step = _protocol(fitter)
    budget = int(benchmark_context["budget"])
    if budget < 1:
        raise ValueError("budget must be positive")
    initial = benchmark_context["initial_params"]
    if not isinstance(initial, dict):
        raise TypeError("initial_params must be a mapping")
    layout = _Layout(initial)
    if layout.size == 0:
        raise ValueError("initial_params is empty")
    start = layout.vector(initial, "parameter", finite=False)
    box = _benchmark_box(
        layout,
        low,
        high,
        benchmark_context.get("bounds"),
        getattr(fitter, "bounds", None),
    )
    problem = _Problem(layout, *box, start)

    _init(fitter)
    name = method.lower()
    if name != "portfolio" and preset is None:
        preset = {"boltzmann": Boltzmann(), "fast": Fast(), "gsa": Gsa()}[name]
    steps = max(1, min(int(steps_per_epoch), budget))
    session = _Session(problem, evaluate, step, budget, step_every=1)
    return _finish(fitter, _drive(session, name, int(seed), preset, steps))

def run_fitter(
    fitter: Any,
    budget: int,
    method: str = "global_optimize",
    preset: Any = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a ChemFit fitter with the gradient-free bridges.

    A fitter that already implements ``fit_anneal`` is called as it stands.
    Otherwise ``method="global_optimize"`` uses the portfolio and
    ``method="sa"`` uses the Boltzmann preset, both through :func:`fit_anneal`.
    """
    if hasattr(fitter, "fit_anneal"):
        return fitter.fit_anneal(
            budget=budget, method=method, preset=preset, seed=seed, **kwargs
        )
    driver = {"global_optimize": "portfolio", "sa": "boltzmann"}.get(method, method)
    return fit_anneal(fitter, int(budget), driver=driver, seed=int(seed))
