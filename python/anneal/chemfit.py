"""ChemFit bridges for the gradient-free drivers.

ChemFit's ``Fitter`` runs a session that the caller drives: ``init``, one
evaluation per candidate, one notice per optimizer step, then ``finish``.
Current ChemFit names the middle two ``evaluate`` and ``step``; ChemFit 3.1
named them ``ask`` and ``tell``. Every bridge here drives ``evaluate`` and
``step`` when the fitter has both, ``ask`` and ``tell`` otherwise, and refuses
a fitter with neither pair before calling ``init``; only :func:`run_benchmark`
still drives ``evaluate`` or ``ask`` without its partner, as anneal 0.10.0
did, with a FutureWarning, and sends the step notices to the other pair's
``tell`` or ``step`` when the fitter has one. ``finish`` receives the best
evaluated parameters and its return value is the result; ChemFit returns the
parameters it was given.

Nested parameter dicts are flattened only at the optimizer boundary and
rebuilt on the way back. A leaf keeps its type: a Python number, a
``Decimal`` included, comes back as a float, a NumPy scalar or array keeps its
shape and its floating dtype (integer, bool and object leaves are optimized,
and returned, as float64), a list or tuple comes back as a list or tuple
nested the same way, and the bounds of a float32 or float16 leaf are rounded
inward so the cast candidate stays inside them. A parameter whose lower and
upper bounds are equal is held fixed. Every evaluation lies inside the box,
the first one is the start, and the budget counts it. The default driver is
the Thompson-allocated portfolio.

Every argument is checked before ``fitter.init()``. The first exception the
fitter raises, or a loss that is not a real number, ends the drive: the
fitter is not called again, ``finish`` is skipped, and the exception reaches
the caller. The few arguments anneal 0.10.0 accepted and ignored, such as
preset keywords under the portfolio, still run with a FutureWarning that
says what to pass instead; they will raise in a future release. So does a
numeric string given for a parameter value, a bound, ``bound_span``,
``default_span`` or a preset keyword a :func:`fit_chemfit` method takes, such
as a bounds pair PyYAML reads from ``[1e-3, 1e1]``: it is read as a number,
with one FutureWarning per call. A string ``budget``, ``seed``,
``steps_per_epoch`` or ``tell_every`` is a TypeError.
"""

from __future__ import annotations

import contextvars
import inspect
import math
import numbers
import sys
import warnings
from collections.abc import Mapping
from decimal import Decimal
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
_DRIVERS = ("portfolio", *_CLASSICAL_DRIVERS)
_SEED_LIMIT = 2**64 - 1
_FLOAT64_MANTISSA = np.finfo(np.float64).nmant
_MISSING = object()


# ---------------------------------------------------------------------------
# Argument checks. All of them run before the fitter's session starts.
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number other than a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


_NOT_ARRAYS = (numbers.Number, str, bytes, bytearray, np.generic, list, tuple, Mapping)


def _one_value(value: Any) -> Any:
    """``value``, or the element of a one-element array of any shape.

    The array may be NumPy's or another library's that NumPy can read; a
    larger one comes back as a NumPy array.
    """
    if value is None or isinstance(value, _NOT_ARRAYS):
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return value
    if array.size == 1:
        return array.reshape(())[()]
    return value if array.dtype.kind == "O" else array


def _dtype_kind(value: Any) -> str:
    """The NumPy kind of ``value``'s dtype, or ``""`` when NumPy cannot name it."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return ""
    try:
        return np.dtype(dtype).kind
    except TypeError:
        return ""


def _as_number(value: Any) -> Any:
    """The real number ``value`` is or holds, else ``None``.

    A real number other than a bool counts, a ``Decimal`` included unless it
    is a signalling NaN, and so does the element of a one-element array of
    any shape, NumPy's or another library's. A string does not. An object of
    another type, such as an array on a GPU that NumPy cannot read, is read
    with ``float()``, as anneal 0.10.0 read it, unless NumPy names its dtype
    bool or complex.
    """
    value = _one_value(value)
    if _is_number(value) or (isinstance(value, Decimal) and not value.is_snan()):
        return value
    if value is None or isinstance(value, (*_NOT_ARRAYS, np.ndarray)):
        return None
    if _dtype_kind(value) in ("b", "c"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return None


def _float_setting(name: str, value: Any) -> float | None:
    """The float setting ``name`` as a float, else ``None``.

    ``value`` is read as :func:`_as_number` reads a number, or, when it is or
    holds a string, such as one PyYAML reads from ``5e-1``, with ``float()``,
    as anneal 0.10.0 read it; that string is reported to the
    :class:`_Strings` the call runs in.
    """
    item = _one_value(value)
    if isinstance(item, (str, bytes, bytearray)):
        try:
            number = float(item)
        except ValueError:
            return None
        _Strings.note(name, 1)
        return number
    number = _as_number(item)
    return None if number is None else float(number)


def _whole(name: str, value: Any, minimum: int, maximum: int = sys.maxsize) -> int:
    """``value`` as an int in ``[minimum, maximum]``, or an error naming it.

    ``value`` is read as :func:`_as_number` reads a number.
    """
    number = _as_number(value)
    if number is None:
        raise TypeError(f"{name} must be a whole number, got {value!r}")
    try:
        count = int(number)
    except (OverflowError, ValueError):
        count = None
    if count is None or count != number:
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if count < minimum:
        least = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {least}, got {value!r}")
    if count > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value!r}")
    return count


def _positive(name: str, value: Any) -> float:
    """``value`` as a positive finite float, or an error naming it.

    ``value`` is read as :func:`_float_setting` reads a float setting.
    """
    number = _float_setting(name, value)
    if number is None:
        raise TypeError(f"{name} must be a number, got {value!r}")
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return number


def _choice(value: Any, choices) -> str | None:
    """``value`` lower-cased when it names one of ``choices``, else ``None``."""
    if isinstance(value, str) and value.lower() in choices:
        return value.lower()
    return None


def _first(mask: np.ndarray) -> int | None:
    """Index of the first true entry of ``mask``, or ``None``."""
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def _real_array(value: Any, what: str) -> np.ndarray:
    """``value`` as a float64 array; anything not real-numeric is an error.

    Numeric strings are read as numbers and reported to the
    :class:`_Strings` the call runs in. What NumPy holds only as objects,
    such as a ``Decimal`` or an object array, is read item by item.
    """
    arr = np.asarray(value)
    if arr.dtype.kind == "O":
        return _object_items(arr, what)
    if arr.dtype.kind in "SU":
        try:
            arr = arr.astype(np.float64)
        except ValueError:
            raise ValueError(f"{what} is not real-numeric") from None
        _Strings.note(what, arr.size)
    if arr.dtype.kind not in "biuf":
        raise ValueError(f"{what} is not real-numeric")
    return arr.astype(np.float64)


def _object_items(arr: np.ndarray, what: str) -> np.ndarray:
    """An object array as float64, each item read with ``float()``.

    An item that is a one-element array is read as its element, and numeric
    string items are reported to the :class:`_Strings` the call runs in. A
    complex number, a larger array or anything ``float()`` refuses is an
    error.
    """
    out = np.empty(arr.shape)
    strings = 0
    for index, item in np.ndenumerate(arr):
        if isinstance(item, np.ndarray) and item.size == 1:
            item = item.reshape(())[()]
        if np.iscomplexobj(item) or isinstance(item, np.ndarray):
            raise ValueError(f"{what} is not real-numeric")
        try:
            out[index] = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{what} is not real-numeric") from None
        strings += isinstance(item, (str, bytes))
    if strings:
        _Strings.note(what, strings)
    return out


def _deprecated(message: str) -> None:
    """A FutureWarning for a call anneal 0.10.0 took that will raise later.

    The warning points at the first caller outside this module.
    """
    frame, depth = sys._getframe(), 0
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame, depth = frame.f_back, depth + 1
    warnings.warn(message, FutureWarning, stacklevel=depth + 1)


_STRINGS: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "anneal_chemfit_strings", default=None
)


class _Strings:
    """The numeric strings one call reads, named in one FutureWarning at its end.

    PyYAML reads ``1e-3`` and ``1e1`` as strings, so a parameter value, a
    bound, a span or a preset keyword from a YAML file may be one. A call
    made inside another reports its strings to the outer one.
    """

    def __init__(self, caller: str):
        self.caller = caller
        self.found: dict[str, int] = {}
        self.token = None

    @staticmethod
    def note(what: str, count: int) -> None:
        """Report ``count`` strings read for ``what``."""
        found = _STRINGS.get()
        if found is None:
            _deprecated(_strings_message("anneal.chemfit", {what: count}))
        else:
            found.setdefault(what, count)

    def __enter__(self) -> None:
        if _STRINGS.get() is None:
            self.token = _STRINGS.set(self.found)

    def __exit__(self, kind, error, trace) -> None:
        if self.token is None:
            return
        _STRINGS.reset(self.token)
        if kind is None and self.found:
            _deprecated(_strings_message(self.caller, self.found))


def _strings_message(caller: str, found: dict[str, int]) -> str:
    names = list(found)
    shown = names if len(names) <= 3 else [*names[:2], f"{len(names) - 2} more"]
    one = sum(found.values()) == 1
    return (
        f"{caller} reads {_listed(shown)} from {'a string' if one else 'strings'}; "
        f"pass {'a number' if one else 'numbers'} instead. "
        "A string will raise in a future release."
    )


# ---------------------------------------------------------------------------
# The fitter's session protocol.
# ---------------------------------------------------------------------------

_PROTOCOLS = (("evaluate", "step"), ("ask", "tell"))


def _no_step() -> None:
    """The step notice of a fitter that has no step method."""


def _protocol(fitter: Any, *, without_step: str | None = None):
    """The fitter's ``(evaluate, step)`` methods, else its ``(ask, tell)``.

    A fitter with neither whole pair, or without ``finish``, is a TypeError.
    ``without_step`` names a bridge that, as in anneal 0.10.0, still drives
    ``evaluate`` or ``ask`` without its partner, with a FutureWarning: the
    step notices go to the other pair's ``tell`` or ``step`` when the fitter
    has one, and nowhere otherwise.
    """
    name = type(fitter).__name__
    half = None
    for evaluate_name, step_name in _PROTOCOLS:
        evaluate = getattr(fitter, evaluate_name, None)
        step = getattr(fitter, step_name, None)
        if callable(evaluate) and callable(step):
            break
        if half is None and callable(evaluate):
            half = evaluate_name, step_name, evaluate
    else:
        if without_step is None or half is None:
            raise TypeError(
                "the fitter needs evaluate and step (ChemFit) or ask and tell "
                f"(ChemFit 3.1); {name} has neither pair"
            )
        evaluate_name, step_name, evaluate = half
        step = None
    if not callable(getattr(fitter, "finish", None)):
        raise TypeError(f"the fitter needs finish; {name} has none")
    if step is None:
        other = "tell" if step_name == "step" else "step"
        step, notices = getattr(fitter, other, None), f"{other} notices"
        if not callable(step):
            step, notices = _no_step, "no step notices"
        _deprecated(
            f"{without_step} drives {name} through {evaluate_name} with {notices}, "
            f"since it has no {step_name}; give it a {step_name} method. "
            f"A fitter without {step_name} will raise in a future release."
        )
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

    A one-element list or tuple (a batch of one) is unwrapped first, and the
    loss is then read as :func:`_as_number` reads a number: a ``Decimal``,
    an ``np.matrix`` and another library's array of one element count.
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    number = _as_number(value)
    if number is None:
        raise TypeError(
            f"the fitter returned a loss of type {type(value).__name__}; "
            "a loss must be one real number"
        )
    return float(number)


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

    ``kind`` is ``"number"`` for a Python number, a ``Decimal`` included,
    ``"scalar"`` for a NumPy scalar, ``"sequence"`` for a list or tuple and
    ``"array"`` for anything else array-like. ``dtype`` is the float type a
    candidate value is cast to; integer, bool and object leaves are
    optimized as float64.
    """
    if isinstance(value, np.generic):
        kind, raw, shape = "scalar", value.dtype, ()
    elif isinstance(value, np.ndarray):
        kind, raw, shape = "array", value.dtype, value.shape
    elif isinstance(value, (numbers.Real, Decimal)):
        return "number", np.dtype(np.float64), ()
    else:
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name} is not real-numeric") from error
        kind = "sequence" if isinstance(value, (list, tuple)) else "array"
        raw, shape = arr.dtype, arr.shape
    if raw.kind in "biuO":
        return kind, np.dtype(np.float64), shape
    if raw.kind != "f":
        raise ValueError(f"parameter {name} is not real-numeric")
    if np.finfo(raw).nmant > _FLOAT64_MANTISSA:
        raise TypeError(
            f"parameter {name} is {raw.name}, which the float64 drivers cannot "
            "carry exactly; convert it to float64 first"
        )
    return kind, raw, shape


def _items_form(value: Any) -> Any:
    """What :func:`_items_like` needs to rebuild a list or tuple leaf.

    A list or tuple becomes ``(list or tuple, the forms of its items)``, a
    NumPy array or scalar becomes its base type, and a number ``float``.
    """
    if isinstance(value, (list, tuple)):
        kind = tuple if isinstance(value, tuple) else list
        return kind, [_items_form(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.ndarray
    if isinstance(value, np.generic):
        return np.generic
    return float


def _items_like(form: Any, values: np.ndarray) -> Any:
    """``values``, an array in the leaf's shape and dtype, laid out as ``form``."""
    if isinstance(form, tuple):
        kind, items = form
        if all(item is float for item in items):
            out = values.tolist()
        else:
            out = [_items_like(item, values[i]) for i, item in enumerate(items)]
        return out if kind is list else tuple(out)
    if form is np.ndarray:
        return np.array(values)
    if form is np.generic:
        return values[()]
    return float(values)


def _numeric_leaf(name: str, value: Any) -> Any:
    """``value`` with its strings and objects read as floats, laid out the same.

    A string leaf becomes a float, and a list, tuple or array of strings
    one of floats; each is reported to the :class:`_Strings` the call runs
    in. A leaf NumPy holds only as objects, such as a ``Decimal``, a list
    of them or an object array, is read item by item the same way. Any
    other leaf is returned as it is.
    """
    if isinstance(value, (str, bytes)):
        return float(_real_array(value, f"parameter {name}"))
    if isinstance(value, (numbers.Real, np.generic)):
        return value
    try:
        kind = np.asarray(value).dtype.kind
    except (TypeError, ValueError):
        return value
    if kind not in "OSU":
        return value
    values = _real_array(value, f"parameter {name}")
    if isinstance(value, (list, tuple)):
        return _items_like(_items_form(value), values)
    if isinstance(value, np.ndarray) or values.ndim:
        return values
    return float(values)


class _Leaf:
    """One parameter leaf: its path, its slice of the flat vector, its type."""

    __slots__ = ("path", "name", "kind", "dtype", "shape", "size", "offset", "form")

    def __init__(self, path: tuple, value: Any, offset: int):
        self.path = path
        self.name = _path_str(path)
        self.kind, self.dtype, self.shape = _leaf_form(self.name, value)
        self.size = math.prod(self.shape)
        self.offset = offset
        self.form = _items_form(value) if self.kind == "sequence" else None

    @property
    def span(self) -> slice:
        return slice(self.offset, self.offset + self.size)

    def rebuild(self, chunk: np.ndarray) -> Any:
        """A fresh leaf holding ``chunk``, in the leaf's own type."""
        if self.kind == "number":
            return float(chunk[0])
        if self.kind == "scalar":
            return self.dtype.type(chunk[0])
        values = chunk.astype(self.dtype).reshape(self.shape)
        if self.kind == "sequence":
            return _items_like(self.form, values)
        return values


class _Layout:
    """Where each leaf of a nested parameter mapping sits in the flat vector.

    A numeric string leaf, or one NumPy holds only as objects, such as a
    ``Decimal`` or an object array, is laid out as the floats it reads as,
    as anneal 0.10.0 read it.
    """

    def __init__(self, template: Mapping):
        self.leaves: list[_Leaf] = []
        offset = 0
        for path, value in _iter_leaves(template):
            leaf = _Leaf(path, _numeric_leaf(_path_str(path), value), offset)
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


def _bounds_every_side(layout: _Layout, bounds: Any) -> bool:
    """Whether a bounds mapping gives both sides of every leaf."""
    try:
        _mapped_box(layout, bounds, None, None)
    except (TypeError, ValueError):
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


# ---------------------------------------------------------------------------
# Classical presets.
# ---------------------------------------------------------------------------


def _preset_types() -> dict[str, type]:
    from anneal import Boltzmann, Fast, Gsa

    return {"boltzmann": Boltzmann, "fast": Fast, "gsa": Gsa}


def _preset_driver(preset: Any) -> str:
    """The classical driver a preset instance belongs to."""
    for name, kind in _preset_types().items():
        if isinstance(preset, kind):
            return name
    raise TypeError(
        f"preset must be Boltzmann(), Fast() or Gsa(), got {type(preset).__name__}"
    )


_PRESET_SCALES = {"boltzmann": "sigma", "fast": "gamma"}


def _check_preset(driver: str, preset: Any) -> None:
    """Refuse the preset values ``anneal.run`` refuses, before the fitter starts."""
    for name in ("t_init", _PRESET_SCALES.get(driver)):
        if name is None:
            continue
        value = getattr(preset, name)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if driver == "gsa":
        if not 1.0 < preset.q_v < 3.0:
            raise ValueError(f"q_v must lie in (1, 3), got {preset.q_v!r}")
        if not math.isfinite(preset.q_a):
            raise ValueError(f"q_a must be finite, got {preset.q_a!r}")


def _preset_kwargs(value: Any) -> dict:
    """``preset_kwargs`` as a dict; anything but a mapping or None is a TypeError."""
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"preset_kwargs must be a dict, got {type(value).__name__}")
    return dict(value or {})


def _classical_preset(driver: str, preset: Any = None, kwargs: Mapping | None = None):
    """The preset the classical ``driver`` runs, built and checked now."""
    kwargs = dict(kwargs or {})
    if preset is None:
        preset = _preset_types()[driver](**kwargs)
    elif kwargs:
        raise ValueError("pass a preset or preset_kwargs, not both")
    elif _preset_driver(preset) != driver:
        raise ValueError(f"a {type(preset).__name__} preset does not run the {driver!r} driver")
    _check_preset(driver, preset)
    return preset


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
    can rebuild the structure. Only real-numeric leaves are supported (a
    ``Decimal`` or an object array is read item by item, and a numeric
    string as a number, with a FutureWarning); non-finite starts are
    rejected, and so are ``longdouble`` leaves, which a float64 vector
    cannot carry exactly.
    """
    if not isinstance(params, Mapping) or not params:
        raise ValueError("params must be a non-empty dict")
    with _Strings("flatten_parameters"):
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
    the type of the matching ``template`` leaf: a Python number, a
    ``Decimal`` included, comes back as a float, a NumPy scalar or array
    keeps its floating dtype, an integer, bool or object leaf comes back as
    float64, and a list or tuple comes back as a list or tuple nested the
    same way. The returned dict mirrors ``template``'s nesting and never
    mutates the template.
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


def _start_vector(layout: _Layout, x0: Any, *, fill: bool = False) -> np.ndarray:
    """The flat start from ``x0``: a dict mirroring the parameters, or a vector.

    A dict leaf with its parameter's size but another shape is read in C
    order, as anneal 0.10.0 read it, with a FutureWarning. When ``fill`` is
    true, one value for the only parameter fills it, as 0.10.0 filled a
    parameter the fitter bounds on both sides, with a FutureWarning too.
    """
    if isinstance(x0, Mapping):
        paths = {path for path, _ in _iter_leaves(x0)}
        if paths != {leaf.path for leaf in layout.leaves}:
            raise ValueError("x0 dict must mirror fitter.initial_parameters")
        if fill and len(layout.leaves) == 1:
            leaf = layout.leaves[0]
            value = _lookup(x0, leaf.path)
            one = _real_array(value, f"x0 {leaf.name}").reshape(-1)
            if one.size == 1 < leaf.size:
                if not np.isfinite(one[0]):
                    raise ValueError(f"x0 {leaf.name} must be finite")
                _deprecated(
                    f"x0 {leaf.name} has shape {np.shape(value)}, but the parameter "
                    f"has shape {leaf.shape}; its one value fills the parameter. Pass "
                    "it in the parameter's shape; another shape will raise in a "
                    "future release."
                )
                return np.full(leaf.size, one[0])
        start = layout.vector(x0, "x0", exact=False)
        for leaf in layout.leaves:
            shape = np.shape(_lookup(x0, leaf.path))
            if shape != leaf.shape:
                _deprecated(
                    f"x0 {leaf.name} has shape {shape}, but the parameter has shape "
                    f"{leaf.shape}; its values are read in C order. Pass it in the "
                    "parameter's shape; another shape will raise in a future release."
                )
        return start
    start = _real_array(x0, "x0").reshape(-1)
    if start.size != layout.size:
        raise ValueError(
            f"x0 has length {start.size} but the parameters flatten to {layout.size}"
        )
    bad = _first(~np.isfinite(start))
    if bad is not None:
        raise ValueError(
            f"x0 must contain only finite values; {layout.name(bad)} is "
            f"{float(start[bad])!r}"
        )
    return start


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
        ``initial_parameters``; a nested dict with the same structure and
        leaf shapes, or a flat vector of the flattened dimension, overrides
        it. A start outside the box is moved onto it. A dict leaf with the
        parameter's size but another shape is read in C order, and one
        value for the fitter's only parameter, when ``fitter.bounds``
        gives both of its sides and ``low`` and ``high`` are omitted, fills
        it, as in anneal 0.10.0; each gives a FutureWarning and will raise
        in a future release.
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
        (e.g. ``{"t_init": 5.0}``); classical drivers only. Under the
        portfolio they are ignored with a FutureWarning, and will raise in
        a future release.

    Every argument is checked before ``fitter.init()``. Returns what
    ``fitter.finish`` returns for the best evaluated parameters (ChemFit
    returns them as given); each leaf keeps its type, shape and floating
    dtype. The first exception raised by the fitter, or a loss that is not a
    real number, stops the fit and is raised without calling ``finish``.
    """
    return _fit_anneal(
        fitter,
        budget,
        driver=driver,
        seed=seed,
        x0=x0,
        low=low,
        high=high,
        bound_span=bound_span,
        steps_per_epoch=steps_per_epoch,
        preset_kwargs=preset_kwargs,
    )


def _fit_anneal(
    fitter: Any,
    budget: int,
    *,
    driver: str = "portfolio",
    seed: int = 0,
    x0: Any = None,
    low: Any = None,
    high: Any = None,
    bound_span: float = 3.0,
    steps_per_epoch: int = 100,
    preset_kwargs: Any = None,
    preset: Any = None,
    caller: str = "fit_anneal",
) -> Any:
    """:func:`fit_anneal`, also taking a preset instance from :func:`run_fitter`.

    ``caller`` names the bridge in the warning about numeric strings.
    """
    evaluate, step = _protocol(fitter)
    name = _choice(driver, _DRIVERS)
    if name is None:
        raise ValueError(
            f"driver must be 'portfolio' or one of {', '.join(_CLASSICAL_DRIVERS)}; "
            f"got {driver!r}"
        )
    budget = _whole("budget", budget, 1)
    seed = _whole("seed", seed, 0, _SEED_LIMIT)
    steps = min(_whole("steps_per_epoch", steps_per_epoch, 1), budget)
    with _Strings(caller):
        span = _positive("bound_span", bound_span)
        kwargs = _preset_kwargs(preset_kwargs)
        if name == "portfolio":
            if kwargs:
                _deprecated(
                    "fit_anneal ignores preset_kwargs under the portfolio driver, "
                    "which takes no preset; leave them out, or pass "
                    "driver='boltzmann', 'fast' or 'gsa' to use them. This will "
                    "raise in a future release."
                )
            preset = None
        else:
            preset = _classical_preset(name, preset, kwargs)

        initial = getattr(fitter, "initial_parameters", None)
        if not isinstance(initial, Mapping) or not initial:
            raise ValueError("fitter.initial_parameters must be a non-empty dict")
        layout = _Layout(initial)
        start, _ = flatten_parameters(initial)
        if x0 is not None:
            fill = (
                low is None
                and high is None
                and _bounds_every_side(layout, getattr(fitter, "bounds", None))
            )
            start = _start_vector(layout, x0, fill=fill)
        if (low is None) != (high is None):
            raise ValueError("low and high must be given together")
        if low is None:
            box = _mapped_box(layout, getattr(fitter, "bounds", None), start, span)
        else:
            box = (_real_array(low, "low").ravel(), _real_array(high, "high").ravel())
            if box[0].size != layout.size or box[1].size != layout.size:
                raise ValueError(
                    f"low/high have lengths {box[0].size}/{box[1].size} "
                    f"but the flattened parameters have dimension {layout.size}"
                )
        # run and global_optimize refuse a start outside the box. A caller vector
        # such as zeros is pulled onto the box before the first evaluation.
        problem = _Problem(layout, *_settle(layout, *box), start)

    # The fitter owns bookkeeping; every evaluation is one optimizer step.
    _init(fitter)
    session = _Session(problem, evaluate, step, budget, step_every=1)
    return _finish(fitter, _drive(session, name, seed, preset, steps))


# ---------------------------------------------------------------------------
# fit_chemfit and its vector view.
# ---------------------------------------------------------------------------


class ChemFitVector:
    """Flattened view of ChemFit (possibly nested, array-valued) parameters.

    Scalar leaves become one coordinate each; array leaves expand element-wise
    in C order under the same dotted key. The vector layout is fixed at
    construction, so :meth:`pack` / :meth:`unpack` round-trip between anneal's
    flat box and ChemFit's nested parameter dicts; :meth:`unpack` gives each
    leaf the template's type, shape and floating dtype.
    """

    def __init__(self, template: dict[str, Any]):
        with _Strings("ChemFitVector"):
            self._layout = _Layout(template)
            self.x0 = self._layout.vector(template, "parameter")
        self.keys: list[str] = self._layout.names()
        self.shapes: dict[str, tuple[int, ...]] = {
            leaf.name: leaf.shape for leaf in self._layout.leaves
        }

    @property
    def dim(self) -> int:
        return self._layout.size

    def _base_key(self, key: str) -> str:
        return key.split("[", 1)[0]

    def pack(self, params: dict[str, Any]) -> np.ndarray:
        """Flatten a nested parameter dict into the fixed vector layout."""
        with _Strings("ChemFitVector.pack"):
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
    with _Strings("chemfit_box"):
        span = _positive("default_span", default_span)
        box = _mapped_box(layout, getattr(fitter, "bounds", None), vector.x0, span)
        low, high = _settle(layout, *box)
    fixed = _first(low == high)
    if fixed is not None:
        raise ValueError(
            f"ChemFit bound for {layout.name(fixed)!r} is empty: "
            f"lower={float(low[fixed])} upper={float(high[fixed])}; "
            "a driver box needs lower < upper"
        )
    return low, high


_CHEMFIT_PRESET_DEFAULTS = {
    "boltzmann": {"t_init": 5.0, "sigma": 0.5},
    "fast": {"t_init": 3.0, "gamma": 0.5},
    "gsa": {"t_init": 3.0, "q_v": 2.62, "q_a": 1.7},
}


_PRESET_KEYWORDS = ("t_init", "sigma", "gamma", "q_v", "q_a")


def _listed(names: Any) -> str:
    """``a``, ``a and b``, or ``a, b and c``."""
    names = list(names)
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _chemfit_preset(method: str, preset_kwargs: dict[str, Any]):
    """The preset of a :func:`fit_chemfit` method, from its keyword defaults.

    A preset keyword the method does not take is ignored, as anneal 0.10.0
    ignored it, with a FutureWarning; any other keyword is a TypeError. Each
    keyword the method takes is read as :func:`_float_setting` reads a float
    setting.
    """
    defaults = _CHEMFIT_PRESET_DEFAULTS.get(method, {})
    unknown = [key for key in preset_kwargs if key not in _PRESET_KEYWORDS]
    if unknown:
        got = (
            f"an unexpected keyword argument {unknown[0]!r}"
            if len(unknown) == 1
            else f"unexpected keyword arguments {', '.join(map(repr, unknown))}"
        )
        raise TypeError(
            f"fit_chemfit() got {got}; its preset keywords are t_init, sigma, "
            "gamma, q_v and q_a"
        )
    ignored = [key for key in preset_kwargs if key not in defaults]
    if ignored and method == "portfolio":
        them = "it" if len(ignored) == 1 else "them"
        _deprecated(
            f"fit_chemfit ignores {_listed(ignored)} under method 'portfolio', "
            f"which takes no preset; leave {them} out, or pass method='boltzmann', "
            f"'fast' or 'gsa' to use {them}. This will raise in a future release."
        )
    elif ignored:
        _deprecated(
            f"fit_chemfit ignores {_listed(ignored)}, which method {method!r} does "
            f"not take; it takes {_listed(defaults)}. Pass only those; a preset "
            "keyword of another method will raise in a future release."
        )
    if method == "portfolio":
        return None
    values = {}
    for key, default in defaults.items():
        value = preset_kwargs.get(key, default)
        number = _float_setting(key, value)
        if number is None:
            raise TypeError(f"{key} must be a number, got {value!r}")
        values[key] = number
    return _classical_preset(method, kwargs=values)


def _initial_mapping(initial: Any) -> Mapping:
    """``fitter.initial_parameters`` for :func:`fit_chemfit`, as a mapping.

    ChemFit 3.1's Fitter keeps ``initial_params`` as given, and anneal 0.10.0
    read a list or tuple of ``(key, value)`` pairs as the dict they make; that
    still runs, with a FutureWarning. Anything else that is not a mapping is
    a TypeError.
    """
    if isinstance(initial, Mapping):
        return initial
    pairs = None
    if isinstance(initial, (list, tuple)):
        try:
            pairs = dict(initial)
        except (TypeError, ValueError):
            pass
    if pairs is None:
        raise TypeError("fitter.initial_parameters must be a mapping")
    if pairs:
        _deprecated(
            f"fit_chemfit reads fitter.initial_parameters, a {type(initial).__name__} "
            "of (key, value) pairs, as a dict; give the fitter a dict. A fitter "
            "whose initial_parameters is not a mapping will raise in a future "
            "release."
        )
    return pairs


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
            ``tell`` (ChemFit 3.1). ``initial_parameters`` given as a list
            or tuple of ``(key, value)`` pairs, which ChemFit 3.1 keeps, is
            read as the dict they make, as in anneal 0.10.0, with a
            FutureWarning; it will raise in a future release.
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
        **preset_kwargs: ``t_init`` / ``sigma`` (boltzmann), ``t_init`` /
            ``gamma`` (fast), ``t_init`` / ``q_v`` / ``q_a`` (gsa) for the
            classical preset constructors. One of these that the method
            does not take, or any under the portfolio, is ignored with a
            FutureWarning, as 0.10.0 ignored it, and will raise in a
            future release; any other keyword is a TypeError.

    Every argument is checked before ``fitter.init()``.

    Returns:
        What ``fitter.finish(best_params)`` returns; ChemFit returns
        ``best_params``, whose leaves keep their types. The first exception
        raised by the fitter, or a loss that is not a real number, stops the
        fit and is raised without calling ``finish``.
    """
    evaluate, step = _protocol(fitter)
    name = _choice(method, _DRIVERS)
    if name is None:
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )
    budget = _whole("budget", budget, 1)
    seed = _whole("seed", seed, 0, _SEED_LIMIT)
    with _Strings("fit_chemfit"):
        span = _positive("default_span", default_span)
        tell_every = _whole("tell_every", tell_every, 1)
        steps = min(_whole("steps_per_epoch", steps_per_epoch, 1), budget)
        preset = _chemfit_preset(name, preset_kwargs)

        initial = _initial_mapping(getattr(fitter, "initial_parameters", None))
        vector = ChemFitVector(initial)
        if vector.dim == 0:
            raise ValueError("fitter.initial_parameters holds no parameters")
        layout = vector._layout
        box = _mapped_box(layout, getattr(fitter, "bounds", None), vector.x0, span)
        problem = _Problem(layout, *_settle(layout, *box), vector.x0)

    _init(fitter)
    every = tell_every if name == "portfolio" else steps
    session = _Session(problem, evaluate, step, budget, step_every=every)
    return _finish(fitter, _drive(session, name, seed, preset, steps))


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


def _flat_bound(value: Any, size: int, what: str) -> np.ndarray:
    """An explicit bound vector; a single value is broadcast."""
    arr = _real_array(value, what).reshape(-1)
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
        return _settle(
            layout,
            _flat_bound(low, layout.size, "low"),
            _flat_bound(high, layout.size, "high"),
        )
    if (
        isinstance(context_bounds, Mapping)
        and "low" in context_bounds
        and "high" in context_bounds
    ):
        return _settle(
            layout,
            _flat_bound(context_bounds["low"], layout.size, 'bounds["low"]'),
            _flat_bound(context_bounds["high"], layout.size, 'bounds["high"]'),
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
    with _Strings("bounds_from_fitter"):
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
    with _Strings("resolve_bounds"):
        layout = _Layout(initial)
        if layout.size == 0:
            raise ValueError("initial_params is empty")
        return _benchmark_box(layout, low, high, context_bounds, fitter_bounds)


def _benchmark_preset(method: str, preset: Any) -> tuple[str, Any]:
    """The driver and checked preset :func:`run_benchmark` runs for ``method``."""
    if preset is None:
        return method, None if method == "portfolio" else _classical_preset(method)
    own = _preset_driver(preset)
    kind = type(preset).__name__
    if method == "portfolio":
        _deprecated(
            f"run_benchmark ignores the {kind} preset under method 'portfolio', "
            f"which takes none; leave it out, or pass method={own!r} to run it. "
            "This will raise in a future release."
        )
        return "portfolio", None
    if method == "sa":
        _deprecated(
            f"run_benchmark has no method 'sa'; it runs the {kind} preset as "
            f"method {own!r}. Pass method={own!r}; 'sa' will raise in a future "
            "release."
        )
    elif method != own:
        _deprecated(
            f"run_benchmark runs the {kind} preset it was given, not method "
            f"{method!r}; pass method={own!r} with it. A preset of another method "
            "will raise in a future release."
        )
    return own, _classical_preset(own, preset)


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
    ``boltzmann``, ``fast``, or ``gsa``; ``preset``, when given, is the
    matching preset instance. The chain starts at ``initial_params``, moved
    onto the box when it lies outside. Every coordinate the fitter sees lies
    in the box, and ``budget`` caps the evaluations, the start included. The
    fitter is driven through ``evaluate`` / ``step`` or ``ask`` / ``tell``,
    with one step notice per evaluation, and each parameter leaf keeps its
    type.

    As in anneal 0.10.0, a preset of another method, or ``method="sa"``
    with a preset, runs the preset; a preset under the portfolio is ignored;
    and a fitter with ``evaluate`` but no ``step``, or ``ask`` but no
    ``tell``, is driven with its ``tell`` or ``step`` as the step notice, or
    without step notices when it has neither. Each gives a FutureWarning and
    will raise in a future release.

    ``low`` and ``high`` may be vectors or scalars (broadcast). When they
    are omitted, bounds are read from ``benchmark_context["bounds"]`` or
    ``fitter.bounds``. A coordinate whose two bounds are equal is held fixed.
    Every argument is checked before ``fitter.init()``; the first exception
    raised by the fitter is raised without calling ``finish``.
    """
    fitter = benchmark_context["fitter"]
    evaluate, step = _protocol(fitter, without_step="run_benchmark")
    key = _choice(method, (*_DRIVERS, "sa"))
    if key is None or (key == "sa" and preset is None):
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )
    budget = _whole("budget", benchmark_context["budget"], 1)
    seed = _whole("seed", seed, 0, _SEED_LIMIT)
    steps = min(_whole("steps_per_epoch", steps_per_epoch, 1), budget)
    name, preset = _benchmark_preset(key, preset)

    initial = benchmark_context["initial_params"]
    if not isinstance(initial, Mapping):
        raise TypeError("initial_params must be a mapping")
    with _Strings("run_benchmark"):
        layout = _Layout(initial)
        if layout.size == 0:
            raise ValueError("initial_params is empty")
        start = layout.vector(initial, "parameter")
        box = _benchmark_box(
            layout,
            low,
            high,
            benchmark_context.get("bounds"),
            getattr(fitter, "bounds", None),
        )
        problem = _Problem(layout, *box, start)

    _init(fitter)
    session = _Session(problem, evaluate, step, budget, step_every=1)
    return _finish(fitter, _drive(session, name, seed, preset, steps))


_FITTER_METHODS = {
    "global_optimize": "portfolio",
    "sa": None,
    "portfolio": "portfolio",
    "boltzmann": "boltzmann",
    "fast": "fast",
    "gsa": "gsa",
}
_FITTER_OPTIONS = ("x0", "low", "high", "bound_span", "steps_per_epoch", "preset_kwargs")


def _fitter_keyword_error(key: str) -> TypeError:
    """The error for a keyword :func:`run_fitter` does not pass on."""
    message = (
        f"run_fitter() got an unexpected keyword argument {key!r}; it passes only "
        f"{', '.join(_FITTER_OPTIONS[:-1])} and {_FITTER_OPTIONS[-1]} on to fit_anneal"
    )
    if key in ("tell_every", "default_span"):
        message += f" ({key} is an option of fit_chemfit)"
    elif key in _PRESET_KEYWORDS:
        message += f" (pass {key} in preset_kwargs)"
    return TypeError(message)


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
    ``method="sa"`` runs ``preset`` (``Boltzmann()`` when none is given),
    both through :func:`fit_anneal`; ``"portfolio"``, ``"boltzmann"``,
    ``"fast"`` and ``"gsa"`` name a :func:`fit_anneal` driver directly, and
    a classical one runs ``preset`` when it is that driver's kind.
    The keywords ``x0``, ``low``, ``high``, ``bound_span``,
    ``steps_per_epoch`` and ``preset_kwargs`` go to :func:`fit_anneal`; any
    other keyword is a TypeError.

    As in anneal 0.10.0, a preset is ignored under the portfolio or with a
    classical method of another kind, and so is ``preset_kwargs`` under the
    portfolio. Each gives a FutureWarning and will raise in a future release.
    """
    if hasattr(fitter, "fit_anneal"):
        return fitter.fit_anneal(
            budget=budget, method=method, preset=preset, seed=seed, **kwargs
        )
    for key in kwargs:
        if key not in _FITTER_OPTIONS:
            raise _fitter_keyword_error(key)
    key = _choice(method, _FITTER_METHODS)
    if key is None:
        raise ValueError(
            f"method must be one of {', '.join(map(repr, _FITTER_METHODS))}; "
            f"got {method!r}"
        )
    options = dict(kwargs)
    preset_kwargs = _preset_kwargs(options.pop("preset_kwargs", None))
    driver = _FITTER_METHODS[key]
    if preset is not None:
        own = _preset_driver(preset)
        if driver is None:
            driver = own
        elif driver != own:
            runs = "the portfolio"
            if driver != "portfolio":
                runs = f"its own {driver} preset"
            _deprecated(
                f"run_fitter ignores the {type(preset).__name__} preset under method "
                f"{method!r}, which runs {runs}; leave it out, or pass method='sa' "
                "to run it. This will raise in a future release."
            )
            preset = None
    elif driver is None:
        driver = "boltzmann"
    if driver == "portfolio" and preset_kwargs:
        _deprecated(
            f"run_fitter ignores preset_kwargs under method {method!r}, which runs "
            "the portfolio; leave them out, or pass method='boltzmann', 'fast' or "
            "'gsa' to use them. This will raise in a future release."
        )
        preset_kwargs = {}
    return _fit_anneal(
        fitter,
        budget,
        driver=driver,
        seed=seed,
        preset=preset,
        preset_kwargs=preset_kwargs,
        caller="run_fitter",
        **options,
    )
