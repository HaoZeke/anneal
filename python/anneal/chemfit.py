"""ChemFit bridges for the gradient-free drivers.

ChemFit's ``Fitter`` runs a session that the caller drives: ``init``, one
evaluation per candidate, one notice per optimizer step, then ``finish``.
Current ChemFit names the middle two ``evaluate`` and ``step``; ChemFit 3.1
named them ``ask`` and ``tell``. Every bridge here drives ``evaluate`` and
``step`` when the fitter has both, ``ask`` and ``tell`` otherwise, and refuses
a fitter with neither pair before calling ``init``. ``finish`` receives the
best evaluated parameters and its return value is the result; ChemFit returns
the parameters it was given.

Nested parameter dictionaries are flattened only at the optimizer
boundary and rebuilt on the way back. Every evaluation stays inside the box,
and the fitter's initial parameters are the start unless a caller passes
another ``x0``. The default driver is the Thompson-allocated portfolio.

The first exception the fitter raises, or a loss that is not a real number,
ends the drive: the fitter is not called again, ``finish`` is skipped, and
the exception reaches the caller.
"""

from __future__ import annotations

import inspect
import math
import numbers
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


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a real number other than a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


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


def _path_str(path: tuple) -> str:
    return ".".join(str(key) for key in path)


def _iter_leaves(params: dict[str, Any], path: tuple = ()):
    """Yield ``(path, value)`` for every non-dict leaf of a nested dict."""
    for key, value in params.items():
        if isinstance(value, dict):
            yield from _iter_leaves(value, path + (key,))
        else:
            yield path + (key,), value


def _assign_path(params: dict[str, Any], path: tuple, value: Any) -> None:
    for key in path[:-1]:
        params = params[key]
    params[path[-1]] = value


def _lookup_path(params: Any, path: tuple):
    node = params
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _deep_copy(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _deep_copy(value) for key, value in node.items()}
    if isinstance(node, np.ndarray):
        return node.copy()
    return node


def flatten_parameters(params: dict[str, Any]):
    """Flatten a nested parameter dict to ``(vector, spec)``.

    Scalar leaves contribute one entry; array leaves contribute their
    ravelled entries in C order. ``spec`` records ``(path, shape)`` per
    leaf (``shape == ()`` for scalars) so :func:`unflatten_parameters`
    can rebuild the structure. Only real-numeric leaves are supported;
    non-finite starts are rejected.
    """
    if not isinstance(params, dict) or not params:
        raise ValueError("params must be a non-empty dict")
    segments: list[np.ndarray] = []
    spec: list[tuple[tuple, tuple]] = []
    for path, value in _iter_leaves(params):
        try:
            arr = np.asarray(value, dtype=np.float64).ravel()
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"parameter {_path_str(path)} is not real-numeric"
            ) from e
        if arr.size == 0:
            raise ValueError(f"parameter {_path_str(path)} is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"parameter {_path_str(path)} must be finite")
        shape = np.shape(value)
        segments.append(arr)
        spec.append((path, shape if isinstance(shape, tuple) else ()))
    return np.concatenate(segments), spec


def spec_total(spec) -> int:
    """Flattened dimension of a :func:`flatten_parameters` spec."""
    return sum(int(np.prod(shape)) if shape != () else 1 for _, shape in spec)


def unflatten_parameters(vector: np.ndarray, spec, template: dict[str, Any]):
    """Rebuild a nested parameter dict from a flat vector and a spec.

    Array leaves are reshaped to their original shape; the returned dict
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
        _assign_path(out, path, chunk[0] if shape == () else chunk.reshape(shape))
        offset += size
    return out


def _bound_pair(entry: Any, size: int, fallback: tuple[np.ndarray, np.ndarray]):
    """Normalize one bounds leaf to ``(low, high)`` arrays of length ``size``."""
    if entry is None:
        return fallback
    try:
        low_raw, high_raw = entry
    except (TypeError, ValueError) as e:
        raise ValueError(f"bounds entry {entry!r} must be a (lower, upper) pair") from e
    low_fb, high_fb = fallback
    low = (
        np.broadcast_to(np.asarray(low_raw, dtype=np.float64), (size,)).copy()
        if low_raw is not None
        else low_fb
    )
    high = (
        np.broadcast_to(np.asarray(high_raw, dtype=np.float64), (size,)).copy()
        if high_raw is not None
        else high_fb
    )
    return low, high


def _resolve_bounds(
    fitter_bounds: Any,
    x0: np.ndarray,
    spec,
    bound_span: float,
    low: np.ndarray | None,
    high: np.ndarray | None,
):
    """Build flat ``(low, high)`` vectors for the flattened parameters."""
    dim = x0.size
    if (low is None) != (high is None):
        raise ValueError("low and high must be given together")
    if low is not None:
        low_arr = np.asarray(low, dtype=np.float64).ravel()
        high_arr = np.asarray(high, dtype=np.float64).ravel()
        if low_arr.size != dim or high_arr.size != dim:
            raise ValueError(
                f"low/high have lengths {low_arr.size}/{high_arr.size} "
                f"but the flattened parameters have dimension {dim}"
            )
        return low_arr, high_arr
    if not np.isfinite(bound_span) or bound_span <= 0:
        raise ValueError("bound_span must be positive and finite")
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    offset = 0
    for path, shape in spec:
        size = int(np.prod(shape)) if shape != () else 1
        center = x0[offset : offset + size]
        fallback = (center - bound_span, center + bound_span)
        entry = _lookup_path(fitter_bounds, path) if isinstance(fitter_bounds, dict) else None
        try:
            leaf_low, leaf_high = _bound_pair(entry, size, fallback)
        except ValueError as e:
            raise ValueError(f"bounds for {_path_str(path)}: {e}") from e
        lows.append(leaf_low)
        highs.append(leaf_high)
        offset += size
    return np.concatenate(lows), np.concatenate(highs)


# ---------------------------------------------------------------------------
# One drive of a fitter.
# ---------------------------------------------------------------------------


class _Stop(BaseException):
    """Ends a driver run from inside the objective; the bridge catches it."""


class _Session:
    """The objective a driver calls: one fitter evaluation per call.

    It sends a step notice every ``step_every`` evaluations and records the
    first exception from the fitter or from reading its loss. That exception
    stops the driver.
    """

    def __init__(self, candidate, evaluate, step, budget: int, step_every: int):
        self._candidate = candidate
        self._evaluate = evaluate
        self._step = step
        self.budget = budget
        self.step_every = step_every
        self.count = 0
        self.error: BaseException | None = None

    def __call__(self, x) -> float:
        if self.error is not None:
            raise _Stop
        try:
            point = np.array(x, dtype=np.float64)
            loss = _loss_value(self._evaluate(self._candidate(point)))
            self.count += 1
            if self.count % self.step_every == 0:
                self._step()
        except BaseException as error:
            self.error = error
            raise _Stop from None
        return loss


def _drive(
    session: _Session,
    low: np.ndarray,
    high: np.ndarray,
    start: np.ndarray | None,
    driver: str,
    seed: int,
    preset: Any = None,
    steps_per_epoch: int = 1,
) -> np.ndarray:
    """Run ``driver`` on the session and return the best point it reports.

    The first fitter exception is raised here, after the driver has returned.
    """
    from anneal import global_optimize, run

    budget = session.budget
    best = None
    try:
        if driver == "portfolio":
            best = global_optimize(session, low, high, budget, seed=seed, x0=start)["best_pos"]
        else:
            best = run(
                session,
                low,
                high,
                preset,
                n_epochs=max(1, budget // steps_per_epoch),
                steps_per_epoch=steps_per_epoch,
                seed=seed,
                x0=start,
            ).best_pos
    except _Stop:
        pass
    if session.error is not None:
        raise session.error
    return np.asarray(best, dtype=np.float64)


def _classical_preset(driver: str, preset_kwargs: dict[str, Any] | None):
    from anneal import Boltzmann, Fast, Gsa

    kwargs = dict(preset_kwargs or {})
    if driver == "boltzmann":
        return Boltzmann(**kwargs)
    if driver == "fast":
        return Fast(**kwargs)
    return Gsa(**kwargs)


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
      budget: total objective-evaluation budget. The portfolio charges one
        unit per evaluation; the classical drivers run
        ``n_epochs * steps_per_epoch <= budget`` evaluations.
      driver: ``"portfolio"`` (default; Thompson-allocated SOTA driver
        over the gradient-free arms — QMC restarts, basin hopping,
        differential evolution, GSA, parallel tempering — with no
        gradient), or one of ``"boltzmann"``, ``"fast"``, ``"gsa"`` for
        single-chain ablation runs.
      seed: RNG seed.
      x0: warm start. ``None`` (default) uses the fitter's
        ``initial_parameters``; a nested dict with the same structure or
        a flat vector of the flattened dimension overrides it.
      low, high: explicit flat bound vectors. When omitted, bounds come
        from the fitter's ``bounds`` dict (``(lower, upper)`` pairs
        mirroring ``initial_params``); entries without bounds fall back
        to ``x0 +/- bound_span``.
      bound_span: half-width of the fallback box around unbounded entries.
      steps_per_epoch: classical-driver epoch width; epochs are derived
        as ``max(1, budget // steps_per_epoch)``.
      preset_kwargs: extra kwargs for the preset constructor
        (e.g. ``{"t_init": 5.0}``); classical drivers only.

    Returns what ``fitter.finish`` returns for the best evaluated parameters
    (ChemFit returns them as given), with array leaves restored to their
    original shapes. The first exception raised by the fitter, or a loss
    that is not a real number, stops the fit and is raised without calling
    ``finish``.
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
    fitter_bounds = getattr(fitter, "bounds", None) or {}

    start_vector, spec = flatten_parameters(initial_parameters)
    template = initial_parameters
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

    low_vec, high_vec = _resolve_bounds(
        fitter_bounds, start_vector, spec, float(bound_span), low, high
    )
    # run and global_optimize refuse a start outside the box. A caller vector
    # such as zeros is pulled onto the box before the first evaluation.
    start_vector = np.minimum(np.maximum(start_vector, low_vec), high_vec)

    # The fitter owns bookkeeping; every evaluation is one optimizer step.
    _init(fitter)
    preset = None if driver == "portfolio" else _classical_preset(driver, preset_kwargs)
    steps = max(1, min(int(steps_per_epoch), budget))
    session = _Session(
        lambda x: unflatten_parameters(x, spec, template), evaluate, step, budget, step_every=1
    )
    best_pos = _drive(session, low_vec, high_vec, start_vector, driver, int(seed), preset, steps)

    best_params = unflatten_parameters(best_pos, spec, template)
    return _finish(fitter, best_params)

try:  # ChemFit flattens nested dicts with dotted keys; mirror that order.
    from pydictnest import flatten_dict as _pydict_flatten
    from pydictnest import unflatten_dict as _pydict_unflatten
except ImportError:  # pragma: no cover - minimal fallback when chemfit is absent
    _pydict_flatten = None
    _pydict_unflatten = None


def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted keys, keeping array leaves intact."""
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_mapping(value, name))
        else:
            flat[name] = value
    return flat


def _unflatten_mapping(flat: dict[str, Any]) -> dict[str, Any]:
    """Invert dotted keys back into nested dicts."""
    if _pydict_unflatten is not None:
        return dict(_pydict_unflatten(dict(flat), dict_factory=dict))
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = str(key).split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


class ChemFitVector:
    """Flattened view of ChemFit (possibly nested, array-valued) parameters.

    Scalar leaves become one coordinate each; ``numpy`` array leaves expand
    element-wise in C order under the same dotted key. The vector layout is
    fixed at construction, so :meth:`pack` / :meth:`unpack` round-trip
    between anneal's flat box and ChemFit's nested parameter dicts.
    """

    def __init__(self, template: dict[str, Any]):
        flat = (
            dict(_pydict_flatten(template))
            if _pydict_flatten is not None
            else _flatten_mapping(template)
        )
        self.keys: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        values: list[float] = []
        for key, value in flat.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                self.keys.append(key)
                self.shapes[key] = ()
                values.append(float(arr))
            else:
                self.shapes[key] = arr.shape
                for index in np.ndindex(arr.shape):
                    self.keys.append(f"{key}[{','.join(map(str, index))}]")
                values.extend(float(v) for v in arr.reshape(-1))
        self.x0 = np.asarray(values, dtype=np.float64)

    @property
    def dim(self) -> int:
        return len(self.keys)

    def _base_key(self, key: str) -> str:
        return key.split("[", 1)[0]

    def pack(self, params: dict[str, Any]) -> np.ndarray:
        """Flatten a nested parameter dict into the fixed vector layout."""
        flat = (
            dict(_pydict_flatten(params))
            if _pydict_flatten is not None
            else _flatten_mapping(params)
        )
        out = np.empty(len(self.keys), dtype=np.float64)
        cursor = 0
        seen: set[str] = set()
        for key in self.keys:
            base = self._base_key(key)
            if base not in seen:
                seen.add(base)
                arr = np.asarray(flat[base], dtype=np.float64).reshape(-1)
                shape = self.shapes[base]
                size = int(np.prod(shape)) if shape != () else 1
                out[cursor : cursor + size] = arr
                cursor += size
        return out

    def unpack(self, vector: np.ndarray) -> dict[str, Any]:
        """Rebuild the nested parameter dict from a flat vector."""
        flat: dict[str, Any] = {}
        cursor = 0
        seen: set[str] = set()
        for key in self.keys:
            base = self._base_key(key)
            if base in seen:
                continue
            seen.add(base)
            shape = self.shapes[base]
            if shape == ():
                flat[base] = float(vector[cursor])
                cursor += 1
            else:
                size = int(np.prod(shape))
                flat[base] = np.asarray(vector[cursor : cursor + size]).reshape(shape)
                cursor += size
        return _unflatten_mapping(flat)


def _span_bound_pair(leaf: Any) -> tuple[Any, Any]:
    """Normalize one bounds leaf to a ``(lower, upper)`` pair of scalars."""
    if leaf is None:
        return (None, None)
    if isinstance(leaf, (list, tuple)) and len(leaf) == 2:
        try:
            lower = None if leaf[0] is None else float(leaf[0])
            upper = None if leaf[1] is None else float(leaf[1])
            return (lower, upper)
        except (TypeError, ValueError):
            pass
    arr = np.asarray(leaf)
    if arr.shape == (2,) and np.issubdtype(arr.dtype, np.number):
        return (float(arr[0]), float(arr[1]))
    return (None, None)


def chemfit_box(
    fitter: Any,
    vector: ChemFitVector,
    default_span: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite ``(low, high)`` box vectors for a ChemFit fitter.

    Bounds mirror ``fitter.bounds`` (same structure as the initial
    parameters). A parameter without bounds gets
    ``init +/- default_span``; scalar ``(lower, upper)`` pairs apply to
    every element of an array leaf.
    """
    raw_bounds: dict[str, Any] = getattr(fitter, "bounds", None) or {}
    flat_bounds = (
        dict(_pydict_flatten(raw_bounds))
        if _pydict_flatten is not None
        else _flatten_mapping(raw_bounds)
    )
    low = np.empty(vector.dim, dtype=np.float64)
    high = np.empty(vector.dim, dtype=np.float64)
    for i, key in enumerate(vector.keys):
        base = vector._base_key(key)
        lower, upper = _span_bound_pair(flat_bounds.get(base))
        center = float(vector.x0[i])
        if lower is None:
            lower = center - float(default_span)
        if upper is None:
            upper = center + float(default_span)
        if not lower < upper:
            raise ValueError(
                f"ChemFit bound for {base!r} is empty: lower={lower} upper={upper}"
            )
        low[i] = lower
        high[i] = upper
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
        budget: total objective evaluations (one ``evaluate`` or ``ask``
            each).
        method: ``"portfolio"`` (default; Thompson-allocated SOTA including
            parallel-tempering communicating chains), or ``"boltzmann"``,
            ``"fast"``, ``"gsa"`` for the bound-respecting classical chain
            from the matching initial parameters.
        seed: RNG seed forwarded to the anneal driver.
        default_span: half-width around the initial value for parameters
            ChemFit leaves unbounded.
        tell_every: portfolio evaluations between step notices (``step`` or
            ``tell``) so registered callbacks still fire.
        steps_per_epoch: classical-chain evaluations per epoch; epochs are
            derived as ``budget // steps_per_epoch``.
        **preset_kwargs: ``t_init`` / ``sigma`` / ``gamma`` / ``q_v`` /
            ``q_a`` forwarded to the classical preset constructors.

    Returns:
        What ``fitter.finish(best_params)`` returns; ChemFit returns
        ``best_params``. The first exception raised by the fitter, or a
        loss that is not a real number, stops the fit and is raised without
        calling ``finish``.
    """
    from anneal import Boltzmann, Fast, Gsa

    evaluate, step = _protocol(fitter)
    if int(budget) < 1:
        raise ValueError("budget must be positive")
    vector = ChemFitVector(dict(fitter.initial_parameters))
    if vector.dim == 0:
        raise ValueError("fitter.initial_parameters holds no parameters")
    low, high = chemfit_box(fitter, vector, default_span=default_span)

    _init(fitter)
    if method == "portfolio":
        preset, start, every = None, None, max(1, int(tell_every))
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
        preset = presets[method]
        start = np.asarray(vector.x0, dtype=np.float64)
        every = max(1, int(steps_per_epoch))
    else:
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )
    session = _Session(vector.unpack, evaluate, step, int(budget), step_every=every)
    steps = max(1, int(steps_per_epoch))
    best = _drive(session, low, high, start, method, int(seed), preset, steps)

    best_params = vector.unpack(best)
    return _finish(fitter, best_params)

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


def _as_float_vector(value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1 and size != 1:
        arr = np.full(size, float(arr[0]), dtype=np.float64)
    if arr.size != size:
        raise ValueError(f"bound length {arr.size} does not match parameter length {size}")
    return arr


def bounds_from_fitter(initial: dict[str, Any], fitter_bounds: Any, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a ChemFit bounds mapping that mirrors ``initial_params``."""
    if not isinstance(fitter_bounds, dict) or not fitter_bounds:
        return None
    flat_init, spec = flatten_params(initial)
    if flat_init.size != size:
        return None
    low_chunks: list[np.ndarray] = []
    high_chunks: list[np.ndarray] = []
    for path, shape in spec:
        cursor: Any = fitter_bounds
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        if isinstance(cursor, dict):
            return None
        pair = cursor
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            return None
        leaf = 1
        for dim in shape:
            leaf *= int(dim)
        low_chunks.append(_as_float_vector(pair[0], leaf))
        high_chunks.append(_as_float_vector(pair[1], leaf))
    return np.concatenate(low_chunks), np.concatenate(high_chunks)


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
    ``initial``.
    """
    flat, _spec = flatten_params(initial)
    size = int(flat.size)
    if low is not None or high is not None:
        if low is None or high is None:
            raise ValueError("low and high must be passed together")
        return _as_float_vector(low, size), _as_float_vector(high, size)
    if isinstance(context_bounds, dict) and "low" in context_bounds and "high" in context_bounds:
        return (
            _as_float_vector(context_bounds["low"], size),
            _as_float_vector(context_bounds["high"], size),
        )
    mirrored = bounds_from_fitter(initial, context_bounds, size)
    if mirrored is None:
        mirrored = bounds_from_fitter(initial, fitter_bounds, size)
    if mirrored is None:
        raise ValueError(
            "pass low and high, or bounds that mirror initial_params; "
            "the chain will not invent a box"
        )
    return mirrored


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
    ``initial_params``. Every coordinate the fitter sees lies in the box.
    The fitter is driven through ``evaluate`` / ``step`` or ``ask`` /
    ``tell``, with one step notice per evaluation.

    ``low`` and ``high`` may be vectors or scalars (broadcast). When they
    are omitted, bounds are read from ``benchmark_context["bounds"]`` or
    ``fitter.bounds``. The first exception raised by the fitter is raised
    without calling ``finish``.
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
    x0, spec = flatten_params(initial)
    context_bounds = benchmark_context.get("bounds")
    fitter_bounds = getattr(fitter, "bounds", None)
    box_low, box_high = resolve_bounds(
        initial,
        low=low,
        high=high,
        context_bounds=context_bounds,
        fitter_bounds=fitter_bounds,
    )
    if np.any(box_high <= box_low):
        raise ValueError("each upper bound must be greater than the lower bound")
    x0 = np.minimum(np.maximum(x0, box_low), box_high)

    _init(fitter)
    name = method.lower()
    if name != "portfolio" and preset is None:
        preset = {"boltzmann": Boltzmann(), "fast": Fast(), "gsa": Gsa()}[name]
    steps = max(1, min(int(steps_per_epoch), budget))
    session = _Session(
        lambda flat: unflatten_params(flat, spec), evaluate, step, budget, step_every=1
    )
    best = _drive(session, box_low, box_high, x0, name, int(seed), preset, steps)
    return _finish(fitter, unflatten_params(best, spec))

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
