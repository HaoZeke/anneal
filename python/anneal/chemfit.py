"""ChemFit bridges for the gradient-free drivers.

Every bridge drives a fitter through ``init``, one loss call per candidate,
and ``finish``. The loss call is ChemFit 4's ``evaluate`` / ``step`` when the
fitter has them and the ``ask`` / ``tell`` of ChemFit 3.1 otherwise. Nested
parameter dictionaries are flattened only at the optimizer
boundary and rebuilt on the way back. Every evaluation stays inside the box,
and the fitter's initial parameters are the start unless a caller passes
another ``x0``. The default driver is the Thompson-allocated portfolio.
"""

from __future__ import annotations

from numbers import Integral
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
_SEED_MODULUS = 2**64


def _whole_number(name: str, value: Any) -> int:
    """``value`` as an int of at least 1, or a ValueError naming ``name``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a whole number, at least 1, got {value!r}")
    return int(value)


def _seed(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < _SEED_MODULUS
    ):
        raise ValueError(f"seed must be a whole number in [0, 2**64), got {value!r}")
    return int(value)


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


def _classical_preset(driver: str, preset: Any, preset_kwargs: dict[str, Any] | None):
    from anneal import Boltzmann, Fast, Gsa

    family = {"boltzmann": Boltzmann, "fast": Fast, "gsa": Gsa}[driver]
    if preset is not None:
        if preset_kwargs:
            raise ValueError("pass preset or preset_kwargs, not both")
        if not isinstance(preset, family):
            raise ValueError(
                f"preset {type(preset).__name__} does not match driver {driver!r}"
            )
        return preset
    return family(**dict(preset_kwargs or {}))


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
    steps_per_epoch: int | None = None,
    preset_kwargs: dict[str, Any] | None = None,
    preset: Any = None,
) -> dict[str, Any]:
    """Fit a ChemFit ``Fitter`` with an anneal gradient-free optimizer.

    Args:
      fitter: ChemFit ``Fitter`` (duck-typed: ``initial_parameters``,
        ``bounds``, ``init``, ``evaluate`` and ``step`` or ``ask`` and
        ``tell``, ``finish``).
      budget: total objective evaluations, the start included. The
        portfolio makes at most ``budget``; the classical drivers make
        exactly ``budget``.
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
      steps_per_epoch: classical-driver epoch width, 100 when omitted; the
        cooling schedule runs ``ceil(budget / steps_per_epoch)`` epochs, the
        last one cut short at ``budget``.
      preset_kwargs: extra kwargs for the preset constructor
        (e.g. ``{"t_init": 5.0}``); classical drivers only.
      preset: an ``anneal.Boltzmann``, ``Fast`` or ``Gsa`` matching a
        classical driver, instead of ``preset_kwargs``. The portfolio takes
        none of ``steps_per_epoch``, ``preset_kwargs`` and ``preset``, and
        raises ``ValueError`` when one is given.

    ``budget`` and ``steps_per_epoch`` must be whole numbers of at least 1 and
    ``seed`` one in ``[0, 2**64)``; ``driver`` is matched exactly. Every value
    is checked before ``fitter.init()``. The search runs on the parameters
    themselves, so a preset's ``sigma`` or ``gamma`` is in their units;
    ``Fitter.fit_anneal`` from the ChemFit patches searches the unit cube,
    where the same preset steps in units of each box width.

    Returns the finished parameter dict (``fitter.finish`` of the best
    position), with array leaves restored to their original shapes.
    """
    from anneal import global_optimize, run

    if driver not in ("portfolio", *_CLASSICAL_DRIVERS):
        raise ValueError(
            f"driver must be 'portfolio' or one of {', '.join(_CLASSICAL_DRIVERS)}; "
            f"got {driver!r}"
        )
    budget = _whole_number("budget", budget)
    seed = _seed(seed)
    if steps_per_epoch is not None:
        steps_per_epoch = _whole_number("steps_per_epoch", steps_per_epoch)
    if driver == "portfolio":
        unused = [
            name
            for name, value in (
                ("steps_per_epoch", steps_per_epoch),
                ("preset_kwargs", preset_kwargs),
                ("preset", preset),
            )
            if value is not None
        ]
        if unused:
            raise ValueError(f"driver 'portfolio' does not take {', '.join(unused)}")
        classical = None
    else:
        classical = _classical_preset(driver, preset, preset_kwargs)

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
    fitter.init()

    def obj(x: np.ndarray) -> float:
        params = unflatten_parameters(np.asarray(x, dtype=np.float64), spec, template)
        loss = _loss(fitter, params)
        _step(fitter)
        return loss

    if driver == "portfolio":
        result = global_optimize(obj, low_vec, high_vec, budget, seed=seed, x0=start_vector)
        best_pos = np.asarray(result["best_pos"], dtype=np.float64)
    else:
        steps = min(100 if steps_per_epoch is None else steps_per_epoch, budget)
        history = run(
            obj,
            low_vec,
            high_vec,
            classical,
            n_epochs=-(-budget // steps),
            steps_per_epoch=steps,
            seed=seed,
            x0=start_vector,
            max_evals=budget,
        )
        best_pos = np.asarray(history.best_pos, dtype=np.float64)

    best_params = unflatten_parameters(best_pos, spec, template)
    return fitter.finish(best_params)

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
            optional ``bounds``. Only ``init``, ``evaluate`` / ``step`` (or
            ``ask`` / ``tell``) and ``finish`` are used, so gradient-free
            drivers never need forces.
        budget: total objective evaluations (one loss call each), the start
            included; the classical chains spend all of it.
        method: ``"portfolio"`` (default; Thompson-allocated SOTA including
            parallel-tempering communicating chains), or ``"boltzmann"``,
            ``"fast"``, ``"gsa"`` for the bound-respecting classical chain
            from the matching initial parameters.
        seed: RNG seed forwarded to the anneal driver.
        default_span: half-width around the initial value for parameters
            ChemFit leaves unbounded.
        tell_every: portfolio evaluations between ``step`` (or ``tell``)
            calls so registered callbacks still fire.
        steps_per_epoch: classical-chain evaluations per epoch; the cooling
            schedule runs ``ceil(budget / steps_per_epoch)`` epochs.
        **preset_kwargs: ``t_init`` / ``sigma`` / ``gamma`` / ``q_v`` /
            ``q_a`` forwarded to the classical preset constructors.

    Returns:
        The dict from ``fitter.finish(best_params)``.
    """
    from anneal import Boltzmann, Fast, Gsa, global_optimize, run

    if int(budget) < 1:
        raise ValueError("budget must be positive")
    vector = ChemFitVector(dict(fitter.initial_parameters))
    if vector.dim == 0:
        raise ValueError("fitter.initial_parameters holds no parameters")
    low, high = chemfit_box(fitter, vector, default_span=default_span)

    fitter.init()
    n_evals = 0

    def ask_vector(x: np.ndarray) -> float:
        nonlocal n_evals
        params = vector.unpack(np.asarray(x, dtype=np.float64))
        loss = _loss(fitter, params)
        n_evals += 1
        if method == "portfolio" and n_evals % max(1, int(tell_every)) == 0:
            _step(fitter)
        elif method != "portfolio" and n_evals % max(1, int(steps_per_epoch)) == 0:
            _step(fitter)
        return loss

    start = np.minimum(np.maximum(vector.x0, low), high)
    if method == "portfolio":
        out = global_optimize(
            ask_vector, low, high, budget=int(budget), seed=int(seed), x0=start
        )
        best = np.asarray(out["best_pos"], dtype=np.float64)
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
        steps = max(1, int(steps_per_epoch))
        history = run(
            ask_vector,
            low,
            high,
            presets[method],
            n_epochs=-(-int(budget) // steps),
            steps_per_epoch=steps,
            seed=int(seed),
            x0=start,
            max_evals=int(budget),
        )
        best = np.asarray(history.best_pos, dtype=np.float64)
    else:
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )

    best_params = vector.unpack(np.asarray(best, dtype=np.float64))
    return fitter.finish(best_params)

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


def _loss(fitter: Any, params: dict[str, Any]) -> float:
    if hasattr(fitter, "evaluate"):
        return float(fitter.evaluate(params))
    if hasattr(fitter, "ask"):
        loss = fitter.ask(params)
        if isinstance(loss, list):
            if len(loss) != 1:
                raise ValueError("ask returned more than one loss for one candidate")
            return float(loss[0])
        return float(loss)
    raise TypeError("fitter needs evaluate or ask")


def _step(fitter: Any) -> None:
    if hasattr(fitter, "step"):
        fitter.step()
    elif hasattr(fitter, "tell"):
        fitter.tell()


def _finish(fitter: Any, params: dict[str, Any]) -> Any:
    try:
        return fitter.finish(params)
    except TypeError:
        return fitter.finish()


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
    The fitter sees at most ``budget`` evaluations, the start included, and
    exactly ``budget`` from a classical chain.

    ``low`` and ``high`` may be vectors or scalars (broadcast). When they
    are omitted, bounds are read from ``benchmark_context["bounds"]`` or
    ``fitter.bounds``.
    """
    from anneal import Boltzmann, Fast, Gsa, global_optimize, run

    fitter = benchmark_context["fitter"]
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

    if hasattr(fitter, "init"):
        fitter.init()

    def obj(flat: np.ndarray) -> float:
        params = unflatten_params(np.asarray(flat, dtype=np.float64), spec)
        loss = _loss(fitter, params)
        _step(fitter)
        return loss

    name = method.lower()
    if name == "portfolio":
        out = global_optimize(obj, box_low, box_high, budget=budget, seed=seed, x0=x0)
        best = np.asarray(out["best_pos"], dtype=np.float64)
    else:
        if preset is None:
            preset = {"boltzmann": Boltzmann(), "fast": Fast(), "gsa": Gsa()}[name]
        steps = max(1, min(int(steps_per_epoch), budget))
        history = run(
            obj,
            box_low,
            box_high,
            preset,
            n_epochs=-(-budget // steps),
            steps_per_epoch=steps,
            seed=int(seed),
            x0=x0,
            max_evals=budget,
        )
        best = np.asarray(history.best_pos, dtype=np.float64)
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

    ``method="global_optimize"`` is the portfolio and ``method="sa"`` the
    Boltzmann preset; other names pass through. A fitter that implements
    ``fit_anneal`` is called with that method name, otherwise :func:`fit_anneal`
    drives it. ``preset`` and every keyword reach whichever path runs; one
    that path does not take raises rather than being dropped. The paths
    measure a preset's step differently: ``Fitter.fit_anneal`` in units of
    each box width, :func:`fit_anneal` in the units of the parameters.
    """
    driver = {"global_optimize": "portfolio", "sa": "boltzmann"}.get(method, method)
    if hasattr(fitter, "fit_anneal"):
        return fitter.fit_anneal(
            budget=budget, method=driver, preset=preset, seed=seed, **kwargs
        )
    return fit_anneal(
        fitter, int(budget), driver=driver, seed=int(seed), preset=preset, **kwargs
    )
