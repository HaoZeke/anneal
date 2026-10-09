"""ChemFit session driver for gradient-free anneal.

The review harness passes a mapping with ``fitter``, ``budget``, and
``initial_params``. ``run_benchmark`` starts at those parameters, keeps
every trial inside the box, and returns ``fitter.finish(...)``.

The fitter may be the current ChemFit session (``init`` / ``ask`` /
``tell`` / ``finish``) or the review-response names (``evaluate`` /
``step``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
        epochs = max(1, budget // steps)
        history = run(
            obj,
            box_low,
            box_high,
            preset,
            n_epochs=epochs,
            steps_per_epoch=steps,
            seed=int(seed),
            x0=x0,
        )
        best = np.asarray(history.best_pos, dtype=np.float64)
    return _finish(fitter, unflatten_params(best, spec))
