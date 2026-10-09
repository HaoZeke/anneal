"""ChemFit integration: drive anneal's gradient-free optimizers from a ChemFit ``Fitter``.

This module speaks the ``Fitter`` user-driven protocol exactly as ChemFit
defines it — :meth:`init`, :meth:`ask`, :meth:`tell`, :meth:`finish` — and
maps ChemFit's nested parameter dictionaries onto the flat vectors anneal
optimizes. There is no ChemFit import here and no ChemFit dependency: the
fitter is duck-typed, so this works against any object exposing
``initial_parameters``, ``bounds``, ``init``, ``ask``, ``tell`` and
``finish``.

Bounds handling
---------------
Every anneal evaluation point lies inside ``[low, high]``: the classical
drivers mirror-reflect proposals into the box and the portfolio reflects
before evaluation. ChemFit therefore only ever sees in-bounds candidates,
which is what its own bounds machinery assumes.

Warm starts
-----------
ChemFit's ``initial_params`` become the optimization start by default
(``x0``): the classical chains start there and the portfolio evaluates the
start once up front, installing it as the incumbent the arms improve on.

Typical use (structural review response with per-atom positions)::

    from anneal.chemfit import fit_anneal

    result = fit_anneal(
        fitter,
        budget=2000,
        driver="portfolio",  # SOTA default; "boltzmann"/"fast"/"gsa" for ablation
        low=-3.0 * np.ones(3 * n_atoms),
        high=3.0 * np.ones(3 * n_atoms),
        seed=0,
    )
    # result is the finished parameter dict, ready to report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from anneal import Boltzmann, Fast, Gsa, global_optimize, run

__all__ = ["fit_anneal", "flatten_parameters", "unflatten_parameters"]

_CLASSICAL_DRIVERS = ("boltzmann", "fast", "gsa")


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


def _classical_preset(driver: str, preset_kwargs: dict[str, Any] | None):
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
      fitter: ChemFit ``Fitter`` (duck-typed: ``initial_parameters``,
        ``bounds``, ``init``, ``ask``, ``tell``, ``finish``).
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

    Returns the finished parameter dict (``fitter.finish`` of the best
    position), with array leaves restored to their original shapes.
    """
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

    # The fitter owns bookkeeping; every ask is one optimizer step.
    fitter.init()

    def obj(x: np.ndarray) -> float:
        params = unflatten_parameters(np.asarray(x, dtype=np.float64), spec, template)
        loss = fitter.ask(params)
        fitter.tell()
        return float(loss)

    if driver == "portfolio":
        result = global_optimize(obj, low_vec, high_vec, budget, seed=int(seed), x0=start_vector)
        best_pos = np.asarray(result["best_pos"], dtype=np.float64)
    else:
        preset = _classical_preset(driver, preset_kwargs)
        steps = max(1, min(int(steps_per_epoch), budget))
        epochs = max(1, budget // steps)
        history = run(
            obj,
            low_vec,
            high_vec,
            preset,
            n_epochs=epochs,
            steps_per_epoch=steps,
            seed=int(seed),
            x0=start_vector,
        )
        best_pos = np.asarray(history.best_pos, dtype=np.float64)

    best_params = unflatten_parameters(best_pos, spec, template)
    return fitter.finish(best_params)
