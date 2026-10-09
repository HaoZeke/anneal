"""ChemFit integration: gradient-free fitting over ChemFit's ask/tell/finish protocol.

The reported bug used two ChemFit calls that do not exist
(``fitter.evaluate`` / ``fitter.step``), dropped the only reshape that
mattered (``positions.reshape(...)`` without assignment is a no-op), passed
no initial parameters to the chain, and ran the unconstrained classical
presets, so nothing kept proposals inside ``[low, high]``. This module is
the corrected driver:

- initial parameters come from ``fitter.initial_parameters`` and reach the
  chain as ``x0`` (flattened, clipped into the box);
- bounds come from ``fitter.bounds`` with the same structure, falling back
  to ``init +/- default_span`` where ChemFit leaves a parameter unbounded;
- every anneal evaluation goes through ``fitter.ask`` (one loss per
  candidate dict) with periodic ``fitter.tell()`` for callbacks, and the
  run closes with ``fitter.finish(best_params)``;
- the default driver is the SOTA gradient-free portfolio
  (:func:`anneal.global_optimize`, which allocates across QMC restarts,
  basin hopping, differential evolution, GSA, parallel-tempering
  communicating chains, and trust-region polls by Thompson sampling),
  with the bound-respecting classical presets available as
  ``method="boltzmann"`` / ``"fast"`` / ``"gsa"`` for ablations.

Typical review-response shape::

    import numpy as np
    from anneal.chemfit import fit_chemfit

    def run_anneal(benchmark_context):
        fitter = benchmark_context["fitter"]
        return fit_chemfit(fitter, budget=benchmark_context["budget"], seed=0)
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


def _bound_pair(leaf: Any) -> tuple[Any, Any]:
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
        lower, upper = _bound_pair(flat_bounds.get(base))
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
            optional ``bounds``. Only the ask/tell/finish protocol is used,
            so gradient-free drivers never need forces.
        budget: total objective evaluations (one ``ask`` each).
        method: ``"portfolio"`` (default; Thompson-allocated SOTA including
            parallel-tempering communicating chains), or ``"boltzmann"``,
            ``"fast"``, ``"gsa"`` for the bound-respecting classical chain
            from the matching initial parameters.
        seed: RNG seed forwarded to the anneal driver.
        default_span: half-width around the initial value for parameters
            ChemFit leaves unbounded.
        tell_every: portfolio evaluations between ``fitter.tell()`` calls
            so registered callbacks still fire.
        steps_per_epoch: classical-chain evaluations per epoch; epochs are
            derived as ``budget // steps_per_epoch``.
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
        loss = fitter.ask(params)
        if isinstance(loss, list):
            if len(loss) != 1:
                raise ValueError("expected one loss per candidate")
            loss = loss[0]
        n_evals += 1
        if method == "portfolio" and n_evals % max(1, int(tell_every)) == 0:
            fitter.tell()
        elif method != "portfolio" and n_evals % max(1, int(steps_per_epoch)) == 0:
            fitter.tell()
        return float(loss)

    if method == "portfolio":
        out = global_optimize(
            ask_vector, low, high, budget=int(budget), seed=int(seed)
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
        n_epochs = max(1, int(budget) // max(1, int(steps_per_epoch)))
        history = run(
            ask_vector,
            low,
            high,
            presets[method],
            n_epochs=n_epochs,
            steps_per_epoch=max(1, int(steps_per_epoch)),
            seed=int(seed),
            x0=np.asarray(vector.x0, dtype=np.float64),
        )
        best = np.asarray(history.best_pos, dtype=np.float64)
    else:
        raise ValueError(
            f"unknown method {method!r}: expected 'portfolio', 'boltzmann', 'fast', or 'gsa'"
        )

    best_params = vector.unpack(np.asarray(best, dtype=np.float64))
    return fitter.finish(best_params)


__all__ = ["ChemFitVector", "chemfit_box", "fit_chemfit"]
