from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import array_api_compat as _array_api_compat
import numpy as np


@dataclass(frozen=True)
class DeviceHistory:
    """Device-resident history with callback and evaluated-point counts."""

    epochs: Any
    temps: Any
    accepted: Any
    rejected: Any
    best_vals: Any
    best_pos: Any
    best_val: Any
    current_pos: Any
    current_val: Any
    namespace: Any
    device: Any
    n_evals: int = 0
    evaluated_points: int = 0

    @property
    def total_accepted(self) -> Any:
        return _asarray(
            self.namespace.sum(self.accepted),
            xp=self.namespace,
            device=self.device,
            dtype=getattr(self.accepted, "dtype", None),
        )

    @property
    def total_rejected(self) -> Any:
        return _asarray(
            self.namespace.sum(self.rejected),
            xp=self.namespace,
            device=self.device,
            dtype=getattr(self.rejected, "dtype", None),
        )


def _array_namespace(*arrays: Any) -> Any:
    values = tuple(array for array in arrays if array is not None)
    if not values:
        raise ValueError("At least one array is required")
    try:
        return _array_api_compat.array_namespace(*values, use_compat=True)
    except ValueError:
        return _array_api_compat.array_namespace(*values)


def _device(array: Any) -> Any:
    return _array_api_compat.device(array)


def _asarray(
    value: Any,
    *,
    xp: Any,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        array = xp.asarray(value, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        array = xp.asarray(value, **kwargs)
    if device is not None and _device(array) != device:
        array = _array_api_compat.to_device(array, device)
    return array


def _to_dtype(array: Any, dtype: Any | None) -> Any:
    if dtype is None or getattr(array, "dtype", None) == dtype:
        return array
    astype = getattr(array, "astype", None)
    if astype is not None:
        return astype(dtype)
    to = getattr(array, "to", None)
    if to is not None:
        return to(dtype=dtype)
    return array


def _library_name(array: Any) -> str:
    module = type(array).__module__
    if module.startswith("cupy"):
        return "cupy"
    if module.startswith("torch"):
        return "torch"
    return "numpy"


class _Random:
    def __init__(self, reference: Any, *, xp: Any, device: Any, dtype: Any, seed: int):
        self.xp = xp
        self.device = device
        self.dtype = dtype
        self.library = _library_name(reference)
        if self.library == "cupy":
            import cupy

            self._cupy = cupy
            self._rng = cupy.random.default_rng(seed)
        elif self.library == "torch":
            import torch

            self._torch = torch
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(seed)
        else:
            self._rng = np.random.default_rng(seed)

    def _finalize(self, array: Any) -> Any:
        array = _asarray(array, xp=self.xp, device=self.device)
        return _to_dtype(array, self.dtype)

    def uniform(self, shape: tuple[int, ...]) -> Any:
        if self.library == "cupy":
            return self._rng.random(size=shape, dtype=self.dtype)
        if self.library == "torch":
            return self._torch.rand(
                shape,
                generator=self._rng,
                device=self.device,
                dtype=self.dtype,
            )
        return self._finalize(self._rng.random(shape))

    def normal(self, shape: tuple[int, ...]) -> Any:
        if self.library == "cupy":
            return self._rng.standard_normal(size=shape, dtype=self.dtype)
        if self.library == "torch":
            return self._torch.randn(
                shape,
                generator=self._rng,
                device=self.device,
                dtype=self.dtype,
            )
        return self._finalize(self._rng.standard_normal(shape))


def _count_from_bool(value: Any, *, xp: Any, device: Any) -> Any:
    boolean = _asarray(value, xp=xp, device=device)
    return _to_dtype(boolean, getattr(xp, "int64", None))


def _objective_value(value: Any, *, xp: Any, device: Any, dtype: Any) -> Any:
    if isinstance(value, np.generic):
        array = _asarray(value, xp=xp, device=device, dtype=dtype)
    elif isinstance(value, bool | int | float | complex):
        raise ValueError("device objectives must return an Array API array")
    elif _array_api_compat.is_array_api_obj(value) or hasattr(value, "__dlpack__"):
        array = _asarray(value, xp=xp, device=device, dtype=dtype)
    else:
        raise ValueError("device objectives must return an Array API array")
    if array.shape != ():
        raise ValueError("device objectives must return a scalar Array API array")
    return array


def run_device(
    obj_fn: Callable[[Any], Any],
    low: Any,
    high: Any,
    preset: Any,
    *,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    start: Any | None = None,
) -> DeviceHistory:
    """Run the native preset controller with device-resident array operations."""
    from . import _core

    return _core._run_device(
        obj_fn,
        low,
        high,
        preset,
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        start=start,
    )


@dataclass(frozen=True)
class EnsembleHistory:
    """Result of a batched ensemble SA run over ``n_chains`` parallel chains.

    ``best_pos``/``best_val`` are per-chain (shape ``(n_chains, dim)`` and
    ``(n_chains,)``); ``global_best_pos``/``global_best_val`` reduce over the
    ensemble. Array fields stay on the device and in the namespace inferred
    from the bounds. ``n_evals`` counts batched objective calls;
    ``evaluated_points`` counts all points in those batches.
    """

    best_pos: Any
    best_val: Any
    global_best_pos: Any
    global_best_val: Any
    accepted: Any
    rejected: Any
    namespace: Any
    device: Any
    n_evals: int = 0
    evaluated_points: int = 0


def _ensemble_objective_value(
    value: Any, n_chains: int, *, xp: Any, device: Any, dtype: Any
) -> Any:
    if isinstance(value, bool | int | float | complex):
        raise ValueError("device objectives must return an Array API array")
    if not (
        _array_api_compat.is_array_api_obj(value)
        or hasattr(value, "__dlpack__")
        or isinstance(value, np.ndarray)
    ):
        raise ValueError("device objectives must return an Array API array")
    array = _asarray(value, xp=xp, device=device, dtype=dtype)
    if array.shape != (n_chains,):
        raise ValueError(
            f"batched device objectives must return shape ({n_chains},), got {array.shape}"
        )
    return array


def run_ensemble(
    obj_fn: Callable[[Any], Any],
    low: Any,
    high: Any,
    preset: Any,
    *,
    n_chains: int,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
) -> EnsembleHistory:
    """Run ``n_chains`` independent SA chains as one batched device kernel.

    The state is ``(n_chains, dim)`` and ``obj_fn`` is called on the whole
    batch, returning ``(n_chains,)``; every proposal, acceptance, and update is
    vectorized over the ensemble. With a CuPy-backed ``low`` the ensemble runs
    resident on the GPU, which is where batching over chains pays off. The
    transition-kernel decomposition is the same as the single-chain
    :func:`run_device`; only the leading batch axis is added.
    """
    from . import _core

    return _core._run_device_ensemble(
        obj_fn,
        low,
        high,
        preset,
        n_chains=n_chains,
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
    )


__all__ = ["DeviceHistory", "EnsembleHistory", "run_device", "run_ensemble"]
