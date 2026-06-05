from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import array_api_compat as _array_api_compat
import numpy as np


@dataclass(frozen=True)
class DeviceHistory:
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

    def gamma(self, shape: tuple[int, ...], concentration: float, scale: float) -> Any:
        if self.library == "torch":
            raise ValueError("Gsa device runs require gamma random sampling")
        return self._finalize(self._rng.gamma(concentration, scale, size=shape))


def _temperature(preset: Any, epoch: int) -> float:
    name = type(preset).__name__
    if name == "Boltzmann":
        return float(preset.t_init) * math.log(2.0) / math.log(epoch + 2.0)
    if name == "Fast":
        return float(preset.t_init) / (epoch + 1.0)
    if name == "Gsa":
        q_v = float(preset.q_v)
        if abs(q_v - 1.0) < 1e-12:
            if epoch == 0:
                return float(preset.t_init)
            return float(preset.t_init) * math.log(2.0) / math.log(epoch + 1.0)
        if epoch == 0:
            return float(preset.t_init)
        exp = q_v - 1.0
        numerator = (2.0**exp) - 1.0
        denominator = ((epoch + 1.0) ** exp) - 1.0
        return float(preset.t_init) * numerator / denominator
    raise TypeError("preset must be Boltzmann, Fast, or Gsa")


def _proposal(current: Any, temp: float, preset: Any, random: _Random, xp: Any) -> Any:
    name = type(preset).__name__
    if name == "Boltzmann":
        return current + float(preset.sigma) * random.normal(current.shape)
    if name == "Fast":
        uniform = random.uniform(current.shape)
        return current + float(preset.gamma) * xp.tan(math.pi * (uniform - 0.5))
    if name == "Gsa":
        q_v = float(preset.q_v)
        dof = (3.0 - q_v) / (q_v - 1.0)
        normal = random.normal(current.shape)
        gamma = random.gamma(current.shape, dof / 2.0, 2.0 / dof)
        scale = temp ** (1.0 / (3.0 - q_v))
        return current + scale * normal / xp.sqrt(gamma)
    raise TypeError("preset must be Boltzmann, Fast, or Gsa")


def _acceptance_probability(
    delta: Any,
    *,
    temp: float,
    preset: Any,
    xp: Any,
    device: Any,
    dtype: Any,
) -> Any:
    zero = _asarray(0.0, xp=xp, device=device, dtype=dtype)
    one = _asarray(1.0, xp=xp, device=device, dtype=dtype)
    temp_array = _asarray(temp, xp=xp, device=device, dtype=dtype)
    if type(preset).__name__ == "Gsa" and abs(float(preset.q_a) - 1.0) >= 1e-12:
        q_a = float(preset.q_a)
        base = one + (q_a - 1.0) * delta / temp_array
        uphill = xp.where(base <= zero, zero, base ** (1.0 / (1.0 - q_a)))
    else:
        uphill = xp.exp(-delta / temp_array)
    return xp.where(delta <= zero, one, uphill)


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
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")

    xp = _array_namespace(low)
    device = _device(low)
    dtype = getattr(low, "dtype", None)
    low_array = _asarray(low, xp=xp, device=device, dtype=dtype)
    high_array = _asarray(high, xp=xp, device=device, dtype=dtype)
    if low_array.shape != high_array.shape:
        raise ValueError("low and high must have the same shape")
    if len(low_array.shape) != 1:
        raise ValueError("low and high must be one-dimensional arrays")

    random = _Random(low_array, xp=xp, device=device, dtype=dtype, seed=seed)
    if start is None:
        current = low_array + random.uniform(low_array.shape) * (high_array - low_array)
    else:
        current = _asarray(start, xp=xp, device=device, dtype=dtype)
    if current.shape != low_array.shape:
        raise ValueError("start must have the same shape as low and high")
    current = xp.minimum(xp.maximum(current, low_array), high_array)
    current_val = _objective_value(obj_fn(current), xp=xp, device=device, dtype=dtype)
    best_pos = current + _asarray(0.0, xp=xp, device=device, dtype=dtype)
    best_val = current_val + _asarray(0.0, xp=xp, device=device, dtype=dtype)

    epochs = []
    temps = []
    accepted_history = []
    rejected_history = []
    best_vals = []
    int_dtype = getattr(xp, "int64", None)

    for epoch in range(n_epochs):
        temp = _temperature(preset, epoch)
        accepted_epoch = _asarray(0, xp=xp, device=device, dtype=int_dtype)
        rejected_epoch = _asarray(0, xp=xp, device=device, dtype=int_dtype)

        for _ in range(steps_per_epoch):
            candidate = _proposal(current, temp, preset, random, xp)
            candidate = xp.minimum(xp.maximum(candidate, low_array), high_array)
            candidate_val = _objective_value(obj_fn(candidate), xp=xp, device=device, dtype=dtype)
            delta = candidate_val - current_val
            probability = _acceptance_probability(
                delta,
                temp=temp,
                preset=preset,
                xp=xp,
                device=device,
                dtype=dtype,
            )
            accepted = _asarray(random.uniform(()) < probability, xp=xp, device=device)
            accepted_count = _count_from_bool(accepted, xp=xp, device=device)
            rejected_count = _count_from_bool(xp.logical_not(accepted), xp=xp, device=device)
            accepted_epoch = accepted_epoch + accepted_count
            rejected_epoch = rejected_epoch + rejected_count
            current = xp.where(accepted, candidate, current)
            current_val = xp.where(accepted, candidate_val, current_val)
            improved = _asarray(current_val < best_val, xp=xp, device=device)
            best_pos = xp.where(improved, current, best_pos)
            best_val = xp.where(improved, current_val, best_val)

        epochs.append(_asarray(epoch, xp=xp, device=device, dtype=int_dtype))
        temps.append(_asarray(temp, xp=xp, device=device, dtype=dtype))
        accepted_history.append(accepted_epoch)
        rejected_history.append(rejected_epoch)
        best_vals.append(best_val)

    return DeviceHistory(
        epochs=xp.stack(epochs),
        temps=xp.stack(temps),
        accepted=xp.stack(accepted_history),
        rejected=xp.stack(rejected_history),
        best_vals=xp.stack(best_vals),
        best_pos=best_pos,
        best_val=best_val,
        current_pos=current,
        current_val=current_val,
        namespace=xp,
        device=device,
    )


@dataclass(frozen=True)
class EnsembleHistory:
    """Result of a batched ensemble SA run over ``n_chains`` parallel chains.

    ``best_pos``/``best_val`` are per-chain (shape ``(n_chains, dim)`` and
    ``(n_chains,)``); ``global_best_pos``/``global_best_val`` reduce over the
    ensemble. All fields stay on the device and in the namespace inferred from
    the bounds.
    """

    best_pos: Any
    best_val: Any
    global_best_pos: Any
    global_best_val: Any
    accepted: Any
    rejected: Any
    namespace: Any
    device: Any


def _ensemble_objective_value(value: Any, n_chains: int, *, xp: Any, device: Any, dtype: Any) -> Any:
    if isinstance(value, bool | int | float | complex):
        raise ValueError("device objectives must return an Array API array")
    if not (_array_api_compat.is_array_api_obj(value) or hasattr(value, "__dlpack__")
            or isinstance(value, np.ndarray)):
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
    if n_chains <= 0:
        raise ValueError("n_chains must be positive")
    if n_epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("n_epochs and steps_per_epoch must be positive")

    xp = _array_namespace(low)
    device = _device(low)
    dtype = getattr(low, "dtype", None)
    low_array = _asarray(low, xp=xp, device=device, dtype=dtype)
    high_array = _asarray(high, xp=xp, device=device, dtype=dtype)
    if low_array.shape != high_array.shape or len(low_array.shape) != 1:
        raise ValueError("low and high must be one-dimensional arrays of equal shape")
    dim = low_array.shape[0]
    shape = (n_chains, dim)

    random = _Random(low_array, xp=xp, device=device, dtype=dtype, seed=seed)
    span = high_array - low_array
    current = low_array + random.uniform(shape) * span
    current = xp.minimum(xp.maximum(current, low_array), high_array)
    current_val = _ensemble_objective_value(obj_fn(current), n_chains, xp=xp, device=device, dtype=dtype)
    best_pos = current + _asarray(0.0, xp=xp, device=device, dtype=dtype)
    best_val = current_val + _asarray(0.0, xp=xp, device=device, dtype=dtype)

    int_dtype = getattr(xp, "int64", None)
    accepted_total = _asarray(0, xp=xp, device=device, dtype=int_dtype)
    rejected_total = _asarray(0, xp=xp, device=device, dtype=int_dtype)

    for epoch in range(n_epochs):
        temp = _temperature(preset, epoch)
        for _ in range(steps_per_epoch):
            candidate = _proposal(current, temp, preset, random, xp)
            candidate = xp.minimum(xp.maximum(candidate, low_array), high_array)
            candidate_val = _ensemble_objective_value(
                obj_fn(candidate), n_chains, xp=xp, device=device, dtype=dtype
            )
            delta = candidate_val - current_val
            probability = _acceptance_probability(
                delta, temp=temp, preset=preset, xp=xp, device=device, dtype=dtype
            )
            accepted = _asarray(random.uniform((n_chains,)) < probability, xp=xp, device=device)
            accepted_col = accepted[:, None]
            accepted_total = accepted_total + xp.sum(_to_dtype(accepted, int_dtype))
            rejected_total = rejected_total + xp.sum(_to_dtype(xp.logical_not(accepted), int_dtype))
            current = xp.where(accepted_col, candidate, current)
            current_val = xp.where(accepted, candidate_val, current_val)
            improved = current_val < best_val
            best_pos = xp.where(improved[:, None], current, best_pos)
            best_val = xp.where(improved, current_val, best_val)

    g = int(xp.argmin(best_val))
    return EnsembleHistory(
        best_pos=best_pos,
        best_val=best_val,
        global_best_pos=best_pos[g],
        global_best_val=best_val[g],
        accepted=accepted_total,
        rejected=rejected_total,
        namespace=xp,
        device=device,
    )


__all__ = ["DeviceHistory", "EnsembleHistory", "run_device", "run_ensemble"]
