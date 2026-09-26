from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import array_api_compat as _array_api_compat

from anneal.device import DeviceHistory


_HISTORY_TENSOR_FIELDS = (
    "epochs",
    "temps",
    "accepted",
    "rejected",
    "best_vals",
    "best_pos",
    "best_val",
    "current_pos",
    "current_val",
)


@dataclass(frozen=True)
class TvmFfiTensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: Any
    dlpack_device: tuple[int, int] | None


def _require_dlpack(array: Any) -> None:
    if not hasattr(array, "__dlpack__"):
        raise TypeError("array does not implement __dlpack__")


def _array_device(array: Any) -> Any:
    try:
        return _array_api_compat.device(array)
    except (AttributeError, TypeError, ValueError):
        return getattr(array, "device", None)


def _dlpack_device(array: Any) -> tuple[int, int] | None:
    device = getattr(array, "__dlpack_device__", None)
    if device is None:
        return None
    return device()


def _tvm_ffi_module(tvm_ffi: Any | None) -> Any:
    if tvm_ffi is not None:
        return tvm_ffi
    try:
        return import_module("tvm_ffi")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tvm_ffi is optional; install tvm-ffi or pass a module-like object "
            "with from_dlpack"
        ) from exc


def tvm_ffi_tensor(array: Any, *, tvm_ffi: Any | None = None) -> Any:
    _require_dlpack(array)
    module = _tvm_ffi_module(tvm_ffi)
    from_dlpack = getattr(module, "from_dlpack", None)
    if from_dlpack is None:
        raise TypeError("tvm_ffi object must provide from_dlpack")
    return from_dlpack(array)


def tvm_ffi_tensor_metadata(array: Any) -> TvmFfiTensorMetadata:
    _require_dlpack(array)
    return TvmFfiTensorMetadata(
        shape=tuple(int(dim) for dim in getattr(array, "shape", ())),
        dtype=str(getattr(array, "dtype", "")),
        device=_array_device(array),
        dlpack_device=_dlpack_device(array),
    )


def tvm_ffi_tensors_from_history(
    history: DeviceHistory, *, tvm_ffi: Any | None = None
) -> dict[str, Any]:
    return {
        field: tvm_ffi_tensor(getattr(history, field), tvm_ffi=tvm_ffi)
        for field in _HISTORY_TENSOR_FIELDS
    }


__all__ = [
    "TvmFfiTensorMetadata",
    "tvm_ffi_tensor",
    "tvm_ffi_tensor_metadata",
    "tvm_ffi_tensors_from_history",
]
