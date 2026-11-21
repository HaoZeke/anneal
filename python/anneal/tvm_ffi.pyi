# Copyright (c) anneal developers
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anneal.device import DeviceHistory

@dataclass(frozen=True)
class TvmFfiTensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    device: Any
    dlpack_device: tuple[int, int] | None

def tvm_ffi_tensor(array: Any, *, tvm_ffi: Any | None = None) -> Any: ...
def tvm_ffi_tensor_metadata(array: Any) -> TvmFfiTensorMetadata: ...
def tvm_ffi_tensors_from_history(
    history: DeviceHistory,
    *,
    tvm_ffi: Any | None = None,
) -> dict[str, Any]: ...

__all__: list[str]
