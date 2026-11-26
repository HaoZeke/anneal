# Copyright (c) anneal developers
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_ParameterSpec = list[tuple[tuple[str, ...], tuple[int, ...]]]

class ChemFitVector:
    x0: npt.NDArray[np.float64]
    keys: list[str]
    shapes: dict[str, tuple[int, ...]]
    def __init__(self, template: dict[str, Any]) -> None: ...
    @property
    def dim(self) -> int: ...
    def pack(self, params: dict[str, Any]) -> npt.NDArray[np.float64]: ...
    def unpack(self, vector: npt.NDArray[np.float64]) -> dict[str, Any]: ...

def flatten_parameters(
    params: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], _ParameterSpec]: ...
def spec_total(spec: _ParameterSpec) -> int: ...
def unflatten_parameters(
    vector: npt.NDArray[np.float64],
    spec: _ParameterSpec,
    template: dict[str, Any],
) -> dict[str, Any]: ...
def fit_anneal(
    fitter: Any,
    budget: int,
    *,
    driver: str = "portfolio",
    seed: int = 0,
    x0: dict[str, Any] | npt.NDArray[np.float64] | None = None,
    low: npt.NDArray[np.float64] | None = None,
    high: npt.NDArray[np.float64] | None = None,
    bound_span: float = 3.0,
    steps_per_epoch: int = 100,
    preset_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]: ...
def chemfit_box(
    fitter: Any,
    vector: ChemFitVector,
    default_span: float = 3.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def fit_chemfit(
    fitter: Any,
    budget: int,
    method: str = "portfolio",
    seed: int = 0,
    default_span: float = 3.0,
    tell_every: int = 50,
    steps_per_epoch: int = 100,
    **preset_kwargs: Any,
) -> dict[str, Any]: ...
def flatten_params(
    params: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], _ParameterSpec]: ...
def unflatten_params(
    vector: npt.NDArray[np.float64],
    spec: _ParameterSpec,
) -> dict[str, Any]: ...
def bounds_from_fitter(
    initial: dict[str, Any],
    fitter_bounds: Any,
    size: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None: ...
def resolve_bounds(
    initial: dict[str, Any],
    *,
    low: Any = None,
    high: Any = None,
    context_bounds: Any = None,
    fitter_bounds: Any = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def run_benchmark(
    benchmark_context: dict[str, Any],
    *,
    method: str = "portfolio",
    seed: int = 0,
    steps_per_epoch: int = 100,
    low: Any = None,
    high: Any = None,
    preset: Any = None,
) -> Any: ...
def run_fitter(
    fitter: Any,
    budget: int,
    method: str = "global_optimize",
    preset: Any = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]: ...

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
