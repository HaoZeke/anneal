# Copyright (c) anneal developers
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
    n_evals: int = 0
    evaluated_points: int = 0

    @property
    def total_accepted(self) -> Any: ...
    @property
    def total_rejected(self) -> Any: ...

@dataclass(frozen=True)
class EnsembleHistory:
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
) -> DeviceHistory: ...
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
) -> EnsembleHistory: ...

__all__ = [
    "DeviceHistory",
    "EnsembleHistory",
    "run_device",
    "run_ensemble",
]
