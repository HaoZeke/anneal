"""ASE calculators for rgpot.

Importing this package imports ASE. ``import rgpot`` does not.
"""

from __future__ import annotations

from typing import Any

from rgpot.ase.anneal import AnnealOptimizer, AnnealRunner
from rgpot.ase.calculator import RgpotCalculator, available_kinds

__all__ = [
    "AnnealOptimizer",
    "AnnealRunner",
    "RgpotCalculator",
    "RgpotRpcCalculator",
    "available_kinds",
]


def __getattr__(name: str) -> Any:
    if name == "RgpotRpcCalculator":
        from rgpot.ase.rpc import RgpotRpcCalculator

        return RgpotRpcCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
