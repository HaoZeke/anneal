"""Compatibility helpers for ASE's external calculator registry."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import (
    external_calculators,
    register_calculator_class,
)
from ase.calculators.names import names


def register_calculator(name: str, calculator: type[Any]) -> None:
    """Register *calculator* under *name*, replacing a stale registration."""
    if external_calculators.get(name) is calculator:
        return
    if name in external_calculators or name in names:
        external_calculators[name] = calculator
        return
    register_calculator_class(name, calculator)
