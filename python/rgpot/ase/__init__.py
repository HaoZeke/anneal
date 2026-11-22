"""ASE calculators for rgpot.

Importing this package imports ASE. ``import rgpot`` does not.
"""

from __future__ import annotations

from typing import Any

from rgpot.ase.anneal import AnnealOptimizer, AnnealRunner
from rgpot.ase.calculator import RgpotCalculator, available_kinds
from rgpot.ase.rpc import RgpotRpcCalculator

__all__ = [
    "AnnealOptimizer",
    "AnnealRunner",
    "RgpotCalculator",
    "RgpotRpcCalculator",
    "available_kinds",
]


def _make_ase_plugins() -> set[Any]:
    """Build ASE 3.29 calculator plugins without importing its API eagerly."""
    try:
        from ase._4.plugins.plugin import CalculatorPlugin  # pyright: ignore[reportMissingImports]  # ASE<3.29
    except ImportError:
        return set()
    citation = "https://github.com/OmniPotentRPC/rgpot"
    return {
        CalculatorPlugin(
            name="rgpot",
            citation=citation,
            implementation="rgpot.ase.calculator.RgpotCalculator",
        ),
        CalculatorPlugin(
            name="rgpot_rpc",
            citation=citation,
            implementation="rgpot.ase.rpc.RgpotRpcCalculator",
        ),
    }


def __getattr__(name: str) -> Any:
    if name == "__ase_plugins__":
        plugins = _make_ase_plugins()
        # The import above re-enters through ASE's entry-point scan and
        # stores the plugins it bound to this module. Keep that set.
        # setdefault also makes the first store win across threads.
        return globals().setdefault(name, plugins)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
