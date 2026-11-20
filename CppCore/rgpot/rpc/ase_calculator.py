"""Deprecated import path for the Cap'n Proto ASE calculator.

Use :class:`rgpot.ase.RgpotRpcCalculator`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "ase_calculator.RgpotCalculator is deprecated; "
    "use rgpot.ase.RgpotRpcCalculator",
    DeprecationWarning,
    stacklevel=2,
)

from rgpot.ase.rpc import RgpotRpcCalculator as RgpotCalculator  # noqa: E402

__all__ = ["RgpotCalculator"]
