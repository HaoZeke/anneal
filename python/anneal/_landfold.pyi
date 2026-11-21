# Copyright (c) anneal developers
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy.typing as npt

from anneal._core import LandfoldMap as LandfoldMap

def landfold(rows: npt.ArrayLike, **kwargs: Any) -> dict[str, Any]: ...
def landfold_book(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    **kwargs: Any,
) -> dict[str, Any]: ...

__all__ = ["LandfoldMap", "landfold", "landfold_book"]
