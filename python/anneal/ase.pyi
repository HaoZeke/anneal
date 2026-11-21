# Copyright (c) anneal developers
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy.typing as npt

class Anneal:
    atoms: Any
    seed: int
    budget: int | None
    mode: str
    logfile: Any
    trajectory: Any
    restart: Any
    master: Any
    force_consistent: Any
    append_trajectory: bool
    nsteps: int
    fmax: float
    charged: int

    def __init__(
        self,
        atoms: Any,
        restart: Any = None,
        logfile: Any = "-",
        trajectory: Any = None,
        master: Any = None,
        force_consistent: Any = None,
        append_trajectory: bool = False,
        seed: int = 0,
        budget: int | None = None,
        mode: str = "local",
    ) -> None: ...
    def get_number_of_steps(self) -> int: ...
    def converged(self, forces: npt.ArrayLike | None = None) -> bool: ...
    def log(self, *args: Any, **kwargs: Any) -> None: ...
    def attach(
        self,
        function: Any,
        interval: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def step(self) -> None: ...
    def irun(
        self,
        fmax: float = 0.05,
        steps: int | None = None,
    ) -> Iterator[bool]: ...
    def run(
        self,
        fmax: float = 0.05,
        steps: int | None = None,
    ) -> bool: ...

__all__: list[str]
