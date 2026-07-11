"""Benchmark problem catalog.

Bundles the four built-in objectives (StybTang, Rosenbrock, Rastrigin,
Ackley) at multiple dimensions plus a handful of additional global-
optimisation textbook functions (Schwefel, Levy, Griewank). Designed
so the same registry powers (a) the Dolan-Moré / Moré-Wild profiles
in plots/, (b) the IISE Section 6 benchmarking table.

The deeper CUTEst / GALAHAD integration via pycutest lives in
`cutest_runner.py` and requires the verify profile plus system deps
(gfortran + sifdecode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Problem:
    """A benchmark problem: name, dimension, callable, box bounds, known optimum."""

    name: str
    dim: int
    fn: Callable[[np.ndarray], float]
    low: np.ndarray
    high: np.ndarray
    f_star: float


def _styb_tang(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(0.5 * np.sum(x**4 - 16 * x**2 + 5 * x))


def _rosenbrock(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _rastrigin(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def _ackley(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.e)


def _schwefel(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def _levy(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    middle = np.sum(
        (w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2)
    )
    return float(term1 + middle + term3)


def _griewank(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    s = np.sum(x**2) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return float(1.0 + s - p)


def _box(value: float, dim: int) -> np.ndarray:
    return np.full(dim, value, dtype=np.float64)


CATALOG: dict[str, Problem] = {}


def _register(name, dim, fn, low_v, high_v, f_star):
    CATALOG[name] = Problem(
        name=name,
        dim=dim,
        fn=fn,
        low=_box(low_v, dim),
        high=_box(high_v, dim),
        f_star=f_star,
    )


# Styblinski-Tang: f* = -39.16599 * D
for d in (2, 5, 10):
    _register(f"styb_tang_{d}d", d, _styb_tang, -5.0, 5.0, -39.16599 * d)

# Rosenbrock: f* = 0 at x = (1, 1, ..., 1)
for d in (2, 5, 10):
    _register(f"rosenbrock_{d}d", d, _rosenbrock, -2.048, 2.048, 0.0)

# Rastrigin: f* = 0 at origin
for d in (2, 5, 10):
    _register(f"rastrigin_{d}d", d, _rastrigin, -5.12, 5.12, 0.0)

# Ackley: f* = 0 at origin
for d in (2, 5, 10):
    _register(f"ackley_{d}d", d, _ackley, -32.768, 32.768, 0.0)

# Schwefel: f* = 0 at x_i = 420.9687
for d in (2, 5):
    _register(f"schwefel_{d}d", d, _schwefel, -500.0, 500.0, 0.0)

# Levy: f* = 0 at x = (1, 1, ..., 1)
for d in (2, 5):
    _register(f"levy_{d}d", d, _levy, -10.0, 10.0, 0.0)

# Griewank: f* = 0 at origin
for d in (2, 5):
    _register(f"griewank_{d}d", d, _griewank, -600.0, 600.0, 0.0)


def list_problems() -> list[str]:
    """All problem ids in the catalog."""
    return sorted(CATALOG.keys())


def get_problem(name: str) -> Problem:
    return CATALOG[name]
