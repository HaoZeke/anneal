"""Optimized Stochastic Annealing acceptance under noise.

Implements the sequential acceptance rule of Ball, Branke, and Meisel,
"Optimal Sampling for Simulated Annealing under Noise," INFORMS Journal on
Computing 30(1):200-215, 2018 (doi:10.1287/ijoc.2017.0774).

The objective difference is observed only through noisy samples
``delta_i ~ Normal(Delta, sigma^2)`` with known ``sigma``. For each proposed
move the rule accumulates ``c_n = c_{n-1} + delta_n`` and, at every draw, makes
a three-way decision (accept, reject, or take another sample), stopping at the
first accept or reject. Their universally optimal per-step acceptance rule (Eq.
19) is

    A(c_n, c_{n-1}) = min(1, exp(-2 (c_n + beta sigma^2 / 2)
                                   (c_{n-1} + beta sigma^2 / 2) / sigma^2)),

with the simple optimal rejection threshold c* = 0. The whole procedure obeys
detailed balance at each step while maximizing the acceptance probability per
sample, so it is the principled acceptance rule when Delta is known only up to
noise. In the typed algebra it is an ``Accept`` component that consumes a noisy
Delta estimator rather than an exact value, which is exactly the regime the
finite-precision audit of the manuscript describes: the rounding error on
Delta is a (bounded) noise channel, and OSA accepts optimally under it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class OsaResult:
    accepted: bool
    n_samples: int


def osa_accept(
    sample_delta: Callable[[], float],
    temp: float,
    sigma: float,
    rng: np.random.Generator,
    *,
    c_star: float = 0.0,
    max_samples: int = 100_000,
) -> OsaResult:
    """Decide accept/reject for one move from noisy cost-difference samples.

    ``sample_delta()`` returns one observation ``delta_i ~ Normal(Delta,
    sigma^2)``. Returns the decision and the number of samples drawn. ``c_star``
    is the rejection threshold (0.0 is the simple optimal strategy of the
    paper); ``max_samples`` guards against an unbounded chain.
    """
    if temp <= 0.0:
        raise ValueError("temp must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    beta = 1.0 / temp
    half = 0.5 * beta * sigma * sigma
    inv_var = 1.0 / (sigma * sigma)
    c_prev = 0.0  # c_0
    c = 0.0
    for n in range(1, max_samples + 1):
        c = c + float(sample_delta())
        exponent = -2.0 * (c + half) * (c_prev + half) * inv_var
        a = 1.0 if exponent >= 0.0 else math.exp(exponent)
        if rng.random() < a:
            return OsaResult(True, n)
        if c > c_star:
            return OsaResult(False, n)
        c_prev = c
    return OsaResult(False, max_samples)


def gaussian_delta_sampler(delta: float, sigma: float, rng: np.random.Generator) -> Callable[[], float]:
    """A sampler drawing ``Normal(delta, sigma^2)`` observations of a fixed Delta."""
    return lambda: float(rng.normal(delta, sigma))


def acceptance_rate(delta: float, temp: float, sigma: float, *, trials: int = 20_000,
                    c_star: float = 0.0, seed: int = 0) -> tuple[float, float]:
    """Empirical OSA acceptance rate and mean samples per decision for fixed Delta."""
    rng = np.random.default_rng(seed)
    accepts = 0
    total_samples = 0
    for _ in range(trials):
        res = osa_accept(gaussian_delta_sampler(delta, sigma, rng), temp, sigma, rng, c_star=c_star)
        accepts += int(res.accepted)
        total_samples += res.n_samples
    return accepts / trials, total_samples / trials


def _self_test():
    """Check detailed balance and the Metropolis-noise-free limit."""
    temp, sigma = 1.0, 0.5
    beta = 1.0 / temp
    # Detailed balance: PA(Delta) / PA(-Delta) should equal exp(-beta Delta).
    print(f"{'Delta':>6} {'PA(+)':>8} {'PA(-)':>8} {'ratio':>8} {'exp(-bD)':>9} {'<n>':>6}")
    worst = 0.0
    for delta in (0.25, 0.5, 1.0):
        pa_pos, n_pos = acceptance_rate(delta, temp, sigma, seed=1)
        pa_neg, _ = acceptance_rate(-delta, temp, sigma, seed=2)
        ratio = pa_pos / pa_neg
        target = math.exp(-beta * delta)
        rel = abs(ratio - target) / target
        worst = max(worst, rel)
        print(f"{delta:6.2f} {pa_pos:8.4f} {pa_neg:8.4f} {ratio:8.4f} {target:9.4f} {n_pos:6.2f}")
    print(f"worst detailed-balance relative error: {worst:.3f}")
    # Noise-free limit (tiny sigma): the n=1 rule is min(1, exp(-beta*Delta)),
    # i.e. Metropolis, since c_star=0 accepts/rejects on the first sample.
    pa_uphill, _ = acceptance_rate(1.0, 1.0, 1e-3, trials=5000, seed=3)
    print(f"near-noise-free PA(Delta=1) = {pa_uphill:.4f} (Metropolis exp(-1) = {math.exp(-1):.4f})")
    assert worst < 0.10, f"detailed balance violated: {worst}"
    assert abs(pa_uphill - math.exp(-1.0)) < 0.05, f"Metropolis limit off: {pa_uphill}"
    print("OSA self-test OK")


if __name__ == "__main__":
    _self_test()
