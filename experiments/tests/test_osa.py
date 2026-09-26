"""Tests for the Optimized Stochastic Annealing acceptance rule (Ball et al. 2018)."""

import math

import numpy as np

from experiments.osa import (
    OsaResult,
    acceptance_rate,
    gaussian_delta_sampler,
    osa_accept,
)


def test_osa_preserves_detailed_balance():
    """PA(Delta) / PA(-Delta) must equal exp(-beta Delta) for the noisy rule."""
    temp, sigma = 1.0, 0.5
    beta = 1.0 / temp
    for delta in (0.25, 0.5, 1.0):
        pa_pos, _ = acceptance_rate(delta, temp, sigma, trials=20_000, seed=1)
        pa_neg, _ = acceptance_rate(-delta, temp, sigma, trials=20_000, seed=2)
        ratio = pa_pos / pa_neg
        target = math.exp(-beta * delta)
        assert abs(ratio - target) / target < 0.05, (delta, ratio, target)


def test_osa_recovers_metropolis_in_the_noise_free_limit():
    """With negligible noise the c*=0 rule decides on the first sample and the
    uphill acceptance rate is the Metropolis value exp(-beta Delta)."""
    pa, _ = acceptance_rate(1.0, 1.0, 1e-3, trials=8_000, seed=3)
    assert abs(pa - math.exp(-1.0)) < 0.03
    # Downhill moves are always accepted.
    pa_down, _ = acceptance_rate(-1.0, 1.0, 1e-3, trials=2_000, seed=4)
    assert pa_down > 0.99


def test_osa_returns_sample_count():
    rng = np.random.default_rng(0)
    res = osa_accept(
        gaussian_delta_sampler(0.5, 0.5, rng), temp=1.0, sigma=0.5, rng=rng
    )
    assert isinstance(res, OsaResult)
    assert res.n_samples >= 1


def test_osa_rejects_invalid_parameters():
    rng = np.random.default_rng(0)
    sampler = gaussian_delta_sampler(0.0, 1.0, rng)
    for bad in ({"temp": 0.0, "sigma": 1.0}, {"temp": 1.0, "sigma": 0.0}):
        try:
            osa_accept(sampler, rng=rng, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
