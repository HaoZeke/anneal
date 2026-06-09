"""Tests for the Thompson-allocated portfolio driver."""

import math

import numpy as np
import pytest

anneal = pytest.importorskip("anneal")

from experiments.portfolio import (  # noqa: E402
    PortfolioConfig,
    RESTART_ARM,
    _ArmPosterior,
    _enabled_arms,
    thompson_portfolio,
)
from experiments.tensor_surrogate import AdditiveSurrogate  # noqa: E402


class _Budget(Exception):
    pass


class Counter:
    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.best = float("inf")

    def _consume(self):
        if self.n >= self.budget:
            raise _Budget()
        self.n += 1

    def __call__(self, x):
        self._consume()
        v = float(self.fn(np.asarray(x, float).reshape(-1)))
        if math.isfinite(v) and v < self.best:
            self.best = v
        return v

    def counted_grad(self, grad):
        def jac(x):
            self._consume()
            return np.asarray(grad(np.asarray(x, float).reshape(-1)), float)

        return jac


def _styblinski_tang(dim):
    def fn(x):
        return 0.5 * float(np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))

    def grad(x):
        return 0.5 * (4.0 * x ** 3 - 32.0 * x + 5.0)

    low = np.full(dim, -5.0)
    high = np.full(dim, 5.0)
    return fn, grad, low, high


def _rastrigin(dim):
    def fn(x):
        return 10.0 * dim + float(np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def grad(x):
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    low = np.full(dim, -5.12)
    high = np.full(dim, 5.12)
    return fn, grad, low, high


def test_budget_is_never_exceeded():
    dim = 6
    fn, grad, low, high = _rastrigin(dim)
    budget = 600
    counter = Counter(fn, budget)
    thompson_portfolio(counter, low, high, dim, grad, np.random.default_rng(7))
    assert counter.n <= budget


def test_budget_respected_without_gradients():
    dim = 4
    fn, _, low, high = _rastrigin(dim)
    budget = 400
    counter = Counter(fn, budget)
    thompson_portfolio(counter, low, high, dim, None, np.random.default_rng(3))
    assert counter.n <= budget
    assert math.isfinite(counter.best)


def test_portfolio_beats_uniform_random_on_styblinski_tang():
    dim = 6
    fn, grad, low, high = _styblinski_tang(dim)
    budget = 1500
    counter = Counter(fn, budget)
    best = thompson_portfolio(
        counter, low, high, dim, grad, np.random.default_rng(11)
    )
    rng = np.random.default_rng(11)
    rand_counter = Counter(fn, budget)
    try:
        while True:
            rand_counter(rng.uniform(low, high))
    except _Budget:
        pass
    assert best < rand_counter.best
    # The 6D global minimum sits at about -39.166 * dim; the portfolio with
    # gradients and a final polish reaches the global basin reliably.
    assert best < -39.166 * dim * 0.95


def test_posterior_discounting_bounds_counts():
    config = PortfolioConfig(discount=0.9)
    post = _ArmPosterior(config)
    for _ in range(500):
        post.update(True)
    # Discounting caps the effective count at 1/(1-gamma) + prior.
    assert post.alpha <= 1.0 + 1.0 / (1.0 - 0.9) + 1.0
    assert post.beta >= 1.0


def test_restart_arm_always_enabled():
    config = PortfolioConfig()
    assert RESTART_ARM in _enabled_arms(2, None, config)
    assert RESTART_ARM in _enabled_arms(30, lambda x: x, config)


def test_additive_surrogate_from_points_recovers_separable():
    rng = np.random.default_rng(5)
    dim = 3
    low = np.full(dim, -2.0)
    high = np.full(dim, 2.0)
    X = rng.uniform(low, high, size=(400, dim))
    y = np.sum(X ** 2, axis=1)
    surr = AdditiveSurrogate.from_points(X, y, low, high, degree=6)
    samples = surr.sample(64, 0.05, rng)
    # Low-temperature draws concentrate near the separable minimum at 0.
    assert float(np.median(np.abs(samples))) < 0.5
    assert surr.pilot_work_units() == 0.0
