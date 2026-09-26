from __future__ import annotations

import types

import numpy as np

from experiments.scripts import run_cutest_full_suite as suite


class _Problem:
    dim = 2
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    design_low = low
    design_high = high

    @staticmethod
    def fn(x):
        return float(np.dot(x, x))

    @staticmethod
    def grad(x):
        return 2.0 * np.asarray(x)


def _args():
    return types.SimpleNamespace(
        n_epochs=3,
        k_fixed=20,
        n_chains=4,
        k_min=2,
        k_check=2,
        k_max=8,
        rhat_threshold=1.1,
        straggler_top_k=1,
    )


def test_full_suite_dispatches_additive_independence_to_its_driver(monkeypatch):
    captured = {}

    def fake_driver(module, prob, seed, n_epochs, k_per_epoch):
        captured.update(
            module=module,
            prob=prob,
            seed=seed,
            n_epochs=n_epochs,
            k_per_epoch=k_per_epoch,
        )
        return -3.0, 61

    fake_anneal = object()
    monkeypatch.setattr(suite, "_anneal_module", lambda: fake_anneal)
    monkeypatch.setattr(suite, "_run_cutest_additive_independence", fake_driver)

    result = suite.run_driver(_Problem(), "additive_indep", 7, _args())

    assert result == (-3.0, 61)
    assert captured == {
        "module": fake_anneal,
        "prob": captured["prob"],
        "seed": 7,
        "n_epochs": 3,
        "k_per_epoch": 20,
    }
    assert captured["prob"].dim == 2


def test_full_suite_dispatches_gle_with_native_gradient_metadata(monkeypatch):
    captured = {}

    def fake_driver(
        module,
        prob,
        grad_fn,
        grad_kind,
        seed,
        n_epochs,
        k_per_epoch,
    ):
        captured.update(
            module=module,
            prob=prob,
            grad=grad_fn(np.array([0.5, -0.5])),
            grad_kind=grad_kind,
            seed=seed,
            n_epochs=n_epochs,
            k_per_epoch=k_per_epoch,
        )
        return -2.0, 42

    fake_anneal = object()
    monkeypatch.setattr(suite, "_anneal_module", lambda: fake_anneal)
    monkeypatch.setattr(suite, "_run_cutest_gle_langevin", fake_driver)

    result = suite.run_driver(_Problem(), "gle_langevin", 11, _args())

    assert result == (-2.0, 42)
    assert captured["module"] is fake_anneal
    assert captured["prob"].dim == 2
    assert captured["grad_kind"] == "native"
    assert np.array_equal(captured["grad"], np.array([1.0, -1.0]))
    assert (captured["seed"], captured["n_epochs"], captured["k_per_epoch"]) == (
        11,
        3,
        20,
    )
