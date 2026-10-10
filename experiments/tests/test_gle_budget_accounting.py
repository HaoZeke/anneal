from __future__ import annotations

import csv
import sys
import types

import numpy as np
import pytest

from experiments.scripts import run_cutest_benchmarks as benchmarks
from experiments.scripts import run_cutest_full_suite as suite


class _CountedProblem:
    name = "COUNTED_GLE"
    dim = 2
    low = np.array([2.0, -5.0])
    high = np.array([6.0, 1.0])
    design_low = low
    design_high = high

    def __init__(self, native_gradient=True):
        self.objective_points = []
        self.gradient_points = []
        if not native_gradient:
            self.grad = None

    def fn(self, x):
        point = np.asarray(x, dtype=np.float64).copy()
        self.objective_points.append(point)
        return float(np.dot(point, point))

    def grad(self, x):
        point = np.asarray(x, dtype=np.float64).copy()
        self.gradient_points.append(point)
        return 2.0 * point

    @property
    def work(self):
        return len(self.objective_points) + len(self.gradient_points)


class _CountedScalarGle:
    """Exercise Python adapters with the scalar core's callback-count contract."""

    def __init__(self, frequency_probes=False):
        self.calls = []
        self.probe_points = []
        self.module = types.SimpleNamespace(
            gle_langevin=self.run,
            additive_independence=lambda *_args, **_kwargs: {
                "best_val": 0.0,
                "n_evals": 0,
            },
        )
        if frequency_probes:
            self.module.estimate_gle_omega0 = self.estimate_frequency

    def estimate_frequency(self, _fn, grad_fn, low, high):
        centre = 0.5 * (np.asarray(low) + np.asarray(high))
        for axis in range(len(centre)):
            for sign in (-1.0, 1.0):
                point = centre.copy()
                point[axis] += sign * 1e-3
                self.probe_points.append(point)
                grad_fn(point)
        return 1.25

    def run(self, fn, grad_fn, low, high, max_fevals, **kwargs):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        x0 = kwargs.get("x0")
        start = 0.5 * (low + high) if x0 is None else np.asarray(x0).copy()
        self.calls.append(
            {
                "low": low.copy(),
                "high": high.copy(),
                "start": start.copy(),
                "max_fevals": int(max_fevals),
                "n_epochs": kwargs.get("n_epochs"),
            }
        )
        # The scalar core counts gradients; each dynamics gradient also has
        # one objective evaluation, including the supplied starting point.
        for _ in range(int(max_fevals)):
            best_val = fn(start)
            grad_fn(start)
        return {
            "best_val": best_val,
            "best_pos": start,
            "n_evals": int(max_fevals),
            "n_preconditioner_grads": 0,
        }


def _assert_common_start(prob, backend):
    assert len(backend.calls) == 1
    call = backend.calls[0]
    np.testing.assert_array_equal(call["low"], prob.design_low)
    np.testing.assert_array_equal(call["high"], prob.design_high)
    np.testing.assert_array_equal(
        call["start"], 0.5 * (prob.design_low + prob.design_high)
    )
    return call


@pytest.mark.parametrize("n_epochs", [1, 3])
def test_full_suite_standalone_gle_uses_requested_work_and_epochs(
    monkeypatch, n_epochs
):
    prob = _CountedProblem()
    backend = _CountedScalarGle()
    args = types.SimpleNamespace(n_epochs=n_epochs, k_fixed=20)
    monkeypatch.setattr(suite, "_anneal_module", lambda: backend.module)

    best_val, fevals = suite.run_driver(prob, "gle_langevin", 11, args)

    call = _assert_common_start(prob, backend)
    total_budget = 1 + args.n_epochs * args.k_fixed
    assert best_val == 20.0
    assert fevals == prob.work == 2 * call["max_fevals"]
    assert call["max_fevals"] == total_budget // 2
    assert call["n_epochs"] == n_epochs
    assert 0 <= total_budget - prob.work < 2


def test_full_suite_standalone_gle_charges_frequency_probes(monkeypatch):
    prob = _CountedProblem()
    backend = _CountedScalarGle(frequency_probes=True)
    args = types.SimpleNamespace(n_epochs=3, k_fixed=20)
    monkeypatch.setattr(suite, "_anneal_module", lambda: backend.module)

    best_val, fevals = suite.run_driver(prob, "gle_langevin", 11, args)

    call = _assert_common_start(prob, backend)
    total_budget = 1 + args.n_epochs * args.k_fixed
    probe_work = 2 * prob.dim
    assert len(backend.probe_points) == probe_work
    for observed, expected in zip(
        prob.gradient_points[:probe_work], backend.probe_points, strict=True
    ):
        np.testing.assert_array_equal(observed, expected)
    assert best_val == 20.0
    assert prob.work == probe_work + 2 * call["max_fevals"]
    assert fevals == prob.work
    assert call["max_fevals"] == (total_budget - probe_work) // 2
    assert call["n_epochs"] == args.n_epochs
    assert 0 <= total_budget - prob.work < 2


@pytest.mark.parametrize("native_gradient", [True, False], ids=["native", "fd"])
def test_manifest_standalone_gle_charges_actual_work_within_budget(
    monkeypatch, tmp_path, native_gradient
):
    prob = _CountedProblem(native_gradient=native_gradient)
    backend = _CountedScalarGle()
    output = tmp_path / "counted_gle.csv"
    n_epochs = 3
    k_fixed = 20
    monkeypatch.setattr(benchmarks, "load_default_manifest", lambda: [prob])
    monkeypatch.setitem(sys.modules, "anneal", backend.module)
    for driver in (
        "classical_sa",
        "mcmc_sa",
        "mcmc_sa_budgeted",
        "pt_sa_budgeted",
        "bayesian_mixing_sa",
        "portfolio_sa",
        "_bgsa_run",
    ):
        monkeypatch.setattr(benchmarks, driver, lambda *_args, **_kwargs: (0.0, 0))
    monkeypatch.setattr(benchmarks, "SCIPY_DRIVERS", {})
    monkeypatch.setattr(benchmarks, "DRIVERS", ["gle_langevin"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cutest_benchmarks.py",
            "--out",
            str(output),
            "--seeds",
            "1",
            "--n-epochs",
            str(n_epochs),
            "--k-fixed",
            str(k_fixed),
        ],
    )

    benchmarks.main()

    call = _assert_common_start(prob, backend)
    with output.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["driver"] == "gle_langevin"
        ]
    assert len(rows) == 1
    assert float(rows[0]["best_val"]) == 20.0
    # The manifest's one common f(x0) measurement is outside driver work.
    np.testing.assert_array_equal(prob.objective_points[0], call["start"])
    driver_work = prob.work - 1
    work_per_dynamics_call = 2 if native_gradient else prob.dim + 2
    assert driver_work == call["max_fevals"] * work_per_dynamics_call
    if native_gradient:
        assert len(prob.gradient_points) == call["max_fevals"]
    else:
        assert prob.gradient_points == []
        first_step = prob.objective_points[1 : prob.dim + 3]
        np.testing.assert_array_equal(first_step[0], call["start"])
        np.testing.assert_array_equal(first_step[1], call["start"])
        for axis, point in enumerate(first_step[2:]):
            expected = call["start"].copy()
            expected[axis] += benchmarks.FINITE_DIFFERENCE_GRAD_STEP
            np.testing.assert_array_equal(point, expected)
    assert int(rows[0]["fevals"]) == driver_work
    total_budget = 1 + n_epochs * k_fixed
    assert call["max_fevals"] == total_budget // work_per_dynamics_call
    assert call["n_epochs"] == n_epochs
    assert 0 <= total_budget - driver_work < work_per_dynamics_call
