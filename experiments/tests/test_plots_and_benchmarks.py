"""Smoke tests for benchmark catalog + plot helpers (Dolan-More, More-Wild, Pareto)."""

import os
import tempfile

import numpy as np
import pytest

from experiments.benchmarks.catalog import CATALOG, list_problems, get_problem
from experiments.benchmarks.runner import run_benchmarks


def test_catalog_has_expected_problems():
    names = list_problems()
    assert "styb_tang_2d" in names
    assert "rosenbrock_5d" in names
    assert "ackley_10d" in names
    assert "schwefel_2d" in names
    assert "levy_5d" in names
    assert "griewank_2d" in names
    # All problems return finite values at their f_star anchor (when easy).
    p = get_problem("styb_tang_2d")
    assert np.isfinite(p.fn(np.array([-2.903534, -2.903534])))


def test_runner_emits_csv():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "bench.csv")
        rows = run_benchmarks(
            problems=["styb_tang_2d"],
            solvers=("boltzmann",),
            seeds=2,
            n_epochs=10,
            steps_per_epoch=20,
            out_path=out,
        )
        assert len(rows) == 2
        assert os.path.exists(out)
        with open(out) as f:
            header = f.readline().strip()
        assert header.startswith("problem,dim,solver,seed,fevals")


def test_performance_profile_renders():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.performance_profile import plot_performance_profile
    rng = np.random.default_rng(0)
    costs = rng.uniform(100, 10_000, size=(8, 3))
    costs[0, 1] = np.nan  # one failed cell
    fig, ax = plot_performance_profile(costs, ["boltzmann", "fast", "gsa"])
    assert ax.get_xlim()[0] == 1.0
    assert 0 < ax.get_ylim()[1] <= 1.05


def test_data_profile_renders():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.data_profile import plot_data_profile
    rng = np.random.default_rng(1)
    fevals = rng.uniform(100, 10_000, size=(6, 2))
    dims = np.array([2, 5, 5, 10, 10, 2])
    fig, ax = plot_data_profile(fevals, dims, ["boltzmann", "fast"])
    assert ax.get_xlabel().startswith("Budget")


def test_pareto_renders_and_finds_front():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.pareto import plot_pareto, pareto_front
    pts = np.array([[100.0, 5.0], [200.0, 3.0], [400.0, 1.0], [800.0, 1.5]])
    front = pareto_front(pts)
    assert set(front.tolist()) == {0, 1, 2}  # (800, 1.5) is dominated by (400, 1.0)
    fig, ax = plot_pareto([("solver_a", pts)])
    assert ax.get_legend() is not None
