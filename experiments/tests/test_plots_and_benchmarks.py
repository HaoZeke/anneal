"""Smoke tests for benchmark catalog + plot helpers (Dolan-More, More-Wild, Pareto)."""

import os
import sys
import tempfile
import types

import numpy as np
import pytest

from experiments.benchmarks.catalog import CATALOG, list_problems, get_problem
from experiments.benchmarks.runner import run_benchmarks


class _QuadraticCutestProblem:
    name = "QUAD2"
    dim = 2
    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])

    def fn(self, x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(x * x))


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


def test_cutest_full_suite_enumeration_filters_and_deduplicates(monkeypatch):
    from experiments.scripts import run_cutest_full_suite as suite

    fake_pycutest = types.SimpleNamespace(
        find_problems=lambda constraints: {
            "unconstrained": ["ROSENBR", "BIG", "VARIABLE", "DUP"],
            "bound": ["BOX3", "DUP"],
        }[constraints],
        problem_properties=lambda name: {
            "ROSENBR": {"n": 2},
            "BIG": {"n": 200},
            "VARIABLE": {"n": "variable"},
            "BOX3": {"n": 3},
            "DUP": {"n": 4},
        }[name],
    )

    monkeypatch.setitem(sys.modules, "pycutest", fake_pycutest)
    monkeypatch.setattr(suite, "setup_cutest_env", lambda: None)

    assert suite.list_target_problems(dim_cap=10) == [
        suite.TargetProblem("BOX3", "bound", 3),
        suite.TargetProblem("DUP", "unconstrained", 4),
        suite.TargetProblem("ROSENBR", "unconstrained", 2),
    ]


def test_cutest_full_suite_resume_key_normalises_seed():
    from experiments.scripts import run_cutest_full_suite as suite

    rows = [
        {"problem": "ROSENBR", "driver": "classical", "seed": "0"},
        {"problem": "ROSENBR", "driver": "bgsa", "seed": 1},
    ]

    assert suite.resume_keys(rows) == {
        ("ROSENBR", "classical", "0"),
        ("ROSENBR", "bgsa", "1"),
    }


def test_cutest_summary_reports_status_and_winners(tmp_path):
    from experiments.scripts.summarize_cutest_benchmarks import summarize_csv

    inp = tmp_path / "cutest.csv"
    inp.write_text(
        "problem,kind,dim,driver,seed,fevals,best_val,wall_time_s,f_x0,solved,status\n"
        "P1,unconstrained,2,classical,0,10,1.0,0.1,2.0,1,ok\n"
        "P1,unconstrained,2,bgsa,0,12,0.5,0.2,2.0,1,ok\n"
        "P2,bound,3,classical,0,20,4.0,0.1,5.0,0,ok\n"
        "P2,bound,3,bgsa,0,0,nan,0.0,5.0,0,timeout\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.csv"

    summary = summarize_csv(inp, out)

    assert summary.coverage.problem_count == 2
    assert summary.coverage.cell_count == 4
    bgsa = summary.by_driver["bgsa"]
    assert bgsa.cells == 2
    assert bgsa.ok == 1
    assert bgsa.timeout == 1
    assert bgsa.best_cells == 1
    assert out.read_text(encoding="utf-8").splitlines()[0].startswith("driver,cells,ok")


def test_cutest_classical_budget_matches_fixed_epoch_steps():
    from experiments.scripts import run_cutest_benchmarks as cutest

    _, fevals = cutest.classical_sa(
        _QuadraticCutestProblem(),
        seed=0,
        n_epochs=20,
        k_fixed=80,
    )

    assert fevals == 1601


def test_cutest_legacy_mcmc_never_overshoots_k_max():
    from experiments.scripts import run_cutest_benchmarks as cutest

    n_chains = 4
    k_max = 35
    _, fevals = cutest.mcmc_sa(
        _QuadraticCutestProblem(),
        seed=0,
        n_epochs=1,
        n_chains=n_chains,
        k_min=30,
        k_check=20,
        k_max=k_max,
        rhat_threshold=-1.0,
    )

    assert fevals <= n_chains + n_chains * k_max


def test_cutest_budgeted_mcmc_matches_classical_epoch_budget():
    from experiments.scripts import run_cutest_benchmarks as cutest

    prob = _QuadraticCutestProblem()
    n_epochs = 20
    k_fixed = 80
    n_chains = 4

    _, dense_fevals = cutest.mcmc_sa_budgeted(
        prob,
        seed=0,
        n_epochs=n_epochs,
        n_chains=n_chains,
        epoch_budget=k_fixed,
        k_min=8,
        k_check=8,
        rhat_threshold=-1.0,
    )
    _, sparse_fevals = cutest.mcmc_sa_budgeted(
        prob,
        seed=0,
        n_epochs=n_epochs,
        n_chains=n_chains,
        epoch_budget=k_fixed,
        k_min=8,
        k_check=8,
        rhat_threshold=-1.0,
        sparse=True,
        straggler_top_k=2,
    )

    budget = n_chains + n_epochs * k_fixed
    assert dense_fevals <= budget
    assert sparse_fevals <= budget


def test_cutest_budgeted_pt_matches_classical_epoch_budget_and_is_deterministic():
    from experiments.scripts import run_cutest_benchmarks as cutest

    kwargs = dict(
        prob=_QuadraticCutestProblem(),
        seed=11,
        n_epochs=12,
        n_chains=4,
        epoch_budget=80,
        swap_period=3,
        return_diagnostics=True,
    )

    first = cutest.pt_sa_budgeted(**kwargs)
    second = cutest.pt_sa_budgeted(**kwargs)
    budget = kwargs["n_chains"] + kwargs["n_epochs"] * kwargs["epoch_budget"]

    assert first == second
    assert first[1] <= budget
    assert first[2]["swap_attempts"] > 0
    assert first[2]["swap_accepts"] <= first[2]["swap_attempts"]


def test_cutest_full_suite_accepts_budgeted_pt_driver():
    from experiments.scripts import run_cutest_full_suite as suite

    assert suite.parse_drivers("pt_sa_budgeted") == ("pt_sa_budgeted",)
