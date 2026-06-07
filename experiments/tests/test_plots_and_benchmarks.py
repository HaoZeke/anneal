"""Smoke tests for benchmark catalog + plot helpers (Dolan-More, More-Wild, Pareto)."""

import os
import inspect
import multiprocessing as mp
import sys
import tempfile
import time
import types
import warnings

import numpy as np
import pytest

from experiments.benchmarks.catalog import list_problems, get_problem
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


def test_performance_profile_masks_all_failed_rows_without_warning():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.performance_profile import plot_performance_profile

    costs = np.array([[np.nan, np.nan], [100.0, 200.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig, ax = plot_performance_profile(costs, ["solver_a", "solver_b"])

    assert ax.get_legend() is not None


def test_data_profile_renders():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.data_profile import plot_data_profile

    rng = np.random.default_rng(1)
    fevals = rng.uniform(100, 10_000, size=(6, 2))
    dims = np.array([2, 5, 5, 10, 10, 2])
    fig, ax = plot_data_profile(fevals, dims, ["boltzmann", "fast"])
    assert ax.get_xlabel().startswith("Budget")
    assert "simplex-gradient-equivalent" in ax.get_xlabel()


def test_pareto_renders_and_finds_front():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.pareto import plot_pareto, pareto_front

    pts = np.array([[100.0, 5.0], [200.0, 3.0], [400.0, 1.0], [800.0, 1.5]])
    front = pareto_front(pts)
    assert set(front.tolist()) == {0, 1, 2}  # (800, 1.5) is dominated by (400, 1.0)
    fig, ax = plot_pareto([("solver_a", pts)])
    assert ax.get_xlabel() == "Cost (objective-equivalent evaluations)"
    assert ax.get_legend() is not None


def test_pareto_symlog_y_handles_zero_gaps_and_outliers():
    pytest.importorskip("matplotlib")
    pytest.importorskip("chemparseplot")
    from experiments.plots.pareto import plot_pareto

    pts = np.array([[100.0, 0.0], [200.0, 1e-5], [400.0, 4.0], [800.0, 5e4]])

    fig, ax = plot_pareto([("solver_a", pts)], symlog_y=True, y_linthresh=1e-3)

    assert ax.get_yscale() == "symlog"
    assert ax.get_ylim()[0] == 0.0
    assert ax.get_ylim()[1] > pts[:, 1].max()


def test_cutest_render_uses_stable_driver_order_and_labels():
    from experiments.scripts.render_plots import display_solver_names, ordered_solvers

    rows = [
        {"driver": "bgsa_auto"},
        {"driver": "classical"},
        {"driver": "bayesian_mixing_sa"},
        {"driver": "mcmc_sa_sparse_budgeted"},
    ]

    solvers = ordered_solvers(rows)

    assert solvers == [
        "classical",
        "bayesian_mixing_sa",
        "mcmc_sa_sparse_budgeted",
        "bgsa_auto",
    ]
    assert display_solver_names(solvers) == [
        "Classical SA",
        "Bayesian mixing",
        "Sparse MCMC-SA (budgeted)",
        "Automatic BGSA",
    ]


def test_cutest_render_accepts_requested_driver_subset():
    from experiments.scripts.render_plots import ordered_solvers

    rows = [
        {"driver": "classical"},
        {"driver": "mcmc_sa"},
        {"driver": "bgsa_auto"},
    ]

    assert ordered_solvers(rows, requested=["bgsa_auto", "classical"]) == [
        "bgsa_auto",
        "classical",
    ]
    with pytest.raises(ValueError, match="absent"):
        ordered_solvers(rows, requested=["missing_driver"])


def test_cutest_pareto_points_use_problem_seed_relative_gaps():
    from experiments.scripts.render_plots import pareto_points_by_solver

    rows = [
        {
            "problem": "BIGSCALE",
            "driver": "classical",
            "seed": "0",
            "fevals": "100",
            "best_val": "1000000",
            "f_x0": "2000000",
            "status": "ok",
        },
        {
            "problem": "BIGSCALE",
            "driver": "bayesian_mixing_sa",
            "seed": "0",
            "fevals": "120",
            "best_val": "999000",
            "f_x0": "2000000",
            "status": "ok",
        },
        {
            "problem": "SMALLSCALE",
            "driver": "classical",
            "seed": "0",
            "fevals": "100",
            "best_val": "1",
            "f_x0": "10",
            "status": "ok",
        },
        {
            "problem": "SMALLSCALE",
            "driver": "bayesian_mixing_sa",
            "seed": "0",
            "fevals": "120",
            "best_val": "0",
            "f_x0": "10",
            "status": "ok",
        },
    ]

    points = pareto_points_by_solver(rows, ["classical", "bayesian_mixing_sa"])

    assert points[0][0] == "classical"
    assert points[0][1][:, 1].tolist() == pytest.approx([0.0005, 0.1])
    assert points[1][1][:, 1].tolist() == pytest.approx([0.0, 0.0])


def test_cutest_profile_matrix_keeps_problem_seed_failures():
    from experiments.scripts.render_plots import profile_matrix_by_cell

    rows = [
        {
            "problem": "P",
            "driver": "classical",
            "seed": "0",
            "dim": "2",
            "fevals": "10",
            "best_val": "1.0",
            "solved": "1",
            "status": "ok",
        },
        {
            "problem": "P",
            "driver": "bgsa_auto",
            "seed": "0",
            "dim": "2",
            "fevals": "20",
            "best_val": "0.5",
            "solved": "1",
            "status": "ok",
        },
        {
            "problem": "P",
            "driver": "classical",
            "seed": "1",
            "dim": "2",
            "fevals": "99",
            "best_val": "nan",
            "solved": "0",
            "status": "target-timeout",
        },
        {
            "problem": "P",
            "driver": "bgsa_auto",
            "seed": "1",
            "dim": "2",
            "fevals": "30",
            "best_val": "0.25",
            "solved": "1",
            "status": "ok",
        },
    ]

    cell_keys, fevals, dims = profile_matrix_by_cell(
        rows, ["classical", "bgsa_auto"]
    )

    assert cell_keys == [("P", "0"), ("P", "1")]
    assert dims.tolist() == [2, 2]
    assert fevals[0].tolist() == [10.0, 20.0]
    assert np.isnan(fevals[1, 0])
    assert fevals[1, 1] == 30.0


def test_cutest_data_profile_budget_limit_tracks_solved_budget_range():
    from experiments.scripts.render_plots import data_profile_kappa_max

    fevals = np.array([[1600.0, 1604.0], [30000.0, np.nan]])
    dims = np.array([2, 12])

    assert data_profile_kappa_max(fevals, dims) > 200.0


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


def test_cutest_full_suite_shards_targets_by_stable_index():
    from experiments.scripts import run_cutest_full_suite as suite

    targets = [
        suite.TargetProblem("A", "unconstrained", 1),
        suite.TargetProblem("B", "unconstrained", 1),
        suite.TargetProblem("C", "bound", 1),
        suite.TargetProblem("D", "bound", 1),
        suite.TargetProblem("E", "bound", 1),
    ]

    assert [target.name for target in suite.shard_targets(targets, 0, 2)] == ["A", "C", "E"]
    assert [target.name for target in suite.shard_targets(targets, 1, 2)] == ["B", "D"]
    with pytest.raises(ValueError, match="shard_index"):
        suite.shard_targets(targets, 2, 2)


def test_cutest_shard_combiner_deduplicates_identical_cells(tmp_path):
    from experiments.scripts import combine_cutest_shards as combine

    header = "problem,kind,dim,driver,seed,fevals,best_val,wall_time_s,f_x0,solved,status\n"
    row_a = "P1,bound,2,classical,0,10,1.0,0.1,2.0,1,ok\n"
    row_b = "P2,bound,3,bgsa,0,20,0.5,0.2,2.0,1,ok\n"
    shard_a = tmp_path / "a.csv"
    shard_b = tmp_path / "b.csv"
    shard_a.write_text(header + row_a + row_b, encoding="utf-8")
    shard_b.write_text(header + row_a, encoding="utf-8")

    rows = combine.combine_rows([shard_b, shard_a])

    assert [(row["problem"], row["driver"]) for row in rows] == [
        ("P1", "classical"),
        ("P2", "bgsa"),
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


def test_cutest_effective_design_bounds_shrink_extreme_finite_bounds():
    from experiments.benchmarks.cutest_runner import effective_design_bounds

    low = np.array([-5.0, 1.0, -1.0e10])
    high = np.array([5.0, 5.0, 1.0])
    anchor = np.array([0.0, 0.0, 0.0])

    design_low, design_high, design_x0 = effective_design_bounds(
        low, high, anchor, x_box=5.0
    )

    assert design_x0.tolist() == pytest.approx([0.0, 1.0, 0.0])
    assert design_low.tolist() == pytest.approx([-5.0, 1.0, -5.0])
    assert design_high.tolist() == pytest.approx([5.0, 5.0, 1.0])
    assert np.all(design_low >= low)
    assert np.all(design_high <= high)


def test_cutest_env_requires_cutest_install(tmp_path, monkeypatch):
    from experiments.benchmarks.cutest_runner import cutest_env

    bench = tmp_path / ".bench"
    (bench / "SIFDecode" / "install" / "bin").mkdir(parents=True)
    (bench / "sif").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIXI_PROJECT_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="CUTEst bootstrap incomplete"):
        cutest_env()


def test_cutest_summary_reports_status_and_winners(tmp_path):
    from experiments.scripts.summarize_cutest_benchmarks import summarize_csv

    inp = tmp_path / "cutest.csv"
    inp.write_text(
        "problem,kind,dim,driver,seed,fevals,best_val,wall_time_s,f_x0,solved,status\n"
        "P1,unconstrained,2,classical,0,10,1.0,0.1,2.0,1,ok\n"
        "P1,unconstrained,2,bgsa,0,12,0.5,0.2,2.0,1,ok\n"
        "P2,bound,3,classical,0,20,4.0,0.1,5.0,0,ok\n"
        "P2,bound,3,bgsa,0,0,nan,0.0,5.0,0,timeout\n"
        "P3,bound,4,bgsa,0,0,nan,0.0,7.0,0,target-timeout\n",
        encoding="utf-8",
    )
    out = tmp_path / "summary.csv"

    summary = summarize_csv(inp, out)

    assert summary.coverage.problem_count == 3
    assert summary.coverage.cell_count == 5
    bgsa = summary.by_driver["bgsa"]
    assert bgsa.cells == 3
    assert bgsa.ok == 1
    assert bgsa.timeout == 2
    assert bgsa.best_cells == 1
    assert bgsa.best_share == pytest.approx(0.5)
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


def test_cutest_full_suite_accepts_evict_cache_option():
    from experiments.scripts import run_cutest_full_suite as suite

    args = suite.build_parser().parse_args(["--evict-cache"])

    assert args.evict_cache is True


def test_cutest_full_suite_evicts_pycutest_cache(tmp_path, monkeypatch):
    from experiments.scripts import run_cutest_full_suite as suite

    cache = tmp_path / "cache"
    stale = cache / "stale-problem"
    stale.mkdir(parents=True)
    (stale / "compiled.so").write_text("stale", encoding="utf-8")
    (cache / "stale.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        suite,
        "cutest_env",
        lambda: {"PYCUTEST_CACHE": str(cache)},
    )

    suite.evict_pycutest_cache()

    assert not stale.exists()
    assert not (cache / "stale.txt").exists()
    assert (cache / "pycutest_cache_holder").is_dir()


def test_cutest_full_suite_appends_target_timeout_rows():
    from experiments.scripts import run_cutest_full_suite as suite

    rows = []
    seen = set()
    target = suite.TargetProblem("SLOW", "bound", 17)
    appended = suite.append_target_timeout_rows(
        rows,
        seen,
        target,
        [(0, "classical"), (0, "bayesian_mixing_sa")],
        f0=12.5,
    )

    assert appended == 2
    assert {row["status"] for row in rows} == {"target-timeout"}
    assert {row["f_x0"] for row in rows} == {12.5}
    assert ("SLOW", "classical", "0") in seen


def test_cutest_full_suite_cell_timeout_terminates_hanging_driver(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("subprocess timeout uses fork where available")
    from experiments.scripts import run_cutest_full_suite as suite

    def hangs(_prob, _driver, _seed, _args):
        time.sleep(5)
        return 0.0, 1

    monkeypatch.setattr(suite, "run_driver", hangs)
    args = types.SimpleNamespace(per_problem_timeout=1)

    best_val, fevals, f0, status = suite.run_driver_cell(
        _QuadraticCutestProblem(), "classical", 0, args
    )

    assert np.isnan(best_val)
    assert fevals == 0
    assert np.isnan(f0)
    assert status == "timeout"


def test_bgsa_log_acceptance_probability_is_stable():
    from experiments.scripts.demo_bgsa import _log_accept_probability

    assert _log_accept_probability(1_000.0) == 1.0
    assert _log_accept_probability(999.0) == 1.0
    small = _log_accept_probability(-1_000.0)
    assert 0.0 <= small <= 1.0
    assert np.isfinite(_log_accept_probability(0.0))


def test_bgsa_q_gaussian_kinetics_are_overflow_stable():
    from experiments.scripts.demo_bgsa import dk_dp_q_gaussian, kinetic_q_gaussian

    p = np.array([1e308, -1e308], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isinf(kinetic_q_gaussian(p, 1.5))
        drift = dk_dp_q_gaussian(p, 1.5)
    assert np.all(np.isfinite(drift))
    assert np.all(drift == 0.0)


def test_bgsa_q_grid_bounds_respect_high_dimensional_momentum_limit():
    from experiments.scripts.demo_bgsa import q_grid_bounds

    q_lo, q_hi = q_grid_bounds(30, {"q_mean": 1.15, "q_sd": 0.1})

    assert 1.0 <= q_lo < q_hi
    assert q_hi < 1.0 + 2.0 / 30.0


def test_bgsa_hmc_rejects_nonfinite_hamiltonians_without_warning(monkeypatch):
    from experiments.scripts import demo_bgsa as bgsa

    x0 = np.zeros(2, dtype=np.float64)
    monkeypatch.setattr(bgsa, "LOW", np.full(2, -1.0))
    monkeypatch.setattr(bgsa, "HIGH", np.full(2, 1.0))
    monkeypatch.setattr(bgsa, "OBJ_FN", lambda _x: float("inf"))
    monkeypatch.setattr(bgsa, "OBJ_GRAD", lambda x: np.zeros_like(x))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x, accepted, _n_calls, value = bgsa.hmc_sa_step(
            np.random.default_rng(0), x0, float("inf"), 1.0, 0.0, 1, 2, q=1.5
        )

    assert not accepted
    assert np.all(x == x0)
    assert value == float("inf")


def test_bgsa_hmc_rejects_overflowing_momentum_updates_without_warning(monkeypatch):
    from experiments.scripts import demo_bgsa as bgsa

    x0 = np.zeros(2, dtype=np.float64)
    monkeypatch.setattr(bgsa, "LOW", np.full(2, -1.0))
    monkeypatch.setattr(bgsa, "HIGH", np.full(2, 1.0))
    monkeypatch.setattr(bgsa, "OBJ_FN", lambda _x: 0.0)
    monkeypatch.setattr(bgsa, "OBJ_GRAD", lambda x: np.full_like(x, 1e308))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        x, accepted, _n_calls, value = bgsa.hmc_sa_step(
            np.random.default_rng(0), x0, 0.0, 1.0, 1e308, 1, 2, q=1.5
        )

    assert not accepted
    assert np.all(x == x0)
    assert value == 0.0


def test_bgsa_hmc_default_uses_omelyan_force_stages(monkeypatch):
    from experiments.scripts import demo_bgsa as bgsa

    grad_calls = 0

    def grad(x):
        nonlocal grad_calls
        grad_calls += 1
        return np.zeros_like(x)

    x0 = np.zeros(2, dtype=np.float64)
    monkeypatch.setattr(bgsa, "LOW", np.full(2, -1.0))
    monkeypatch.setattr(bgsa, "HIGH", np.full(2, 1.0))
    monkeypatch.setattr(bgsa, "OBJ_FN", lambda _x: 0.0)
    monkeypatch.setattr(bgsa, "OBJ_GRAD", grad)

    _x, accepted, n_calls, value = bgsa.hmc_sa_step(
        np.random.default_rng(0), x0, 0.0, 1.0, 0.01, 1, 2, q=1.0
    )

    assert accepted
    assert value == 0.0
    assert grad_calls == 3
    assert n_calls == 4


def test_bgsa_low_discrepancy_init_is_seeded_and_bounded():
    from experiments.scripts import demo_bgsa as bgsa

    low = np.array([-1.0, -2.0, -3.0])
    high = np.array([1.0, 2.0, 3.0])

    first = bgsa.low_discrepancy_init(np.random.default_rng(7), 8, low, high)
    second = bgsa.low_discrepancy_init(np.random.default_rng(7), 8, low, high)
    third = bgsa.low_discrepancy_init(np.random.default_rng(8), 8, low, high)

    assert first.shape == (8, 3)
    assert np.all(first >= low)
    assert np.all(first <= high)
    assert np.allclose(first, second)
    assert not np.allclose(first, third)


def test_bgsa_run_pilot_wires_low_discrepancy_starts(monkeypatch):
    from experiments.scripts import demo_bgsa as bgsa

    monkeypatch.setattr(bgsa, "LOW", np.array([-1.0, -1.0], dtype=np.float64))
    monkeypatch.setattr(bgsa, "HIGH", np.array([1.0, 1.0], dtype=np.float64))
    monkeypatch.setattr(bgsa, "OBJ_FN", lambda x: float(np.sum(np.asarray(x) ** 2)))
    monkeypatch.setattr(bgsa, "OBJ_GRAD", lambda x: 2.0 * np.asarray(x))

    hmc_starts = []
    rw_starts = []
    hmc_params = []
    rw_params = []

    def hmc_pilot(seed, t_init, eps, L, n_steps, q=1.0, x0=None):
        assert x0 is not None
        x0 = np.asarray(x0, dtype=np.float64)
        hmc_starts.append(x0.copy())
        hmc_params.append((t_init, eps, L, q))
        return float(np.sum(x0 * x0)), 0.65, x0, 1

    def rw_pilot(seed, T, sigma, n_steps, x0=None):
        assert x0 is not None
        x0 = np.asarray(x0, dtype=np.float64)
        rw_starts.append(x0.copy())
        rw_params.append((T, sigma))
        return float(np.sum(x0 * x0)), 0.234, 1

    monkeypatch.setattr(bgsa, "hmc_pilot", hmc_pilot)
    monkeypatch.setattr(bgsa, "rw_pilot", rw_pilot)
    monkeypatch.setattr(
        bgsa,
        "fit_empirical_bayes_priors",
        lambda _obs, _dim: {
            "t_mean": 0.0,
            "t_sd": 1.0,
            "e_mean": -3.0,
            "e_sd": 1.0,
            "l_mean": 1.6,
            "l_sd": 0.7,
            "q_mean": 1.05,
            "q_sd": 0.02,
        },
    )
    monkeypatch.setattr(
        bgsa, "fit_laplace_4d", lambda _obs, _dim, priors=None: (1.0, 0.05, 2, 1.05)
    )
    monkeypatch.setattr(bgsa, "fit_t_sigma_rw", lambda _obs: (1.0, 0.2))
    monkeypatch.setattr(bgsa, "_pilot_t_hot_from_acceptance", lambda _obs, _t: 2.0)
    monkeypatch.setattr(
        bgsa,
        "pilot_landscape_features",
        lambda _scout, _pilot: {
            "grad_sens": 0.0,
            "sigma_sens": 0.0,
            "best_val_cv": 0.0,
            "q_v_lift": 0.0,
        },
    )

    bgsa.run_pilot(
        seed=5,
        n_pilot=3,
        pilot_steps=4,
        dim=2,
        n_rw_pilot=2,
        rw_steps=3,
        n_scout=2,
    )

    assert len(hmc_starts) == 5
    assert len(rw_starts) == 2
    assert np.all(np.asarray(hmc_starts) >= bgsa.LOW)
    assert np.all(np.asarray(hmc_starts) <= bgsa.HIGH)
    assert np.all(np.asarray(rw_starts) >= bgsa.LOW)
    assert np.all(np.asarray(rw_starts) <= bgsa.HIGH)
    assert len(set(hmc_params)) > 1
    assert len(set(rw_params)) > 1


def test_bayesian_mixing_sa_initializes_from_low_discrepancy_design():
    from experiments.scripts import run_cutest_benchmarks as cutest

    class RecordingProblem(_QuadraticCutestProblem):
        def __init__(self):
            self.seen = []

        def fn(self, x):
            x = np.asarray(x, dtype=np.float64)
            self.seen.append(x.copy())
            return float(np.sum(x * x))

    prob = RecordingProblem()
    best, fevals, diagnostics = cutest.bayesian_mixing_sa(
        prob, seed=11, max_fevals=2, return_diagnostics=True
    )

    expected = cutest._low_discrepancy_starts(prob.low, prob.high, 2, seed=11)
    assert fevals == 2
    assert diagnostics["proposal_counts"] == [0, 0]
    assert np.isfinite(best)
    assert np.allclose(np.asarray(prob.seen[:2]), expected)


def test_cutest_bgsa_single_chain_uses_rust_hmc_binding(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class FakeHistory:
        best_val = -1.25
        total_accepted = 2
        total_rejected = 4

    def run_hmc(obj_fn, grad_fn, low, high, **kwargs):
        captured["low"] = np.asarray(low)
        captured["high"] = np.asarray(high)
        captured["kwargs"] = kwargs
        assert obj_fn(np.zeros(2)) == 0.0
        assert grad_fn(np.zeros(2)).shape == (2,)
        return FakeHistory()

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    def run_pilot(*args, **kwargs):
        captured["pilot_args"] = args
        captured["pilot_kwargs"] = kwargs
        return (
            3.0,
            0.25,
            2,
            1.1,
            0.4,
            np.array([0.2, -0.2], dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        )

    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=run_pilot,
        hmc_sa=lambda *_args, **_kwargs: pytest.fail("demo HMC must not run"),
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    best_val, fevals = cutest._bgsa_run(
        _QuadraticCutestProblem(),
        seed=7,
        n_epochs=3,
        k_per_epoch=5,
        n_chains=4,
        driver="bgsa",
    )

    assert best_val == -1.25
    assert captured["kwargs"]["x0"] == pytest.approx([0.2, -0.2])
    assert captured["kwargs"]["q"] == pytest.approx(1.1)
    assert captured["kwargs"]["l_steps"] == 2
    assert captured["kwargs"]["steps_per_epoch"] == 1
    assert captured["pilot_args"] == (7, 4, 5)
    assert captured["pilot_kwargs"] == {
        "dim": 2,
        "n_rw_pilot": 4,
        "rw_steps": 5,
        "n_scout": 4,
    }
    assert fevals == 11 + cutest._rust_hmc_fd_work_units(
        dim=2,
        n_trajectories=3,
        l_steps=2,
        total_accepted=2,
    )


def test_cutest_loader_preserves_native_gradient(monkeypatch):
    from experiments.benchmarks import cutest_runner

    class NativeProblem:
        n = 2
        bl = np.array([-1.0, -2.0], dtype=np.float64)
        bu = np.array([3.0, 4.0], dtype=np.float64)
        x0 = np.zeros(2, dtype=np.float64)

        def obj(self, x):
            x = np.asarray(x, dtype=np.float64)
            return float(np.dot(x, x))

        def grad(self, x):
            x = np.asarray(x, dtype=np.float64)
            return np.array([10.0 + x[0], -3.0 + x[1]], dtype=np.float64)

    fake_pycutest = types.SimpleNamespace(
        import_problem=lambda *_args, **_kwargs: NativeProblem()
    )
    monkeypatch.setitem(sys.modules, "pycutest", fake_pycutest)
    monkeypatch.setattr(cutest_runner, "setup_cutest_env", lambda: None)

    prob = cutest_runner.load("NATIVEGRAD")

    assert prob.fn(np.array([2.0, 3.0])) == pytest.approx(13.0)
    assert prob.grad(np.array([2.0, 3.0])) == pytest.approx([12.0, 0.0])


def test_cutest_bgsa_uses_native_gradient_and_native_work_units(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class GradientCutestProblem(_QuadraticCutestProblem):
        def grad(self, x):
            x = np.asarray(x, dtype=np.float64)
            return np.array([10.0 + x[0], -3.0 + x[1]], dtype=np.float64)

    class FakeHistory:
        best_val = -2.0
        total_accepted = 2
        total_rejected = 1

    def run_hmc(_obj_fn, grad_fn, _low, _high, **_kwargs):
        captured["grad_at_probe"] = grad_fn(np.array([0.5, -0.25], dtype=np.float64))
        return FakeHistory()

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=lambda *_args, **_kwargs: (
            3.0,
            0.25,
            2,
            1.1,
            0.4,
            np.array([0.2, -0.2], dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        ),
        hmc_sa=lambda *_args, **_kwargs: pytest.fail("demo HMC must not run"),
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    best_val, fevals = cutest._bgsa_run(
        GradientCutestProblem(),
        seed=7,
        n_epochs=3,
        k_per_epoch=5,
        n_chains=4,
        driver="bgsa",
    )

    assert best_val == -2.0
    assert captured["grad_at_probe"] == pytest.approx([10.5, -3.25])
    assert cutest._rust_hmc_steps_per_epoch_budget(
        epoch_budget=14,
        dim=2,
        l_steps=2,
        grad_kind="native",
    ) == 2
    assert fevals == 11 + cutest._rust_hmc_native_grad_work_units(
        n_trajectories=3,
        l_steps=2,
        total_accepted=2,
    )


def test_cutest_bgsa_hmc_uses_none_for_missing_pilot_position():
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class FiveDimProblem(_QuadraticCutestProblem):
        dim = 5
        low = np.full(5, -1.0)
        high = np.full(5, 1.0)

        def grad(self, x):
            return np.asarray(x, dtype=np.float64)

    class FakeHistory:
        best_val = -2.0
        total_accepted = 0

    def run_hmc(*_args, **kwargs):
        captured["x0"] = kwargs["x0"]
        return FakeHistory()

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)

    cutest._run_cutest_rust_hmc(
        fake_anneal,
        FiveDimProblem(),
        lambda x: np.asarray(x, dtype=np.float64),
        "native",
        seed=7,
        n_epochs=3,
        epoch_budget=200,
        t_map=3.0,
        e_map=0.25,
        L_map=2,
        q_map=1.1,
        best_pilot_pos=None,
    )

    assert captured["x0"] is None


def test_cutest_bgsa_clamps_q_for_high_dimensional_rust_hmc(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class HighDimProblem(_QuadraticCutestProblem):
        dim = 30
        low = np.full(30, -1.0)
        high = np.full(30, 1.0)

        def grad(self, x):
            return np.asarray(x, dtype=np.float64)

    class FakeHistory:
        best_val = -2.0
        total_accepted = 0

    def run_hmc(*_args, **kwargs):
        captured["q"] = kwargs["q"]
        return FakeHistory()

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=lambda *_args, **_kwargs: (
            3.0,
            0.25,
            2,
            1.15,
            0.4,
            np.zeros(30, dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        ),
        hmc_sa=lambda *_args, **_kwargs: pytest.fail("demo HMC must not run"),
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    cutest._bgsa_run(
        HighDimProblem(),
        seed=7,
        n_epochs=3,
        k_per_epoch=200,
        n_chains=4,
        driver="bgsa",
    )

    assert captured["q"] < 1.0 + 2.0 / 30.0


def test_cutest_bgsa_metad_falls_back_when_cv_is_undefined(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    class ReducedCoordinateProblem(_QuadraticCutestProblem):
        dim = 2
        low = np.array([-1.0])
        high = np.array([1.0])

        def grad(self, x):
            return np.asarray(x, dtype=np.float64)

    class FakeHistory:
        best_val = -3.0
        total_accepted = 0

    def run_hmc(*_args, **_kwargs):
        return FakeHistory()

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=lambda *_args, **_kwargs: (
            3.0,
            0.25,
            2,
            1.1,
            0.4,
            np.array([0.2], dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        ),
        bgsa_metad=lambda *_args, **_kwargs: pytest.fail("metad CV is undefined"),
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    best_val, fevals = cutest._bgsa_run(
        ReducedCoordinateProblem(),
        seed=7,
        n_epochs=3,
        k_per_epoch=200,
        n_chains=4,
        driver="bgsa_metad",
    )

    assert best_val == -3.0
    hmc_steps = cutest._rust_hmc_steps_per_epoch_budget(
        epoch_budget=200,
        dim=1,
        l_steps=2,
        grad_kind="native",
    )
    assert fevals == 11 + cutest._rust_hmc_native_grad_work_units(
        n_trajectories=3 * hmc_steps,
        l_steps=2,
    )


def test_cutest_bgsa_hmc_steps_shrink_when_pilot_selects_long_trajectory():
    from experiments.scripts import run_cutest_benchmarks as cutest

    assert cutest._rust_hmc_steps_per_epoch_budget(
        epoch_budget=200,
        dim=6,
        l_steps=32,
        grad_kind="native",
    ) == 2


def test_cutest_bgsa_auto_budgets_hmc_candidates(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class GradientCutestProblem(_QuadraticCutestProblem):
        def grad(self, x):
            return np.asarray(x, dtype=np.float64)

    class FakeHistory:
        best_val = -5.0
        total_accepted = 0

    def run_hmc(*_args, **kwargs):
        captured["hmc_kwargs"] = kwargs
        return FakeHistory()

    def bgsa_metad(*_args, **_kwargs):
        return 1.0, 40, None, None, None, None

    def bgsa_pt_metad(*_args, **kwargs):
        captured["pt_inner"] = kwargs["k_inner"]
        return 0.5, 30, None, None, None, None, None, None, None

    def bgsa_pt_hybrid_v2(*_args, **kwargs):
        captured["hybrid_inner"] = kwargs["k_inner"]
        return 2.0, 70, None, None, None, None, None, None

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=lambda *_args, **_kwargs: (
            3.0,
            0.25,
            2,
            1.1,
            0.4,
            np.array([0.2, -0.2], dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        ),
        bgsa_metad=bgsa_metad,
        bgsa_pt_metad=bgsa_pt_metad,
        bgsa_pt_hybrid_v2=bgsa_pt_hybrid_v2,
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    best_val, fevals = cutest._bgsa_run(
        GradientCutestProblem(),
        seed=7,
        n_epochs=2,
        k_per_epoch=40,
        n_chains=4,
        driver="bgsa_auto",
    )

    assert best_val == -5.0
    assert captured["hmc_kwargs"]["steps_per_epoch"] == 1
    assert captured["pt_inner"] == 2
    assert captured["hybrid_inner"] == 1
    assert fevals == 11 + 40 + 30 + 70 + cutest._rust_hmc_native_grad_work_units(
        n_trajectories=2,
        l_steps=2,
    )


def test_cutest_bgsa_auto_skips_metad_when_cv_is_undefined(monkeypatch):
    from experiments.scripts import run_cutest_benchmarks as cutest

    captured = {}

    class ReducedCoordinateProblem(_QuadraticCutestProblem):
        dim = 2
        low = np.array([-1.0])
        high = np.array([1.0])

        def grad(self, x):
            return np.asarray(x, dtype=np.float64)

    class FakeHistory:
        best_val = -5.0
        total_accepted = 0

    def run_hmc(*_args, **kwargs):
        captured["hmc_kwargs"] = kwargs
        return FakeHistory()

    def bgsa_pt_hybrid_v2(*_args, **kwargs):
        captured["hybrid_inner"] = kwargs["k_inner"]
        return 2.0, 70, None, None, None, None, None, None

    fake_anneal = types.SimpleNamespace(run_hmc=run_hmc)
    fake_demo = types.SimpleNamespace(
        OBJ_FN=None,
        OBJ_GRAD=None,
        LOW=None,
        HIGH=None,
        run_pilot=lambda *_args, **_kwargs: (
            3.0,
            0.25,
            2,
            1.1,
            0.4,
            np.array([0.2], dtype=np.float64),
            11,
            8.0,
            2.0,
            {"grad_sens": 0.0},
        ),
        bgsa_metad=lambda *_args, **_kwargs: pytest.fail("metad CV is undefined"),
        bgsa_pt_metad=lambda *_args, **_kwargs: pytest.fail("metad CV is undefined"),
        bgsa_pt_hybrid_v2=bgsa_pt_hybrid_v2,
    )
    monkeypatch.setitem(sys.modules, "anneal", fake_anneal)
    monkeypatch.setitem(sys.modules, "demo_bgsa", fake_demo)

    best_val, fevals = cutest._bgsa_run(
        ReducedCoordinateProblem(),
        seed=7,
        n_epochs=2,
        k_per_epoch=40,
        n_chains=4,
        driver="bgsa_auto",
    )

    assert best_val == -5.0
    assert captured["hmc_kwargs"]["steps_per_epoch"] == 1
    assert captured["hybrid_inner"] == 1
    assert fevals == 11 + 70 + cutest._rust_hmc_native_grad_work_units(
        n_trajectories=2,
        l_steps=2,
    )


def test_cutest_bgsa_pilot_budget_is_derived_from_epoch_budget():
    from experiments.scripts import run_cutest_benchmarks as cutest

    assert cutest._bgsa_pilot_budget(
        n_epochs=30,
        k_per_epoch=200,
        n_chains=4,
    ) == {
        "n_pilot": 8,
        "pilot_steps": 20,
        "n_rw_pilot": 4,
        "rw_steps": 20,
        "n_scout": 4,
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.scripts.demo_bgsa",
        "experiments.scripts.demo_mcmc_vs_classical",
        "experiments.scripts.run_cutest_benchmarks",
    ],
)
def test_gelman_rubin_returns_inf_for_nonfinite_traces_without_warning(module_name):
    module = __import__(module_name, fromlist=["gelman_rubin_max"])
    traces = [
        [np.array([0.0]), np.array([float("inf")])],
        [np.array([1.0]), np.array([2.0])],
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rhat = module.gelman_rubin_max(traces)

    assert rhat == float("inf")


def test_bgsa_pilot_statistics_ignore_nonfinite_values_without_warning():
    from experiments.scripts.demo_bgsa import (
        _segment_log_weight_increment,
        fit_empirical_bayes_priors,
        pilot_landscape_features,
    )

    obs = [
        {
            "t_init": 1.0,
            "epsilon": 0.10,
            "L": 2,
            "q": 1.1,
            "accept_rate": 0.2,
            "best_val": float("inf"),
        },
        {
            "t_init": 2.0,
            "epsilon": 0.20,
            "L": 3,
            "q": 1.2,
            "accept_rate": 0.5,
            "best_val": 1e308,
        },
        {
            "t_init": 3.0,
            "epsilon": 0.30,
            "L": 4,
            "q": 1.3,
            "accept_rate": 0.7,
            "best_val": 8.0,
        },
        {
            "t_init": 4.0,
            "epsilon": 0.40,
            "L": 5,
            "q": 1.4,
            "accept_rate": 0.8,
            "best_val": float("nan"),
        },
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        priors = fit_empirical_bayes_priors(obs, dim=5)
        features = pilot_landscape_features(obs[:2], obs[2:])
        log_inc = _segment_log_weight_increment(
            np.array([1.0, 1e308, float("inf"), float("nan")])
        )

    assert all(np.isfinite(value) for value in priors.values())
    assert all(np.isfinite(value) for value in features.values())
    assert np.all(np.isfinite(log_inc))
    assert log_inc[0] > log_inc[1]


def test_cutest_bayesian_mixing_has_small_user_api():
    from experiments.scripts import run_cutest_benchmarks as cutest

    params = inspect.signature(cutest.bayesian_mixing_sa).parameters

    assert list(params) == ["prob", "seed", "max_fevals", "return_diagnostics"]


def test_cutest_bayesian_mixing_is_online_deterministic_and_budgeted():
    from experiments.scripts import run_cutest_benchmarks as cutest

    kwargs = dict(
        prob=_QuadraticCutestProblem(),
        seed=19,
        max_fevals=321,
        return_diagnostics=True,
    )

    first = cutest.bayesian_mixing_sa(**kwargs)
    second = cutest.bayesian_mixing_sa(**kwargs)

    assert first == second
    assert first[1] <= kwargs["max_fevals"]
    assert first[2]["n_chains"] >= 2
    assert first[2]["swap_attempts"] > 0
    assert first[2]["posterior_accept_mean"] > 0.0
    assert max(first[2]["proposal_counts"]) > kwargs["max_fevals"] // 2


def test_cutest_bayesian_mixing_keeps_easy_problem_competitive():
    from experiments.scripts import run_cutest_benchmarks as cutest

    prob = _QuadraticCutestProblem()
    classical_val, classical_fevals = cutest.classical_sa(
        prob, seed=0, n_epochs=12, k_fixed=40
    )
    bayes_val, bayes_fevals = cutest.bayesian_mixing_sa(
        prob, seed=0, max_fevals=classical_fevals
    )

    assert bayes_fevals == classical_fevals
    assert bayes_val <= 5.0 * classical_val


def test_cutest_full_suite_accepts_bayesian_mixing_driver():
    from experiments.scripts import run_cutest_full_suite as suite

    assert suite.parse_drivers("bayesian_mixing_sa") == ("bayesian_mixing_sa",)


def test_sota_compare_methods_respect_common_budget():
    from experiments import sota_compare

    def sphere(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(x * x))

    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])

    assert {
        "hybrid_de",
        "hybrid_bmsa",
        "basinhopping",
        "diff_evol",
        "lbfgs_multistart",
        "plain_sa",
    } <= set(sota_compare.METHODS)

    for name in ("hybrid_de", "hybrid_bmsa", "plain_sa"):
        counter = sota_compare.Counter(sphere, budget=80)
        best = sota_compare.METHODS[name](
            counter,
            low,
            high,
            2,
            np.random.default_rng(7),
        )
        assert counter.n <= counter.budget
        assert np.isfinite(best)


def test_sota_cutest_native_gradient_polish_consumes_budget():
    from experiments.scripts import sota_cutest

    grad_calls = 0

    def sphere(x):
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum(x * x))

    def grad(x):
        nonlocal grad_calls
        grad_calls += 1
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * x

    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    counter = sota_cutest.Counter(sphere, budget=80)

    best = sota_cutest.hybrid_de(
        counter,
        low,
        high,
        2,
        grad,
        np.random.default_rng(11),
        n_polish=2,
    )

    assert counter.n <= counter.budget
    assert counter.grad_evals == grad_calls
    assert grad_calls > 0
    assert np.isfinite(best)


def test_anneal_sota_low_discrepancy_population_is_deterministic():
    from experiments.anneal_sota import low_discrepancy_population

    low = np.array([-1.0, -2.0, -3.0])
    high = np.array([1.0, 2.0, 3.0])

    first = low_discrepancy_population(low, high, n=8, skip=1)
    second = low_discrepancy_population(low, high, n=8, skip=1)

    assert first.shape == (8, 3)
    assert np.all(first >= low)
    assert np.all(first <= high)
    assert np.allclose(first, second)


def test_anneal_sota_shifted_low_discrepancy_population_is_seeded():
    from experiments.anneal_sota import low_discrepancy_population

    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])

    first = low_discrepancy_population(low, high, 8, rng=np.random.default_rng(3))
    second = low_discrepancy_population(low, high, 8, rng=np.random.default_rng(3))
    third = low_discrepancy_population(low, high, 8, rng=np.random.default_rng(4))

    assert np.all(first >= low)
    assert np.all(first <= high)
    assert np.allclose(first, second)
    assert not np.allclose(first, third)


def test_anneal_sota_qmc_hybrid_sees_deceptive_basin():
    from experiments import sota_compare
    from experiments.anneal_sota import qmc_annealed_hybrid

    def deceptive_basin(x):
        x = np.asarray(x, dtype=np.float64)
        shallow = np.sum((x - 0.35) ** 2)
        deep = 0.03 * np.sum((x - np.array([-0.5, 1.0 / 3.0])) ** 2) - 0.75
        return float(min(shallow, deep))

    low = np.array([-1.0, -1.0])
    high = np.array([1.0, 1.0])
    counter = sota_compare.Counter(deceptive_basin, budget=120)

    best = qmc_annealed_hybrid(
        counter,
        low,
        high,
        dim=2,
        grad=None,
        rng=np.random.default_rng(7),
        n_polish=2,
    )

    assert counter.n <= counter.budget
    assert best < -0.7
