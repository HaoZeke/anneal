"""Pytest for Config, Ledger, and cluster_search on the pyo3 surface."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from anneal import Config, Ledger, check_stop_limit, cluster_search


def lj_energy(x: np.ndarray) -> float:
    p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    iu = np.triu_indices(len(p), 1)
    inv6 = (1.0 / r2[iu]) ** 3
    return float(4.0 * np.sum(inv6 * inv6 - inv6))


def lj_grad(x: np.ndarray) -> np.ndarray:
    p = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    d = p[:, None, :] - p[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, np.inf)
    inv2 = 1.0 / r2
    inv6 = inv2**3
    coef = 24.0 * inv2 * (2.0 * inv6 * inv6 - inv6)
    g = np.einsum("ij,ijk->ik", coef, d)
    return np.asarray(g.reshape(-1), dtype=np.float64)


def test_config_recommended_and_for_cluster():
    rec = Config.recommended(38)
    base = Config.for_cluster(38)
    comm = Config.communicating(38)
    assert rec.n_points == 38
    assert base.n_points == 38
    assert rec.burst_moves
    assert rec.allocate_moves
    assert rec.depth_reward
    assert rec.tabu_on_stall
    assert rec.shared_visit_policy == "tabu"
    assert rec.orbit_complete_on_new
    assert rec.calibrated_screen
    assert rec.finish_improving
    assert base.finish_improving
    assert not rec.rigid_body
    assert not base.burst_moves
    assert not base.allocate_moves
    assert not base.depth_reward
    assert not base.tabu_on_stall
    assert not comm.burst_moves
    assert comm.allocate_moves
    assert not comm.depth_reward
    assert comm.orbit_complete_on_new
    assert comm.shared_visit_policy == "recognition"
    assert rec.shared_visit_policy == "tabu"


def test_config_rejects_tiny_n():
    with pytest.raises(ValueError, match="at least 2"):
        Config.recommended(1)
    with pytest.raises(ValueError, match="at least 2"):
        Config.for_cluster(1)


def test_ledger_budget():
    led = Ledger(4000)
    assert led.budget == 4000
    assert led.spent == 0
    assert led.remaining == 4000
    assert led.best == float("inf")


def test_cluster_search_returns_best_energy_and_hops():
    n = 4
    out = cluster_search(lj_energy, lj_grad, n, 800, seed=0, recommended=True)
    assert "best" in out
    assert "best_energy" in out
    assert "hops" in out
    assert "solved" not in out
    assert np.asarray(out["best"]).shape == (3 * n,)
    assert np.isfinite(out["best_energy"])
    assert out["hops"] > 0


def test_cluster_search_for_cluster_flag():
    out = cluster_search(lj_energy, lj_grad, 4, 400, seed=1, recommended=False)
    assert np.isfinite(out["best_energy"])
    assert out["hops"] > 0


def test_cluster_search_rejects_bad_n_and_budget():
    with pytest.raises(ValueError, match="at least 2"):
        cluster_search(lj_energy, lj_grad, 1, 100, seed=0)
    with pytest.raises(ValueError, match="positive"):
        cluster_search(lj_energy, lj_grad, 4, 0, seed=0)


def test_cluster_search_start_kwarg():
    import inspect

    from anneal._core import cluster_search as core_cluster_search

    if "start" not in inspect.signature(core_cluster_search).parameters:
        pytest.skip("rebuild the extension for cluster_search start=")
    n = 4
    start = np.array(
        [0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.1],
        dtype=np.float64,
    )
    out = cluster_search(lj_energy, lj_grad, n, 200, seed=0, start=start)
    assert np.asarray(out["best"]).shape == (3 * n,)
    assert np.isfinite(out["best_energy"])


WATER_SITES = np.array(
    [[0.0, 0.0, -0.0656], [0.0, 0.757, 0.5205], [0.0, -0.757, 0.5205]],
    dtype=np.float64,
)


def rigid_energy_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    """LJ on the centres at sigma 3 plus a quadratic pull on the rotations."""
    x = np.asarray(x, dtype=np.float64)
    half = x.size // 2
    centres = x[:half] / 3.0
    energy = lj_energy(centres) + 0.05 * float(np.dot(x[half:], x[half:]))
    grad = np.concatenate([-lj_grad(centres) / 3.0, 0.1 * x[half:]])
    return energy, grad


def test_tip4p_configs_are_rigid():
    comm = Config.communicating_tip4p(10)
    rec = Config.recommended_tip4p(10)
    assert comm.rigid_body and rec.rigid_body
    assert comm.n_points == 10
    assert comm.calibrated_screen
    assert comm.shared_visit_policy == "recognition"
    assert rec.depth_reward and rec.tabu_on_stall
    assert not comm.orbit_complete_on_new
    with pytest.raises(ValueError, match="at least 2"):
        Config.communicating_tip4p(1)


@pytest.mark.parametrize("preset", ["communicating", "recommended", "base"])
def test_cluster_search_water_sites(preset):
    n = 3
    start = np.concatenate(
        [np.array([0.0, 0.0, 0.0, 3.4, 0.0, 0.0, 0.0, 3.4, 0.0]), np.full(3 * n, 0.3)]
    )
    out = cluster_search(
        lambda x: rigid_energy_grad(x)[0],
        lambda x: rigid_energy_grad(x)[1],
        n,
        600,
        seed=0,
        communicating=preset == "communicating",
        recommended=preset == "recommended",
        start=start,
        water_sites=WATER_SITES,
    )
    assert np.asarray(out["best"]).shape == (6 * n,)
    assert out["best_energy"] <= rigid_energy_grad(start)[0]
    assert out["best_energy"] == pytest.approx(rigid_energy_grad(out["best"])[0])


def test_cluster_search_water_sites_validation():
    f = lambda x: rigid_energy_grad(x)[0]  # noqa: E731
    g = lambda x: rigid_energy_grad(x)[1]  # noqa: E731
    start = np.zeros(12)
    with pytest.raises(ValueError, match="needs a start"):
        cluster_search(f, g, 2, 100, water_sites=WATER_SITES)
    with pytest.raises(ValueError, match="6\\*n"):
        cluster_search(f, g, 2, 100, start=np.zeros(6), water_sites=WATER_SITES)
    with pytest.raises(ValueError, match="must be 9"):
        cluster_search(f, g, 2, 100, start=start, water_sites=np.zeros(6))
    with pytest.raises(ValueError, match="excludes"):
        cluster_search(f, g, 2, 100, start=start, water_sites=WATER_SITES, derived=True)
    with pytest.raises(ValueError, match="excludes"):
        cluster_search(f, g, 2, 100, start=start, water_sites=WATER_SITES, rigid=True)
    with pytest.raises(ValueError, match="excludes"):
        cluster_search(
            f, g, 2, 100, start=start, water_sites=WATER_SITES, species=[1, 2]
        )


def test_cluster_search_ras_flag_exists_and_recommended_defaults():
    import inspect

    from anneal._core import cluster_search as core_cluster_search

    sig = inspect.signature(core_cluster_search)
    ras = sig.parameters["ras"]
    assert ras.default is False
    assert ras.kind is inspect.Parameter.KEYWORD_ONLY

    rec = Config.recommended(38)
    assert rec.burst_moves
    assert rec.allocate_moves
    assert rec.depth_reward
    assert rec.tabu_on_stall

    out = cluster_search(lj_energy, lj_grad, 4, 800, seed=0, recommended=True)
    assert "best_energy" in out
    assert "hops" in out
    assert "charged" not in out
    assert "floors" not in out
    assert "returned" not in out
    assert "events" not in out


def test_cluster_search_species_and_rigid_charts():
    n = 4
    species = [1, 1, 2, 2]
    sigmas = [1.0, 1.0, 1.3, 1.3]
    out = cluster_search(
        lj_energy,
        lj_grad,
        n,
        200,
        seed=2,
        species=species,
        sigmas=sigmas,
        target=-0.5,
    )
    assert np.asarray(out["best"]).shape == (3 * n,)
    assert np.isfinite(out["best_energy"])

    def tip_energy(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))

    def tip_grad(x):
        return 2.0 * np.asarray(x, dtype=np.float64)

    rigid = cluster_search(
        tip_energy,
        tip_grad,
        2,
        120,
        seed=3,
        rigid=True,
        layout="block",
        target=1.0,
    )
    assert np.asarray(rigid["best"]).shape == (12,)
    assert np.isfinite(rigid["best_energy"])
    assert rigid["best_energy"] <= 1.0 + 1e-6


def test_cluster_search_ras_true_uses_archive_outcome():
    from anneal._core import cluster_search as core_cluster_search

    try:
        out = core_cluster_search(lj_energy, lj_grad, 4, 800, seed=0, ras=True)
    except ValueError as exc:
        assert "graphkey" in str(exc)
        return
    assert np.isfinite(out["best_energy"])
    assert np.asarray(out["best"]).shape == (12,)
    assert "charged" in out
    assert "floors" in out
    assert "returned" in out
    assert "events" in out
    assert out["charged"] > 0
    assert out["floors"] >= 1


LJ98_GLOBAL = -543.665361
LJ98_SHELF = -543.642957
LJ75_SHELF = -396.282249
HARNESS_WINDOW = 0.001


def test_check_stop_limit_refuses_known_shelves():
    with pytest.raises(ValueError, match="LJ98"):
        check_stop_limit(98, LJ98_SHELF)
    with pytest.raises(ValueError, match="LJ75"):
        check_stop_limit(75, LJ75_SHELF)


def test_check_stop_limit_keeps_the_minimum_and_the_window(capfd):
    assert check_stop_limit(98, LJ98_GLOBAL) == LJ98_GLOBAL
    window = LJ98_GLOBAL + HARNESS_WINDOW
    assert check_stop_limit(98, window) == window
    assert capfd.readouterr().err == ""


def test_check_stop_limit_warns_above_the_window(capfd):
    limit = LJ98_GLOBAL + 0.01
    assert check_stop_limit(98, limit) == limit
    err = capfd.readouterr().err
    assert "warning" in err
    assert "LJ98" in err
    # -543.641 is not the Cs shelf, and it sits above the harness window.
    missed = -543.641
    assert check_stop_limit(98, missed) == missed
    assert "warning" in capfd.readouterr().err


def test_cluster_search_rejects_known_shelves():
    with pytest.raises(ValueError, match="LJ98"):
        cluster_search(lj_energy, lj_grad, 98, 1, seed=0, target=LJ98_SHELF)
    with pytest.raises(ValueError, match="LJ75"):
        cluster_search(lj_energy, lj_grad, 75, 1, seed=0, target=LJ75_SHELF)


def _globalopt_driver():
    path = Path(__file__).resolve().parents[1] / "scripts" / "optbench" / "anneal_globalopt.py"
    spec = importlib.util.spec_from_file_location("anneal_globalopt_stop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_problem(root: Path, name: str, energy: float, kind: str = "lennard-jones") -> None:
    path = root / "problems" / "globalopt"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.toml").write_text(
        "\n".join(
            [
                f'id = "globalopt/{name}"',
                f'problem = "{name}"',
                "[potential]",
                f'kind = "{kind}"',
                "[reference]",
                f"energy = {energy}",
                "[convergence]",
                "max_force_calls = 10",
                "energy_above_minimum = 0.001",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_load_problem_checks_the_published_minimum(tmp_path, capfd):
    driver = _globalopt_driver()
    _write_problem(tmp_path, "lj98", LJ98_GLOBAL)
    problem = driver.load_problem(tmp_path, "lj98")
    assert problem["convergence"]["energy_minimum"] == LJ98_GLOBAL
    assert capfd.readouterr().err == ""

    _write_problem(tmp_path, "lj98", LJ98_GLOBAL + 0.01)
    warned = driver.load_problem(tmp_path, "lj98")
    assert warned["convergence"]["energy_minimum"] == LJ98_GLOBAL + 0.01
    assert "warning" in capfd.readouterr().err

    _write_problem(tmp_path, "lj98", LJ98_SHELF)
    with pytest.raises(ValueError, match="LJ98"):
        driver.load_problem(tmp_path, "lj98")

    _write_problem(tmp_path, "lj75", LJ75_SHELF)
    with pytest.raises(ValueError, match="LJ75"):
        driver.load_problem(tmp_path, "lj75")

    _write_problem(tmp_path, "lj100-binary", -1.0, kind="binary-lennard-jones")
    binary = driver.load_problem(tmp_path, "lj100-binary")
    assert binary["convergence"]["energy_minimum"] == -1.0
