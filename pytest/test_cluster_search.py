"""Pytest for Config, Ledger, and cluster_search on the pyo3 surface."""

import numpy as np
import pytest

from anneal import Config, Ledger, cluster_search


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
