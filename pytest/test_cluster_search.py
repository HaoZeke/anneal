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
    assert rec.n_points == 38
    assert base.n_points == 38
    assert rec.burst_moves
    assert rec.allocate_moves
    assert rec.depth_reward
    assert rec.tabu_on_stall
    assert not base.burst_moves
    assert not base.allocate_moves
    assert not base.depth_reward
    assert not base.tabu_on_stall


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
