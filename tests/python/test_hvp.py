"""Hessian-vector products: call counts, orders and the cached centre."""

from __future__ import annotations

import numpy as np
import pytest

import rgpot

DE, A, RE = 0.7102, 1.6047, 2.897
BOX = np.eye(3) * 20.0


def _cluster() -> tuple[np.ndarray, np.ndarray]:
    s = RE / np.sqrt(2.0)
    sites = np.array(
        [[0, 0, 0], [s, s, 0], [s, 0, s], [0, s, s], [-s, s, 0], [s, -s, 0], [0, -s, s]]
    )
    k = np.arange(sites.size, dtype=float).reshape(sites.shape)
    positions = 10.0 + sites + 0.07 * np.sin(1.3 * k)
    return np.ascontiguousarray(positions), np.full(len(sites), 78, dtype=np.int32)


def _direction(n: int) -> np.ndarray:
    return np.ascontiguousarray(np.sin(0.7 * np.arange(3 * n) + 0.4).reshape(n, 3))


def _dense_hessian(positions: np.ndarray) -> np.ndarray:
    n = len(positions)
    h = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            r = np.linalg.norm(d)
            e = np.exp(-A * (r - RE))
            d1 = 2 * DE * A * e * (1 - e)
            d2 = 2 * DE * A * A * e * (2 * e - 1)
            uu = np.outer(d, d) / r**2
            k = d2 * uu + d1 / r * (np.eye(3) - uu)
            h[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += k
            h[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += k
            h[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] -= k
            h[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] -= k
    return h


def _relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def test_analytic_morse_product_matches_the_dense_hessian() -> None:
    positions, types = _cluster()
    v = _direction(len(positions))
    pot = rgpot.MorsePot()
    assert pot.has_analytic_hvp()
    hv = pot.hvp(positions, types, BOX, v, scheme="analytic")
    want = (_dense_hessian(positions) @ v.ravel()).reshape(v.shape)
    assert hv.shape == v.shape
    assert _relative(hv, want) < 1e-12
    assert pot.hvp_counts() == {
        "products": 1,
        "force_calls": 0,
        "centre_reused": 0,
        "analytic": 1,
    }


@pytest.mark.parametrize(("scheme", "step", "order"), [("forward", 4e-3, 1.0), ("central", 2e-2, 2.0)])
def test_finite_differences_converge_at_their_order(scheme: str, step: float, order: float) -> None:
    positions, types = _cluster()
    v = _direction(len(positions))
    want = (_dense_hessian(positions) @ v.ravel()).reshape(v.shape)
    pot = rgpot.MorsePot()
    coarse = _relative(pot.hvp(positions, types, BOX, v, step=step, scheme=scheme), want)
    fine = _relative(pot.hvp(positions, types, BOX, v, step=step / 2, scheme=scheme), want)
    assert np.log2(coarse / fine) == pytest.approx(order, abs=0.1)


def test_k_forward_products_cost_k_plus_one_force_calls() -> None:
    positions, types = _cluster()
    v = _direction(len(positions))
    k = 8
    pot = rgpot.MorsePot()
    for _ in range(k):
        pot.hvp(positions, types, BOX, v, scheme="forward")
    counts = pot.hvp_counts()
    assert counts["products"] == k
    assert counts["force_calls"] == k + 1
    assert counts["centre_reused"] == k - 1

    naive = rgpot.MorsePot()
    for _ in range(k):
        naive.invalidate_hvp()
        naive.hvp(positions, types, BOX, v, scheme="forward")
    assert naive.hvp_counts()["force_calls"] == 2 * k

    pot.reset_hvp_counts()
    for _ in range(k):
        pot.hvp(positions, types, BOX, v)
    assert pot.hvp_counts()["force_calls"] == 0
    assert pot.hvp_counts()["analytic"] == k


def test_auto_is_forward_without_an_analytic_product() -> None:
    positions = np.ascontiguousarray([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.3, 1.1, 0.2]])
    types = np.ones(3, dtype=np.int32)
    v = _direction(3)
    pot = rgpot.LJPot()
    assert not pot.has_analytic_hvp()
    for _ in range(3):
        pot.hvp(positions, types, BOX, v)
    assert pot.hvp_counts()["force_calls"] == 4
    with pytest.raises(ValueError, match="hessianVectorImpl"):
        pot.hvp(positions, types, BOX, v, scheme="analytic")


def test_the_centre_call_primes_the_products() -> None:
    positions, types = _cluster()
    pot = rgpot.MorsePot()
    energy, forces, variance = pot.hvp_centre(positions, types, BOX)
    ref_energy, ref_forces, ref_variance = rgpot.MorsePot()(positions, types, BOX)
    assert energy == ref_energy
    assert variance == ref_variance
    np.testing.assert_array_equal(forces, ref_forces)
    pot.hvp(positions, types, BOX, _direction(len(positions)), scheme="forward")
    assert pot.hvp_counts()["force_calls"] == 2
    assert pot.hvp_counts()["centre_reused"] == 1


def test_a_new_geometry_cell_or_species_is_a_new_centre() -> None:
    positions, types = _cluster()
    v = _direction(len(positions))
    box = BOX.copy()
    pot = rgpot.MorsePot()

    def calls_for_two() -> int:
        before = pot.hvp_counts()["force_calls"]
        pot.hvp(positions, types, box, v, scheme="forward")
        pot.hvp(positions, types, box, v, scheme="forward")
        return pot.hvp_counts()["force_calls"] - before

    assert calls_for_two() == 3
    assert calls_for_two() == 2
    positions[2, 1] = np.nextafter(positions[2, 1], np.inf)
    assert calls_for_two() == 3
    box[1, 1] = 21.0
    assert calls_for_two() == 3
    types = types.copy()
    types[0] = 79
    assert calls_for_two() == 3
    pot.invalidate_hvp()
    assert calls_for_two() == 3


def test_malformed_requests_are_refused() -> None:
    positions, types = _cluster()
    v = _direction(len(positions))
    pot = rgpot.MorsePot()
    with pytest.raises(ValueError, match="direction"):
        pot.hvp(positions, types, BOX, np.ascontiguousarray(v[:-1]))
    with pytest.raises(ValueError, match="scheme"):
        pot.hvp(positions, types, BOX, v, scheme="backward")
    with pytest.raises(ValueError, match="step"):
        pot.hvp(positions, types, BOX, v, step=-1.0)
    bad = v.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pot.hvp(positions, types, BOX, bad)
    zero = pot.hvp(positions, types, BOX, np.zeros_like(v), scheme="central")
    assert not zero.any()
    assert pot.hvp_counts()["force_calls"] == 0
