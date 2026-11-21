# MIT License
# Copyright 2023--present rgpot developers
"""Dense Hessians and the derivative scheme every request reports."""

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


def _lj_cluster() -> tuple[np.ndarray, np.ndarray]:
    positions, _ = _cluster()
    scaled = 10.0 + (positions - 10.0) * (1.12 / RE)
    return np.ascontiguousarray(scaled), np.ones(len(positions), dtype=np.int32)


def _direction(n: int) -> np.ndarray:
    return np.ascontiguousarray(np.sin(0.7 * np.arange(3 * n) + 0.4).reshape(n, 3))


def _morse_hessian(positions: np.ndarray) -> np.ndarray:
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


def _largest_relative(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def test_the_analytic_morse_hessian_is_the_closed_form() -> None:
    positions, types = _cluster()
    pot = rgpot.MorsePot()
    h, scheme = pot.hessian(positions, types, BOX, "analytic")
    assert scheme == "analytic"
    assert h.shape == (3 * len(positions), 3 * len(positions))
    assert _largest_relative(h, _morse_hessian(positions)) < 1e-12
    np.testing.assert_array_equal(h, h.T)
    counts = pot.derivative_counts()
    assert counts["hessian_analytic"] == 1
    assert counts["hessian_force_calls"] == 0
    assert pot.hvp_counts()["force_calls"] == 0


def test_the_dual_number_lj_hessian_and_product_agree_with_the_analytic_ones() -> None:
    positions, types = _lj_cluster()
    pot = rgpot.LJPot()
    analytic, used = pot.hessian(positions, types, BOX, "exact")
    assert used == "analytic"
    ad, used = pot.hessian(positions, types, BOX, "ad")
    assert used == "ad"
    assert _largest_relative(ad, analytic) < 1e-12
    np.testing.assert_array_equal(ad, ad.T)

    v = _direction(len(positions))
    hv, used = pot.hvp_resolved(positions, types, BOX, v, scheme="ad")
    assert used == "ad"
    want = (ad @ v.ravel()).reshape(v.shape)
    assert _largest_relative(hv, want) < 1e-12
    _, used = pot.hvp_resolved(positions, types, BOX, v, scheme="exact")
    assert used == "analytic"
    counts = pot.derivative_counts()
    assert counts["hessian_analytic"] == 1
    assert counts["hessian_ad"] == 1
    assert counts["hvp_ad"] == 1
    assert counts["hvp_analytic"] == 1
    assert pot.hvp_counts()["force_calls"] == 0


def test_auto_reports_its_fallback_and_exact_refuses() -> None:
    positions, types = _lj_cluster()
    v = _direction(len(positions))
    pot = rgpot.LJClusterPot()
    hv, used = pot.hvp_resolved(positions, types, BOX, v)
    assert used == "forward"
    np.testing.assert_array_equal(hv, pot.hvp(positions, types, BOX, v, scheme="forward"))
    assert pot.derivative_counts()["hvp_forward"] == 2
    calls = pot.hvp_counts()["force_calls"]
    for scheme in ("analytic", "ad", "exact"):
        with pytest.raises(ValueError, match="Hessian-vector product"):
            pot.hvp_resolved(positions, types, BOX, v, scheme=scheme)
    assert pot.hvp_counts()["force_calls"] == calls


def test_refused_hessian_schemes_make_no_force_call() -> None:
    positions, types = _lj_cluster()
    pot = rgpot.LJClusterPot()
    with pytest.raises(ValueError, match="no analytic Hessian"):
        pot.hessian(positions, types, BOX, "analytic")
    with pytest.raises(ValueError, match="no dual-number Hessian"):
        pot.hessian(positions, types, BOX, "ad")
    with pytest.raises(ValueError, match="neither an analytic nor a dual-number Hessian"):
        pot.hessian(positions, types, BOX, "exact")
    assert pot.hvp_counts()["force_calls"] == 0
    assert not any(pot.derivative_counts().values())


def test_the_central_hessian_costs_6n_calls_and_is_second_order() -> None:
    positions, types = _cluster()
    want = _morse_hessian(positions)
    pot = rgpot.MorsePot()
    coarse, used = pot.hessian(positions, types, BOX, "central", step=2e-2)
    assert used == "central"
    np.testing.assert_array_equal(coarse, coarse.T)
    counts = pot.derivative_counts()
    assert counts["hessian_central"] == 1
    assert counts["hessian_force_calls"] == 6 * len(positions)
    assert pot.hvp_counts()["force_calls"] == 0
    fine, _ = pot.hessian(positions, types, BOX, "central", step=1e-2)
    order = np.log2(_largest_relative(coarse, want) / _largest_relative(fine, want))
    assert order == pytest.approx(2.0, abs=0.1)

    pot.reset_hvp_counts()
    assert not any(pot.derivative_counts().values())


def test_derivative_caps_report_each_scheme_and_its_cost() -> None:
    costs = {
        "hvp_forward_force_calls": 1,
        "hvp_central_force_calls": 2,
        "hessian_central_force_calls_per_atom": 6,
    }
    flags = ("hvp_analytic", "hvp_ad", "hessian_analytic", "hessian_ad")
    expected = {
        rgpot.LJPot: (True, True, True, True),
        rgpot.MorsePot: (True, False, True, False),
        rgpot.LJClusterPot: (False, False, False, False),
    }
    for cls, bits in expected.items():
        caps = cls().derivative_caps()
        assert caps == {
            **dict(zip(flags, bits)),
            "hvp_fd": True,
            "hessian_fd": True,
            **costs,
        }


def test_malformed_hessian_requests_are_refused() -> None:
    positions, types = _cluster()
    pot = rgpot.MorsePot()
    with pytest.raises(TypeError):
        pot.hessian(positions, types, BOX)
    with pytest.raises(ValueError, match="scheme"):
        pot.hessian(positions, types, BOX, "auto")
    with pytest.raises(ValueError, match="step"):
        pot.hessian(positions, types, BOX, "central", step=-1.0)
    with pytest.raises(ValueError, match="scheme"):
        pot.hvp_resolved(positions, types, BOX, _direction(len(positions)), scheme="dense")
    assert not any(pot.derivative_counts().values())


def test_lj_coincident_atoms_raise_numerical_error_for_hessians() -> None:
    coincident = np.zeros((2, 3), dtype=np.float64)
    pot = rgpot.LJPot()
    for scheme in ("analytic", "ad", "exact"):
        with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
            pot.hessian(coincident, [1, 1], BOX, scheme)
