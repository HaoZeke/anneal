"""rgpot.SumPot: weights, switches and the per-term record."""

from __future__ import annotations

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def cluster():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.7, 0.2]], dtype=np.float64
    )
    atom_types = np.array([18, 18, 18], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    return positions, atom_types, box


def test_weighted_terms_sum_and_are_recorded():
    pos, z, box = cluster()
    e_lj, f_lj, _ = rgpot.LJPot()(pos, z, box)
    e_m, f_m, _ = rgpot.MorsePot()(pos, z, box)
    pot = rgpot.SumPot(
        [
            {"kind": "lj", "name": "lj", "weight": 2.0},
            {"kind": "morse", "name": "morse", "weight": -0.5},
        ]
    )
    e, f, _ = pot(pos, z, box)
    assert e == pytest.approx(2.0 * e_lj - 0.5 * e_m, rel=1e-13, abs=1e-14)
    np.testing.assert_allclose(f, 2.0 * f_lj - 0.5 * f_m, rtol=1e-13, atol=1e-14)
    parts = pot.components()
    assert [p["name"] for p in parts] == ["lj", "morse"]
    assert [p["weight"] for p in parts] == [2.0, -0.5]
    assert parts[0]["energy"] == pytest.approx(e_lj, rel=1e-14)
    np.testing.assert_allclose(parts[1]["forces"], f_m, rtol=1e-14, atol=1e-15)


def test_negative_weight_cancels_and_disabled_term_is_skipped():
    pos, z, box = cluster()
    cancel = rgpot.SumPot(["lj", {"kind": "lj", "weight": -1.0}])
    e, f, _ = cancel(pos, z, box)
    assert e == 0.0
    assert np.all(f == 0.0)
    assert [p["name"] for p in cancel.components()] == ["term0", "term1"]

    off = rgpot.SumPot(["lj", {"kind": "morse", "enabled": False}])
    e_off, _, _ = off(pos, z, box)
    assert e_off == rgpot.LJPot()(pos, z, box)[0]
    parts = off.components()
    assert parts[1]["enabled"] is False
    assert parts[1]["evaluated"] is False
    assert parts[1]["forces"].shape == (0, 3)
    assert parts[1]["stress"] is None


def test_disabled_term_is_never_built():
    pos, z, box = cluster()
    pot = rgpot.SumPot(["lj", {"kind": "nonesuch", "enabled": False}])
    assert pot(pos, z, box)[0] == rgpot.LJPot()(pos, z, box)[0]
    with pytest.raises(ValueError, match="nonesuch"):
        rgpot.SumPot(["lj", "nonesuch"])(pos, z, box)


def test_caps_and_stress_follow_the_terms():
    pos, z, box = cluster()
    pot = rgpot.SumPot(["lj", {"kind": "morse", "weight": 3.0}])
    caps = pot.caps()
    assert caps["stress"] is True
    assert caps["reentrancy"] == "per_instance"
    out = pot.evaluate(pos, z, box)
    lj = rgpot.LJPot().evaluate(pos, z, box)
    morse = rgpot.MorsePot().evaluate(pos, z, box)
    np.testing.assert_allclose(
        out["stress"], lj["stress"] + 3.0 * morse["stress"], rtol=1e-13, atol=1e-16
    )
    np.testing.assert_allclose(pot.components()[1]["stress"], morse["stress"], rtol=1e-14)


def test_empty_sum_is_zero():
    pos, z, box = cluster()
    e, f, _ = rgpot.SumPot([])(pos, z, box)
    assert e == 0.0
    assert np.all(f == 0.0)
