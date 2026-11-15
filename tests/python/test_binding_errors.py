"""Error kinds of the Python entry points follow the binding contract."""

from __future__ import annotations

import numpy as np
import pytest

import rgpot

BOX = np.eye(3) * 20.0
DIMER = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])


@pytest.mark.parametrize(
    "positions, box",
    [
        (DIMER.astype(np.float32), BOX),
        (np.asfortranarray(np.vstack([DIMER, [[0.0, 1.5, 0.0]]])), BOX),
        (DIMER, BOX.astype(np.float32)),
        (DIMER, BOX.T),
    ],
    ids=["float32-positions", "fortran-positions", "float32-box", "strided-box"],
)
def test_evaluate_batch_wrong_dtype_or_order_is_type_error(positions, box) -> None:
    lj = rgpot.LJPot()
    types = [1] * positions.shape[0]
    with pytest.raises(TypeError, match="evaluate_batch"):
        lj.evaluate_batch([positions], [types], [box])


def test_stillinger_weber_call_raises_numerical_error() -> None:
    # Two atoms 1e-80 Angstrom apart: the pair term overflows to +inf.
    positions = np.array([[0.0, 0.0, 0.0], [1e-80, 0.0, 0.0]])
    with pytest.raises(rgpot.NumericalError, match="not finite"):
        rgpot.StillingerWeberSi()(positions, [14, 14], BOX)


@pytest.mark.parametrize("cls", ["StillingerWeberSi", "TersoffSiC"])
def test_call_matches_evaluate(cls: str) -> None:
    pot = getattr(rgpot, cls)()
    positions = np.array([[0.0, 0.0, 0.0], [2.3, 0.0, 0.0], [0.0, 2.3, 0.0]])
    energy, forces, _ = pot(positions, [14, 14, 14], BOX)
    assert np.isfinite(energy)
    if hasattr(pot, "evaluate"):
        row = pot.evaluate(positions, [14, 14, 14], BOX)
        assert row["energy"] == energy
        np.testing.assert_array_equal(row["forces"], forces)


@pytest.mark.skipif(not rgpot.has_dftd3, reason="built without s-dftd3")
@pytest.mark.parametrize("damping", ["bj", "BJ", "zero", "Zero"])
def test_d3_documented_damping_spellings(damping: str) -> None:
    rgpot.D3Pot(damping=damping)


@pytest.mark.skipif(not rgpot.has_dftd3, reason="built without s-dftd3")
@pytest.mark.parametrize("damping", ["rational", "bj ", "ZERO", ""])
def test_d3_unknown_damping_is_value_error(damping: str) -> None:
    with pytest.raises(ValueError, match="D3 damping"):
        rgpot.D3Pot(damping=damping)
