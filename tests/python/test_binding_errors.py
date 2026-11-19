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


def test_lj_coincident_atoms_raise_numerical_error_everywhere() -> None:
    from ase import Atoms
    from rgpot.ase import RgpotCalculator

    coincident = np.zeros((2, 3), dtype=np.float64)
    direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    pot = rgpot.LJPot()

    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.evaluate(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.evaluate_per_atom(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.evaluate_batch([coincident], [[1, 1]], [BOX])
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.hvp(coincident, [1, 1], BOX, direction, scheme="analytic")
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.hvp(coincident, [1, 1], BOX, direction, scheme="forward")
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        pot.hvp_centre(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        rgpot.evaluate_lj(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        rgpot.rgpot_gromacs_potential()(coincident, [1, 1], BOX)
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        rgpot.rgpot_openmm_potential()(coincident, [1, 1], BOX)

    calc = RgpotCalculator("lj")
    atoms = Atoms("H2", positions=coincident, cell=BOX, pbc=True)
    atoms.calc = calc
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        atoms.get_potential_energy()
    with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
        calc.calculate_many([atoms])

    if rgpot.has_expr:
        expr = rgpot.ExprPot("a", {"a": "lj"})
        with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
            expr(coincident, [1, 1], BOX)
        with pytest.raises(rgpot.NumericalError, match="coincident atoms"):
            expr.evaluate(coincident, [1, 1], BOX)
