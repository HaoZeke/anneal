"""T4: array validation, LJ config round trips, and the F7 mapping."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import rgpot
from rgpot.ase import RgpotCalculator

# Hypothesis memoizes has_reusable_values on the process-wide strategy
# objects. Concurrent validation drops that attribute.
pytestmark = pytest.mark.thread_unsafe


def _box() -> np.ndarray:
    return np.eye(3, dtype=np.float64) * 20.0


def _types(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.int32)


@st.composite
def _two_atom_positions(
    draw: st.DrawFn, *, min_separation: float = 1.0e-6
) -> np.ndarray:
    span = st.floats(
        min_value=min_separation,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
    dx = draw(st.one_of(span, st.floats(min_value=-8.0, max_value=-min_separation)))
    dy = draw(st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False))
    dz = draw(st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False))
    return np.array([[0.0, 0.0, 0.0], [dx, dy, dz]], dtype=np.float64)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.floats(width=64, allow_nan=True, allow_infinity=True, allow_subnormal=True),
        min_size=0,
        max_size=6,
    )
)
def test_echo_f64_round_trip(values: list[float]) -> None:
    array = np.asarray(values, dtype=np.float64)
    echoed = np.asarray(rgpot._core.echo_f64(array), dtype=np.float64)
    assert echoed.tobytes() == array.tobytes()


def test_signed_zero_nan_and_subnormal() -> None:
    array = np.array(
        [-0.0, 0.0, np.nan, np.inf, np.nextafter(0.0, 1.0)],
        dtype=np.float64,
    )
    echoed = np.asarray(rgpot._core.echo_f64(array), dtype=np.float64)
    assert echoed.tobytes() == array.tobytes()


def test_wrong_dtype_and_order_are_type_errors() -> None:
    pot = rgpot.LJPot()
    box = _box()
    types = _types(2)
    float32 = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        pot.evaluate(float32, types, box)
    fortran = np.asfortranarray(np.zeros((2, 3), dtype=np.float64))
    with pytest.raises(TypeError):
        pot.evaluate(fortran, types, box)
    # int64 atom types are accepted. A floating-point type array is not.
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[1, 0] = 1.0
    pot.evaluate(positions, np.ones(2, dtype=np.int64), box)
    with pytest.raises(TypeError):
        pot.evaluate(positions, np.ones(2, dtype=np.float64), box)


def test_wrong_shape_names_both_shapes() -> None:
    pot = rgpot.LJPot()
    positions = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"positions: expected shape \(N, 3\)"):
        pot.evaluate(positions, _types(2), _box())
    positions = np.zeros((2, 3), dtype=np.float64)
    box = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=r"box: expected shape \(3, 3\), got \(2, 2\)"):
        pot.evaluate(positions, _types(2), box)


@settings(max_examples=40, deadline=None)
@given(positions=_two_atom_positions(min_separation=1.0e-4))
def test_two_atom_lj_evaluate_stays_finite_with_minimum_separation(
    positions: np.ndarray,
) -> None:
    pot = rgpot.LJPot()
    out = pot.evaluate(positions, _types(2), _box())
    assert np.isfinite(float(out["energy"]))
    forces = np.asarray(out["forces"], dtype=np.float64)
    assert np.all(np.isfinite(forces))


def test_nan_energy_is_numerical_error() -> None:
    # Coincident atoms make the LJ derivative non-finite. A NaN coordinate
    # is dropped by the neighbour list and does not reach that check.
    pot = rgpot.LJPot()
    positions = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(rgpot.NumericalError):
        pot.evaluate(positions, _types(2), _box())


def test_hessian_and_unknown_kind() -> None:
    from ase import Atoms

    calc = RgpotCalculator("lj")
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(NotImplementedError):
        calc.get_hessian(atoms)
    with pytest.raises(ValueError):
        RgpotCalculator("missing")


@settings(max_examples=20, deadline=None)
@given(
    u0=st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_subnormal=False),
    cutoff=st.floats(min_value=2.0, max_value=12.0, allow_nan=False, allow_subnormal=False),
    psi=st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_subnormal=False),
    switch_width=st.floats(
        min_value=0.0, max_value=0.4, allow_nan=False, allow_subnormal=False
    ),
)
def test_lj_config_round_trip(
    u0: float, cutoff: float, psi: float, switch_width: float
) -> None:
    pot = rgpot.LJPot(
        u0=u0, cutoff=cutoff, psi=psi, switch_width=switch_width
    )
    cfg = pot.config()
    again = rgpot.LJPot(
        u0=float(cfg["u0"]),
        cutoff=float(cfg["cutoff"]),
        psi=float(cfg["psi"]),
        switch_width=float(cfg["switch_width"]),
    )
    positions = np.array([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0]], dtype=np.float64)
    left = pot.evaluate(positions, _types(2), _box())
    right = again.evaluate(positions, _types(2), _box())
    assert float(left["energy"]) == float(right["energy"])
    assert np.asarray(left["forces"]).tobytes() == np.asarray(right["forces"]).tobytes()
