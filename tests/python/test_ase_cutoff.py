# MIT License
# Copyright 2023--present rgpot developers
"""Periodic tiling uses the cutoff a composite potential really has.

A minimum-image kernel gives the wrong energy in a cell whose width is not
more than twice its cutoff, so the calculator tiles such a cell. It needs the
cutoff of the potential for that. ExprPot and SumPot publish theirs as the
largest cutoff of their terms, and the calculator refuses a potential whose
cutoff it cannot determine instead of assuming zero.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from ase.build import bulk

import rgpot
from rgpot.ase import RgpotCalculator
from rgpot.ase.calculator import _call, _kernel_cutoff

needs_expr = pytest.mark.skipif(
    not rgpot.has_expr,
    reason="rgpot.has_expr is False (built without -Dwith_expr)",
)

LJ_DEFAULT_CUTOFF = 15.0
MORSE_DEFAULT_CUTOFF = 9.5

# Argon-like parameters keep the energy of a rattled fcc cell well away from
# zero, so the comparison below is not decided by rounding.
AR_LJ = {"u0": 0.0104, "psi": 3.40}
AR_MORSE = {"De": 0.02, "a": 1.1, "re": 3.8}


def _lj_expr() -> Any:
    return rgpot.ExprPot("a", {"a": "lj"})


def _lj_morse_expr() -> Any:
    return rgpot.ExprPot("a + 0.5*b", {"a": "lj", "b": "morse"})


def _default_dict_expr() -> Any:
    return rgpot.ExprPot("a", {"a": {"kind": "lj", **AR_LJ}})


def _explicit_expr() -> Any:
    return rgpot.ExprPot(
        "a + b",
        {
            "a": {"kind": "lj", "cutoff": 6.0, **AR_LJ},
            "b": {"kind": "morse", "cutoff": 7.5, **AR_MORSE},
        },
    )


def _sum_shorthand() -> Any:
    return rgpot.SumPot(["lj", {"kind": "morse", "weight": -0.5}])


def _sum_explicit() -> Any:
    return rgpot.SumPot(
        [
            {"kind": "lj", "cutoff": 5.5, **AR_LJ},
            {"kind": "morse", "cutoff": 8.0, "weight": 0.5, **AR_MORSE},
        ]
    )


def _lj_object() -> Any:
    return rgpot.LJPot(cutoff=9.0, **AR_LJ)


def _morse_object() -> Any:
    return rgpot.MorsePot(cutoff=8.5, **AR_MORSE)


def _gromacs_object() -> Any:
    return rgpot.rgpot_gromacs_potential()


EXPR_POTENTIALS = [
    pytest.param(_lj_expr, LJ_DEFAULT_CUTOFF, id="expr-shorthand-lj"),
    pytest.param(
        _lj_morse_expr, LJ_DEFAULT_CUTOFF, id="expr-shorthand-lj-morse"
    ),
    pytest.param(
        _default_dict_expr, LJ_DEFAULT_CUTOFF, id="expr-default-cutoff-term"
    ),
    pytest.param(_explicit_expr, 7.5, id="expr-explicit-cutoffs"),
]

OBJECT_POTENTIALS = [
    pytest.param(_sum_shorthand, LJ_DEFAULT_CUTOFF, id="sum-shorthand"),
    pytest.param(_sum_explicit, 8.0, id="sum-explicit"),
    pytest.param(_lj_object, 9.0, id="lj-object"),
    pytest.param(_morse_object, 8.5, id="morse-object"),
    pytest.param(_gromacs_object, 8.0, id="gromacs-object"),
]


def _short_cells() -> list[Any]:
    cubic = bulk("Ar", "fcc", a=5.26, cubic=True)
    cubic.rattle(0.05, seed=7)
    hexagonal = bulk("Ar", "hcp", a=3.76)
    hexagonal.rattle(0.05, seed=11)
    return [pytest.param(cubic, id="cubic"), pytest.param(hexagonal, id="hexagonal")]


def _perpendicular_widths(cell: np.ndarray) -> np.ndarray:
    volume = abs(np.linalg.det(cell))
    return np.array(
        [
            volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
            for i in range(3)
        ]
    )


def _reference_repeat(atoms: Any, cutoff: float) -> tuple[Any, int]:
    """A supercell whose every width clears twice the cutoff with room to spare."""
    widths = _perpendicular_widths(atoms.cell.array)
    reps = [int(math.ceil(2.0 * cutoff / w)) + 1 for w in widths]
    big = atoms.repeat(reps)
    assert np.all(_perpendicular_widths(big.cell.array) > 2.0 * cutoff)
    return big, int(np.prod(reps))


def _check_against_supercell(
    potential: Any, atoms: Any, expected_cutoff: float
) -> None:
    calc = RgpotCalculator(potential)
    assert calc._cutoff == pytest.approx(expected_cutoff)
    widths = _perpendicular_widths(atoms.cell.array)
    assert np.any(widths <= 2.0 * expected_cutoff), "the cell must be short"

    small = atoms.copy()
    small.calc = calc
    energy = small.get_potential_energy()
    forces = small.get_forces()

    geometry = calc._geometry(
        atoms.positions, atoms.numbers, atoms.cell.array, atoms.pbc
    )
    assert geometry.copies > 1

    big, copies = _reference_repeat(atoms, expected_cutoff)
    raw = _call(
        potential,
        np.ascontiguousarray(big.positions),
        big.numbers.astype(np.int32),
        np.ascontiguousarray(big.cell.array),
    )
    assert energy / len(atoms) == pytest.approx(
        raw["energy"] / len(big), rel=1e-10, abs=1e-14
    )
    np.testing.assert_allclose(
        forces, np.asarray(raw["forces"])[: len(atoms)], rtol=0.0, atol=1e-10
    )
    assert float(np.max(np.abs(forces))) > 1e-6, "the forces must be non-trivial"

    reference_calc = RgpotCalculator(potential)
    big.calc = reference_calc
    assert reference_calc._geometry(
        big.positions, big.numbers, big.cell.array, big.pbc
    ).copies == 1
    assert big.get_potential_energy() == pytest.approx(raw["energy"], rel=1e-12)
    if "stress" in calc.implemented_properties:
        np.testing.assert_allclose(
            small.get_stress(), big.get_stress(), rtol=0.0, atol=1e-10
        )

    # Without tiling the kernel folds each pair to its nearest image only, so
    # the short cell alone does not give this energy. This is what a cutoff of
    # zero used to return.
    untiled = _call(
        potential,
        np.ascontiguousarray(atoms.positions),
        atoms.numbers.astype(np.int32),
        np.ascontiguousarray(atoms.cell.array),
    )
    assert abs(untiled["energy"] / len(atoms) - raw["energy"] / len(big)) > 1e-6


@needs_expr
@pytest.mark.parametrize(("build", "cutoff"), EXPR_POTENTIALS)
@pytest.mark.parametrize("atoms", _short_cells())
def test_expr_pot_object_tiles_by_its_term_cutoff(
    build: Any, cutoff: float, atoms: Any
) -> None:
    _check_against_supercell(build(), atoms, cutoff)


@pytest.mark.parametrize(("build", "cutoff"), OBJECT_POTENTIALS)
@pytest.mark.parametrize("atoms", _short_cells())
def test_direct_objects_tile_by_their_cutoff(
    build: Any, cutoff: float, atoms: Any
) -> None:
    _check_against_supercell(build(), atoms, cutoff)


@needs_expr
def test_expr_kind_string_and_object_agree() -> None:
    terms = {"a": "lj", "b": {"kind": "morse", "cutoff": 7.0, **AR_MORSE}}
    by_kind = RgpotCalculator("expr", expression="a + b", terms=terms)
    by_object = RgpotCalculator(rgpot.ExprPot("a + b", terms))
    assert by_kind._cutoff == by_object._cutoff == LJ_DEFAULT_CUTOFF

    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    atoms.rattle(0.05, seed=3)
    other = atoms.copy()
    atoms.calc = by_kind
    other.calc = by_object
    assert atoms.get_potential_energy() == other.get_potential_energy()
    np.testing.assert_array_equal(atoms.get_forces(), other.get_forces())


@needs_expr
def test_expr_pot_cutoff_is_the_largest_term_cutoff() -> None:
    assert _lj_expr().cutoff == LJ_DEFAULT_CUTOFF
    assert rgpot.ExprPot("a", {"a": "morse"}).cutoff == MORSE_DEFAULT_CUTOFF
    assert _lj_morse_expr().cutoff == LJ_DEFAULT_CUTOFF
    assert _default_dict_expr().cutoff == LJ_DEFAULT_CUTOFF
    assert _explicit_expr().cutoff == 7.5
    shrunk = rgpot.ExprPot(
        "a + b", {"a": {"kind": "lj", "cutoff": 4.0}, "b": "morse"}
    )
    assert shrunk.cutoff == MORSE_DEFAULT_CUTOFF
    # A term that sums every image itself needs no wider cell.
    assert rgpot.ExprPot("a", {"a": "d2"}).cutoff == 0.0
    assert rgpot.ExprPot("a + b", {"a": "d2", "b": "lj"}).cutoff == LJ_DEFAULT_CUTOFF


def test_sum_pot_cutoff_counts_only_enabled_terms() -> None:
    assert _sum_shorthand().cutoff == LJ_DEFAULT_CUTOFF
    assert _sum_explicit().cutoff == 8.0
    off = rgpot.SumPot(
        [
            {"kind": "morse", "cutoff": 4.0},
            {"kind": "lj", "cutoff": 40.0, "enabled": False},
        ]
    )
    assert off.cutoff == 4.0
    assert rgpot.SumPot([{"kind": "lj", "enabled": False}]).cutoff == 0.0


def test_unknown_sumpot_term_kind_is_an_error_not_a_zero_cutoff() -> None:
    with pytest.raises(ValueError, match="unknown child kind: nope"):
        rgpot.SumPot(["nope"])


@needs_expr
def test_unknown_expr_term_kind_is_an_error_not_a_zero_cutoff() -> None:
    with pytest.raises(ValueError, match="unknown child kind: nope"):
        rgpot.ExprPot("a", {"a": "nope"})


class _NoCutoff:
    """A callable potential that publishes no cutoff in any form."""

    def evaluate(self, positions, numbers, box):
        raise AssertionError("never evaluated")


class _BadCutoff(_NoCutoff):
    def __init__(self, value: float) -> None:
        self.cutoff = value


def test_potential_without_a_cutoff_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot determine the cutoff of _NoCutoff"):
        RgpotCalculator(_NoCutoff())
    with pytest.raises(ValueError, match="cannot determine the cutoff"):
        _kernel_cutoff(_NoCutoff())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_unusable_cutoff_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and not negative"):
        RgpotCalculator(_BadCutoff(value))


def test_every_advertised_kind_resolves_a_cutoff() -> None:
    from rgpot.ase.calculator import _potential_factories

    seen = 0
    for kind in sorted(_potential_factories()):
        try:
            calc = RgpotCalculator(kind)
        except (TypeError, ValueError, RuntimeError) as err:
            if "cannot determine the cutoff" in str(err):
                raise
            continue  # a kind that needs constructor arguments
        assert math.isfinite(calc._cutoff) and calc._cutoff >= 0.0
        seen += 1
    assert seen > 0
