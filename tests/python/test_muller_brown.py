# Copyright 2023--present rgpot developers

import numpy as np
import pytest

import rgpot

POSITIONS = np.array([[0.2, 0.3, 0.0]])
ATOM_TYPES = np.array([1])
BOX = np.zeros((3, 3))


def test_muller_brown_binding_uses_first_atom_xy() -> None:
    pot = rgpot.MullerBrownPot()
    positions = np.array([[-0.558, 1.442, 9.0], [4.0, 5.0, 6.0]])
    energy, forces, _ = pot(positions, np.array([1, 1]), np.zeros((3, 3)))
    assert energy == pytest.approx(-146.70, abs=0.01)
    assert forces[0, 2] == 0.0
    assert np.array_equal(forces[1], np.zeros(3))


def test_muller_brown_binding_parameters_are_overridable() -> None:
    defaults = rgpot.MullerBrownPot()
    scaled = rgpot.MullerBrownPot(amplitude=[-400, -200, -340, 30])
    positions = np.array([[0.2, 0.3, 0.0]])
    atom_types = np.array([1])
    box = np.zeros((3, 3))
    base_energy, base_forces, _ = defaults(positions, atom_types, box)
    energy, forces, _ = scaled(positions, atom_types, box)
    assert energy == pytest.approx(2.0 * base_energy)
    assert forces == pytest.approx(2.0 * base_forces)


def test_muller_brown_binding_empty_lists_select_the_defaults() -> None:
    defaults = rgpot.MullerBrownPot()
    empty = rgpot.MullerBrownPot(amplitude=[], a=[], b=[], c=[], x0=[], y0=[])
    assert empty.config() == defaults.config()
    positions = np.array([[0.2, 0.3, 0.0]])
    atom_types = np.array([1])
    box = np.zeros((3, 3))
    base_energy, base_forces, _ = defaults(positions, atom_types, box)
    energy, forces, _ = empty(positions, atom_types, box)
    assert energy == base_energy
    assert np.array_equal(forces, base_forces)


def test_muller_brown_binding_empty_list_overrides_only_the_others() -> None:
    defaults = rgpot.MullerBrownPot().config()
    mixed = rgpot.MullerBrownPot(amplitude=[], x0=[0.0, 0.5, 1.0, 1.5]).config()
    assert mixed["amplitude"] == defaults["amplitude"]
    assert mixed["x0"] == [0.0, 0.5, 1.0, 1.5]


def test_muller_brown_binding_rejects_other_lengths() -> None:
    with pytest.raises(ValueError, match="four values"):
        rgpot.MullerBrownPot(a=[1.0, 2.0, 3.0])


def test_muller_brown_constructions_keep_nothing_from_earlier_ones() -> None:
    defaults = rgpot.MullerBrownPot()
    published = defaults.config()
    base_energy, base_forces, _ = defaults(POSITIONS, ATOM_TYPES, BOX)

    amplitude = [-400.0, -200.0, -340.0, 30.0]
    x0 = [2.0, 1.0, 0.0, -1.0]
    override = rgpot.MullerBrownPot(amplitude=amplitude, x0=x0)
    assert override.config()["amplitude"] == amplitude
    assert override.config()["x0"] == x0
    assert override.config()["a"] == published["a"]
    override_energy, _, _ = override(POSITIONS, ATOM_TYPES, BOX)
    assert override_energy != base_energy

    empty = rgpot.MullerBrownPot(amplitude=[], a=[], b=[], c=[], x0=[], y0=[])
    assert empty.config() == published
    energy, forces, _ = empty(POSITIONS, ATOM_TYPES, BOX)
    assert energy == base_energy
    assert np.array_equal(forces, base_forces)

    a = [-2.0, -2.0, -13.0, 1.4]
    partial = rgpot.MullerBrownPot(amplitude=[], a=a)
    expected = dict(published, a=a)
    assert partial.config() == expected
    partial_energy, _, _ = partial(POSITIONS, ATOM_TYPES, BOX)
    assert partial_energy != base_energy

    again = rgpot.MullerBrownPot(a=[], x0=[])
    assert again.config() == published

    assert override.config()["amplitude"] == amplitude
    assert partial.config() == expected
    assert override(POSITIONS, ATOM_TYPES, BOX)[0] == override_energy
