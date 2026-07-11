"""Tests for D5's geometric-contraction reserve."""

from proofs.d5_endgame_switch import WITNESS, polish_steps_required


def test_witness():
    assert WITNESS


def test_polish_requirement_is_zero_for_an_already_solved_gap():
    assert polish_steps_required(gap=1e-4, tolerance=1e-3, contraction=0.5) == 0


def test_polish_requirement_is_the_minimal_nonnegative_integer():
    assert polish_steps_required(gap=1.0, tolerance=0.1, contraction=0.5) == 4
