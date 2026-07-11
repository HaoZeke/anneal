"""Tests for D7's finite-horizon escape calculation."""

from proofs.d7_budgeted_escape_window import (
    WITNESS,
    escape_probability,
    expected_escape_time,
)


def test_witness():
    assert WITNESS


def test_escape_probability_is_finite_horizon_absorption_probability():
    short = escape_probability(m=6, b=2.0, drop=2.0, temp=0.5, budget=20)
    long = escape_probability(m=6, b=2.0, drop=2.0, temp=0.5, budget=2000)

    assert 0.0 <= short < long <= 1.0


def test_expected_escape_time_is_positive():
    assert expected_escape_time(m=6, b=2.0, drop=2.0, temp=0.5) > 0.0
