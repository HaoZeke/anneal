"""Pytest for D9 Good-Turing × record."""

from proofs.d9_good_turing_record import (
    WITNESS,
    check_monotone_in_singletons,
    check_numeric_examples,
    check_record_exchangeability,
    discovery_value,
)


def test_witness():
    assert WITNESS


def test_record_exchangeability():
    assert check_record_exchangeability(6)


def test_numeric_examples():
    assert check_numeric_examples()


def test_monotone():
    assert check_monotone_in_singletons()


def test_matches_portfolio_unit():
    # Same numbers as basin_registry_good_turing_record_gate
    assert abs(discovery_value(1, 6, 3) - (1.0 / 6.0) / 4.0) < 1e-15
