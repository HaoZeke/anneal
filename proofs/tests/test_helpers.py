"""Smoke tests for proofs.helpers.witness."""

import sympy as sp

from proofs.helpers import witness


def test_witness_constant_multiple_true():
    x = sp.symbols("x")
    assert witness(2 * x, x) is True


def test_witness_additive_difference_false():
    x = sp.symbols("x")
    assert witness(x + 1, x) is False


def test_witness_negative_constant_false():
    x = sp.symbols("x")
    assert witness(-x, x) is False
