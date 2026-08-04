"""Tests for D14 screening and costly bias."""

from proofs.d14_screen_bias_force import (
    WITNESS,
    eta_s1,
    feasible_costly_bias,
    force_ratio_s2,
    kappa,
    screen_s2_helps,
)


def test_witness():
    assert WITNESS


def test_eta_s1():
    assert abs(eta_s1(0.5, 100.0, 1.0) - (1 - 50 / 101)) < 1e-12


def test_s2_delta0_never_helps():
    assert not screen_s2_helps(0.5, 0.0, 100.0, 1.0)
    assert force_ratio_s2(0.5, 0.0, 100.0, 1.0) > 1.0


def test_costly_blocks_or_cheap_opens():
    b, g, d, B = 8.69, 1.21, 225, 3e6
    ok_cheap, _, _ = feasible_costly_bias(b, g, d, B, c0=1e-6)
    ok_dear, _, phi = feasible_costly_bias(b, g, d, B, c0=1e5)
    assert ok_cheap
    assert not ok_dear
    assert phi < kappa(b, g, d)
