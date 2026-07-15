"""Pytest for D10 win objective."""

from proofs.d10_portfolio_win_objective import (
    WITNESS,
    argmax_win,
    check_argmax_full_budget_when_conversion_needs_all,
    check_argmax_zero_when_already_converted,
    check_no_exploration_is_q0,
    win_objective_discovery,
)


def test_witness_beta_product():
    assert WITNESS


def test_e0():
    assert check_no_exploration_is_q0()


def test_argmax_boundaries():
    assert check_argmax_full_budget_when_conversion_needs_all()
    assert check_argmax_zero_when_already_converted()


def test_win_monotone_in_p_conv():
    w_lo = win_objective_discovery(50, 20, 10, 0.1, 0.2)
    w_hi = win_objective_discovery(50, 20, 10, 0.1, 0.9)
    assert w_hi > w_lo
