"""Pytest harness for the derived cost-asymmetric screen."""

from proofs.hop_cost_screen import (
    WITNESS,
    measured_hop_is_seven_fifteenths,
    quench_better_iff,
    threshold_in_unit_interval,
    threshold_symbolic,
)


def test_witness():
    assert WITNESS


def test_threshold_identity():
    ok, tau, expected = threshold_symbolic()
    assert ok
    assert tau == expected


def test_value_root():
    ok, root = quench_better_iff()
    assert ok
    assert root is not None


def test_measured_hop():
    ok, tau = measured_hop_is_seven_fifteenths()
    assert ok
    assert tau == tau.limit_denominator()


def test_unit_interval_gap():
    ok, _, _, _ = threshold_in_unit_interval()
    assert ok
