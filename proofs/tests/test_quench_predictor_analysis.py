"""TOMS-style checks for the shipped QuenchPredictor math."""

from proofs import quench_predictor_analysis as qpa


def test_witness():
    assert qpa.WITNESS


def test_each_named_check():
    for name, v in qpa.all_checks():
        assert v, name