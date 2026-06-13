"""Pytest harness for the manuscript theorems plus the sign-convention
negative test for Theorem 1."""

import pytest

from proofs.thm1_bsa_visit import WITNESS as W1, witness_for
from proofs.thm2_fsa_visit import WITNESS as W2
from proofs.thm3_metropolis  import WITNESS as W3
from proofs.thm4_log_cool    import WITNESS as W4
from proofs.thm5_visit_multivariate import WITNESS as W5


@pytest.mark.parametrize("name, witness", [
    ("thm1_bsa_visit", W1),
    ("thm2_fsa_visit", W2),
    ("thm3_metropolis", W3),
    ("thm4_log_cool",   W4),
    ("thm5_visit_multivariate", W5),
])
def test_theorem(name: str, witness: bool):
    assert witness, f"{name} symbolic identity does not hold"


@pytest.mark.parametrize("form, should_match", [
    ("denominator", True),
    ("numerator",  False),
])
def test_thm1_sign_convention(form, should_match):
    assert witness_for(form) is should_match
