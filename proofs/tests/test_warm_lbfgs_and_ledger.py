from proofs.ledger_charge_invariant import WITNESS as LEDGER_OK
from proofs.warm_lbfgs_wolfe import WITNESS as WOLFE_OK


def test_wolfe():
    assert WOLFE_OK


def test_ledger():
    assert LEDGER_OK
