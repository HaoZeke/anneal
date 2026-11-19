``Ledger::hold_tail`` returns the calls it actually held, and
``Ledger::release_tail`` takes that reserve back. Releasing a later hold
no longer drops a reserve the caller did not take.
