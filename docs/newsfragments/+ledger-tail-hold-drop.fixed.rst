A ``TailHold`` that went out of scope without ``Ledger::release_tail``
kept its calls hidden, so the ledger budget stayed short. Dropping the
hold now returns those calls, including on an early return or an error.
Releasing the hold and then dropping it still restores them once.
