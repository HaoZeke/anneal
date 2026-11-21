A ``TailHold`` that went out of scope without ``Ledger::release_tail``
kept its calls hidden, so the ledger budget stayed short. Dropping the
hold now returns those calls, including on an early return or an error.
Releasing the hold and then dropping it still restores them once.
A hold is ``Send``, and holds that drop on different threads, or that
drop while the ledger grants or holds more calls, no longer lose each
other's updates. The ledger budget and its held count change together,
so no call stays hidden.
