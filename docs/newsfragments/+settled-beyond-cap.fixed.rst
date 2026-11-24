``Ledger::charge_many`` keeps ``Ledger::spent`` within the budget.
Settled evaluations that do not fit are counted by the new
``Ledger::settled_beyond_cap``, so ``spent`` plus ``settled_beyond_cap``
is every evaluation a side quench made, and no call goes uncounted.
``Ledger::charged`` returns that sum, and ``Outcome::charged`` and the
other reported charges read it, so the overrun is in the number a
benchmark reads. ``Outcome::settled_beyond_cap`` and the Python
``Ledger.settled_beyond_cap`` and ``Ledger.charged`` show the part that
passed the budget. ``Ledger::settle_child`` settles a sub-ledger with its
own overrun, so the overrun reaches the top of the run. The paired check
example appends a ``beyond_cap`` column to its row.
