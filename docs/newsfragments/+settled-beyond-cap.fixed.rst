``Ledger::charge_many`` keeps ``Ledger::spent`` within the budget.
Settled evaluations that do not fit are counted by the new
``Ledger::settled_beyond_cap``, so ``spent`` plus ``settled_beyond_cap``
is every evaluation a side quench made, and no call goes uncounted.
