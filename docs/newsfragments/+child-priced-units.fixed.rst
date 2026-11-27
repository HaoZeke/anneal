``Ledger::settle_child`` settles the units a child ledger priced through
``Ledger::charge_frac`` onto the parent's ``Ledger::priced``, together
with its spend. Work priced inside a stall branch, a side probe or a
slice was in ``spent`` but missing from ``priced``, so a count of calls
plus ``priced`` fell short of ``spent`` for a run that settled children.
