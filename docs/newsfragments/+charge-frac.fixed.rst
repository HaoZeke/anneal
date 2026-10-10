The documentation of ``Ledger::charge_frac`` says what it charges. Fractions
accumulate and are charged as a whole call each time their sum crosses one;
a residue under one call is never charged, so a ledger can report up to one
call less than the work done, mid-run and at the end, and a charge that runs
into the budget leaves the rest of its work unpaid. It had said the residue
was still owed when the run ended, and nothing collected it.
