``ras=True`` raises ``ValueError`` when the opening quench is not finite.
An objective that is never finite used to abort that call. The other
search path already raised ``ValueError`` for the same miss. Later hops
on the archive walk return the same error, and evaluations already
spent stay on the ledger.
