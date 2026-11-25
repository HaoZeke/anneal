A run closed by its target or a caller stop inside a quench now reports
the lowest energy the potential returned, with its geometry. The
observed-basis probe and the auxiliary quenches recorded through
``record_quenched_answer`` used to drop that point when the certifying
gradient could not be charged on a closed run, so ``Outcome::best``
could sit above the lowest evaluated energy. Calls a side quench made
before it raised the caller stop are also settled on the run's ledger
(``Ledger::charge_many``), so ``Outcome::charged`` matches the
potential evaluations the run made.
