Archive search reconciles its Python charge identity after the callback
error is finished, so a ``KeyboardInterrupt`` or another ending exception
is re-raised instead of a panic. ``NoFiniteStart`` is a ``ValueError`` and
``UnnamedCharge`` is a ``RuntimeError`` on every Python path.
A search compares the screen, full and check charges it added with its own
relaxation counters, so a ledger that already holds named charges, such as
after ``polish_answer``, is not an abort.
``search_from_with_history`` returns ``SearchError::UnnamedCharge`` when the
run ends with an unnamed charge, as it does at the start.
``SearchError`` is ``#[non_exhaustive]``.
A develop harvest matches ``spent`` only when the run charges no lattice
pair rows. An atomic lattice run's ``spent`` also counts those rows.
