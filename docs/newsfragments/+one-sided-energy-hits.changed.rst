``energy_hits`` is the one-sided rule ``energy <= target + tol``.
Non-finite values and a negative tolerance are not hits.
``DEFAULT_HIT_TOL`` is ``1e-3``. Rust and Python both export the
predicate. The occupancy score uses that default where it compared
against ``1e-4``.
