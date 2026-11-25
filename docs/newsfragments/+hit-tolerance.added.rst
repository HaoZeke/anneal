``anneal.SEARCH_HIT_TOL``, ``anneal.DEFAULT_HIT_TOL`` and
``anneal.energy_hits`` are public. They are the Python names of
``anneal_core::SEARCH_HIT_TOL``, ``anneal_core::DEFAULT_HIT_TOL`` and
``anneal_core::energy_hits``. ``energy_hits(energy, target, tol)`` is
the one-sided predicate ``energy <= target + tol``. ``tol`` defaults
to ``DEFAULT_HIT_TOL`` (``1e-3``). ``SEARCH_HIT_TOL`` is ``1e-4``.
The same three names are registered on ``anneal._core``.
