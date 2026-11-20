``energy_hits(energy, target, tol)`` is the one rule for a search reaching a target energy: ``energy <= target + tol``.
A non-finite energy, target or tolerance, or a negative tolerance, is never a hit.
``DEFAULT_HIT_TOL`` is 1e-3 and ``SEARCH_HIT_TOL`` is 1e-4, the window of the solved flag, the first encounter and the ensemble discovery level.
Rust exposes all three at the crate root and Python exposes them as ``anneal.energy_hits``, ``anneal.DEFAULT_HIT_TOL`` and ``anneal.SEARCH_HIT_TOL``.
``scripts/target_tol.py`` holds the same rule for the benchmark scripts.
