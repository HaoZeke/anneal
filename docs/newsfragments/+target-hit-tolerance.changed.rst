Exposed ``target_tol::{DEFAULT_HIT_TOL, SEARCH_HIT_TOL, energy_hits}`` in
both ``anneal_core`` and Python ``anneal`` (``1e-3`` default, ``1e-4``
search/ensemble tolerance), standardized inclusive ``energy <= target + tol``
hit checks that reject NaN or negative tolerances, and scored cluster-search
hits with ``SEARCH_HIT_TOL``.
