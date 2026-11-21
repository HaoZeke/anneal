``SEARCH_HIT_TOL`` is ``1e-4``. The solved flag, the first encounter and
the ensemble discovery level use it, in Rust and in the Python drivers.
``DEFAULT_HIT_TOL`` stays ``1e-3``. A best in ``(reference + 1e-4,
reference + 1e-3]`` is no longer a discovery or a first encounter.
