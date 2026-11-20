The release workflow's sdist smoke test runs ``scripts/check_installed_anneal.py``.
It fails unless ``anneal`` and ``anneal._core`` import from ``site-packages``
inside the venv the sdist was installed into. The earlier check compared the
module path with ``python/anneal/__init__.py``, which no interpreter started
at the repository root can return, so it could not fail.
