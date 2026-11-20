The CI sdist job runs ``scripts/check_installed_anneal.py`` before pytest,
as the release workflow does. The earlier check compared the module path
with ``python/anneal/__init__.py``, which an install and a shadowing
package both fail to match, so it could not fail.
