``tests/target_tol_lint.rs`` and ``scripts/target_tol_lint_scan.py`` fail on a raw comparison with a target energy in Rust, Python, shell and Fortran sources, so every hit goes through ``energy_hits``.
``pytest/test_dead_priors.py`` covers the refusal of the removed prior keys and switches in JSON, TOML, serialised ``ContinuousSymmetry`` and the driver environments.
``pytest/test_target_tol.py`` checks the Python and Fortran forms of the rule against the Rust one.
