``box_ensemble_optimize`` and ``ensemble_optimize`` take every setting of
the Rust ``BoxEnsembleConfig``, ``GleEscapeConfig`` and
``BoxCoverageConfig``: ``identity_tol``, ``shared_deposits``,
``shared_visit_policy``, ``escape="langevin"`` with ``langevin_steps``,
``langevin_omega0``, ``langevin_dt`` and ``langevin_friction``, and
``coverage_height``, ``coverage_well_tempering`` and
``coverage_peer_weight`` beside the existing coverage keywords. A value
that Rust asserts on raises ``InvalidArgumentError``. The result of
``box_ensemble_optimize`` also reports ``charged``.
``pytest/test_box_ensemble_tuning.py`` fails when a field of the three
structs has no keyword.
