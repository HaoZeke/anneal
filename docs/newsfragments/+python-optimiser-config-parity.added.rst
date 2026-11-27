``dmc_population_optimize`` defaults (``target_n=None`` via
``recommend_target_n``, ``steps_per_control=2``, ``beta0=1.0``) now match
Rust ``dmc_population_optimize``. ``box_ensemble_optimize``,
``ensemble_optimize``, ``minimize``, and ``global_optimize`` expose the
remaining ``BoxEnsembleConfig``, ``GleEscapeConfig``, and
``BoxCoverageConfig`` parameters; ``global_optimize_objective`` accepts
``noise_sigma`` and ``policy``; ``osa_acceptance_rate`` and
``recommend_target_n`` are exported at top level; and ``BasinBias``
exposes ``w0 >= 0``, ``merge_radius``, ``height``, ``well_depth``,
``wells``, ``deepest_wells``, and ``cap_basins``.
