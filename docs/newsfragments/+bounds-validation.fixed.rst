Nine search functions panicked when the box was bad, which raised a
``PanicException`` that ``except Exception`` does not catch.
``additive_independence``, ``estimate_gle_omega0``, ``gle_langevin``,
``gle_langevin_preconditioned``, ``qmc_best1bin_scout``, ``qmc_polish``,
``qmc_trust_region_poll`` and ``shifted_qmc_polish`` panicked on a
``low`` above ``high``, and ``low_discrepancy_points``,
``qmc_best1bin_scout``, ``qmc_polish`` and ``shifted_qmc_polish``
panicked on a NaN bound. All nine now raise ``BoundsError``, and the five
that took a NaN bound without complaint (``additive_independence``,
``estimate_gle_omega0``, ``gle_langevin``, ``gle_langevin_preconditioned``
and ``qmc_trust_region_poll``) reject it too. ``BasinBias.potential`` and
``deposit`` raise ``InvalidArgumentError`` when ``x`` does not have
``3 * n_points`` entries, where they panicked on an index.
