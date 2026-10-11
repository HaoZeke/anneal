The ``*_objective`` functions, ``global_optimize_objective``,
``gle_langevin_objective``, ``gle_langevin_preconditioned_objective``,
``qmc_best1bin_scout_objective``, ``qmc_gsa_global_search_objective``,
``qmc_polish_objective`` and ``qmc_trust_region_poll_objective``, read the
values of a ``PyObjective`` through its ``eval`` method. An objective that
raised, or returned something that is not a number, had made them panic
with an exception that ``except Exception`` does not catch, and a NaN value
reached the driver as it was, so that ``qmc_trust_region_poll_objective``
and the GLE functions could report a NaN best value. An exception from the
objective now ends the run and is raised as itself, a value that is not a
number ends it with the ``TypeError`` ``global_optimize`` raises, and a NaN
value is scored as ``+inf``, as in ``global_optimize``.
