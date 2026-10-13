The functions that took a box as it came refuse the walls they cannot work
with, before any call, with a ``ValueError`` naming the coordinate: a NaN
wall and ``low`` above ``high`` in each, an axis whose two walls are the same
infinity in ``polish``, ``shifted_qmc_polish``, ``qmc_best1bin_scout`` and
``qmc_best1bin_scout_objective``, which take other infinite walls, any
infinite wall in ``qmc_polish``, ``qmc_trust_region_poll``,
``estimate_gle_omega0``, ``gle_langevin``, ``gle_langevin_preconditioned``
and the other ``*_objective`` functions, and a wall or width that is not
finite in ``additive_independence`` and ``low_discrepancy_points``. They had
asked points at a coordinate that is not finite or outside the box, as
``gle_langevin`` did at every point it asked on ``(-inf, 0] x [0, 1]``, or
raised ``PanicException``, and ``low_discrepancy_points`` had returned such
points or refused ``low`` above ``high`` without naming the coordinate.
``PyObjective`` refuses a NaN wall of the extension's own ``Bounds``, which
eindir-core's ``Bounds::new`` takes. ``run_qmc``, ``run_device`` and
``run_ensemble``, which refused such boxes already, name the coordinate too.
