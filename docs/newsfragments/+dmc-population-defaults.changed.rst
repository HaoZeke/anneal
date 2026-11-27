``dmc_population_optimize`` no longer defaults ``target_n`` to 16 or
``steps_per_control`` to 4. Omitting ``target_n`` now follows
``recommend_target_n(budget, dim)``, which is between 8 and 36 and at
most ``budget / 6``. ``steps_per_control`` defaults to 2. ``beta0`` is
keyword-only and still defaults to 1. A caller who relied on a
population of 16 and 4 steps between controls gets a different
schedule.
