On a box whose width ``high - low`` overflows ``f64``, such as
``[-1e308, 1e308]``, or that has an infinite wall, the start that
``HmcSaSampler``, ``NutsSaSampler`` and the classical ``SaVariant`` chains
draw when none is given panicked inside the uniform draw, and the
low-discrepancy designs of ``qmc_best1bin_scout``,
``qmc_gsa_global_search``, ``qmc_trust_region_poll``,
``shifted_qmc_projected_gradient_polish`` and the DMC population, and on
the first kind of box those of the portfolio, did not spread across it:
their points lay on a wall, at an infinite coordinate or, for the DMC
population on a box with an infinite lower wall, all at ``-f64::MAX``. They
now draw and lay out such an axis at half its width, doubled, so the points
spread across it, and start an axis with an infinite wall on its finite
wall, or at zero when its walls are ``-inf`` and ``inf``, as
``run_rs_qmc_variant`` does. On a box whose every width is finite this
changes no draw, design or seeded run. A NaN wall, and an
axis whose two walls are the same infinity, such as ``[inf, inf]``, which
``eindir_core::Bounds::new`` accepts though no point is inside, now make
these draws, the designs of the polishers and of the DMC population,
``qmc_trust_region_poll`` and the starts of ``run_rs_qmc_variant`` panic
before any evaluation, naming the coordinate. The draws had panicked inside
the uniform draw, except that the classical chains started such an axis at
its infinity, the designs with ``bounds require high >= low on every axis``,
and the poll and ``run_rs_qmc_variant`` evaluated points outside the box.
