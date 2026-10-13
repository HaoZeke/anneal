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
wall, or at zero when both walls are infinite, as ``run_rs_qmc_variant``
does. On a box whose every width is finite the draws and designs are those
of before, and so are seeded runs. A NaN wall, which
``eindir_core::Bounds::new`` accepts, now makes these draws, the designs of
the polishers and of the DMC population, and ``qmc_trust_region_poll``
panic before any evaluation, naming its coordinate; the draws panicked
inside the uniform draw, the designs with
``bounds require high >= low on every axis``, and the poll evaluated its
points.
