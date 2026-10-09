``global_optimize``, ``global_optimize_objective``,
``qmc_gsa_global_search`` and ``qmc_gsa_global_search_objective`` take
``x0``. The portfolio charges it as its first evaluation and records it as
the first incumbent, so arms that read the incumbent, such as the
trust-region poll, HMC and the population arm, start from it until a lower
point is found; the GSA search uses it as its first chain's start. A
starting point outside the box, of the wrong size, or not finite raises
``ValueError``. ``portfolio_optimize_from`` and
``qmc_gsa_global_search_from`` are the Rust entry points.
