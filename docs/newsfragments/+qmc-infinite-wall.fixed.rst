``run_rs_qmc_variant`` and ``run_rs_qmc_variant_from`` start every chain but
``x0`` on the finite wall of an axis with an infinite wall, or at 0 when both
walls are infinite. Those starts used to be infinite, and such a chain
evaluated the objective only at infinity. Python's ``run_qmc`` refuses such
bounds and is unaffected.
