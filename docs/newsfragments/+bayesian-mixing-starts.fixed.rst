``BayesianMixingSampler`` starts its chains as ``run_rs_qmc_variant`` does, on
the finite wall of an axis with an infinite wall and at half scale on an axis
whose width overflows. Those starts used to be infinite or on ``high``.
