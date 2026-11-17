`fortranpots::CuH2Pot` has an analytic Hessian-vector product from the pair,
density and embedding second derivatives of the copper-hydrogen EAM, through
the new `rgpot_cuh2_hvp_ws` entry point, so `HvpScheme::Auto` and
`rgpot.CuH2Pot.hvp` make no force calls. It reports the same species and
geometry faults as the force.
