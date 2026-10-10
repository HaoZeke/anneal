In values-only portfolio runs the GSA arm runs one chain from 10 dimensions,
and below that one for every eight evaluations of its first slice, between two
and four. The first chain starts at the incumbent, as dual_annealing starts
from ``x0``, and the others at seeded low-discrepancy points; once another arm
has lowered the incumbent below every chain, the worst chain goes on from
there. Chains run quenched, without a local search: each temperature step
visits every coordinate once, with dual_annealing's visiting distribution
cooling from box-wide steps to local ones, and keeps only improvements.
Without a local search, Metropolis moves at dual_annealing's temperatures
spent most of a short budget uphill, and on separable multimodal objectives
(shifted Rastrigin, Styblinski-Tang, Levy) a one-coordinate visit drops that
coordinate into a lower basin where a visit to every coordinate at once almost
never improves a good point. The DE arm's population starts with the
incumbent, as SciPy's differential_evolution places ``x0``, and its first slice
evolves the population once it is seeded.
