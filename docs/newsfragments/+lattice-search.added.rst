``methods::lattice_search`` runs a variant of the dynamic lattice search of
Shao, Cheng and Cai, J. Comput. Chem. 25, 1693 (2004), doi 10.1002/jcc.20096,
on a Lennard-Jones cluster. Its sites are geometric, at the pair-well distance
over the cluster's own triangles, and no site is relaxed; every atom with fewer
than twelve bonds is movable; one deterministic greedy pass weighs the
highest-energy movable atoms against all sites; and one step-capped L-BFGS
quench follows each search. The lattice needs a pair potential with one well
distance.
Every value-and-gradient call is charged as one call; lattice pair terms are
charged at their fraction of the n(n-1)/2 pairs of a full evaluation and
settled in whole calls; under one call per chain can be outstanding at a first
hit or at the end of a run, while choosing a move costs at most one call.
``examples/lj_lattice_probe.rs`` reports the cost and end energies of random
starts taken down the descent.
