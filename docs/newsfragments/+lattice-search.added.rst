``methods::lattice_search`` runs the dynamic lattice search of Shao, Cheng
and Cai on a Lennard-Jones cluster. The vacant sites are hollows over the
cluster's own surface triangles, the highest-energy atoms move greedily to the
lowest-energy sites, and a step-capped L-BFGS quench settles each result.
Every quench step is charged as one call and every lattice pair sum by its
fraction of a full evaluation. ``examples/lj_lattice_probe.rs`` reports the
cost and end energies of random starts taken down the descent.
