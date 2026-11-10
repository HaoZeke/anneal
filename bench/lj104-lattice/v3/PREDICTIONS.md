# Predictions before the serial dynamic-lattice runs

Written before any run of seeds 3000-3029 on this tip.

The search is the dynamic lattice method of Shao, Cheng and
Cai, J. Comput. Chem. 25, 1693 (2004). Vacant sites closer
than 0.05 collapse to the site nearer the cluster centroid.
Atoms are ordered by coordinates, so the search does not
read labels. The budget is 2000000 charged evaluations.

Predicted global-minimum hits, lowest-index cut: 10 of 30.
The Wilson 95% interval on 10/30 is 0.192 to 0.512,
which is 6 to 15 hits.

The uniform tie cut uses the same search. Predicted hits:
10 of 30, with the same Wilson interval, 6 to 15.

A hit is a re-quenched energy at or below -582.086641.
The re-quench asks for a gradient infinity norm of 1e-8.
These minima have stopped near a few times 1e-6, and a
further line search has not lowered that norm. The hit
test is the energy.

No comparison with the reference program is claimed here.
Those seeds have not been run.
