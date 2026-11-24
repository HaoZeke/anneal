``lattice::observed_basis_candidate`` builds a compact lattice
candidate from a well-conditioned Gabriel tetrahedron in the current
geometry. On a directed stall, a search whose geometry is free to move
can propose one lattice body grown from a contact tetrahedron of the
current structure. The cloud stops at as many sites as the body has
distinct 4-point frames. When that count is reached with fewer sites
than the structure has points, the stall proposes nothing. Cluster
hopping runs this probe once per basin on a side ledger, and only while
the remaining budget can pay for every distinct tetrahedron the point
count can present. It is on by default under ``Config::recommended``.
Rigid-body, molecular, frozen, and active-region configurations skip
it. Charging the side quench against the run budget moved the pinned
preset-walk and heated fingerprints in
``metropolis_heat_follows_the_plateau`` and
``lattice_exchange_follows_the_plateau`` in ``tests/plateau_heat.rs``.
