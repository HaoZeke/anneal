A hit on a target energy is now one-sided and inclusive everywhere it is scored.
The cluster search, the ensembles, the lattice search, the atomistic hybrid search, the examples and the benchmark scripts read ``energy <= target + tol`` through the shared rule.
A strict ``<``, a two-sided ``abs`` window and a hard-coded 1e-3 or 1e-4 no longer stand in for it.
An energy exactly at ``target + tol`` is a hit, and an energy below the target is a hit whatever the gap.
