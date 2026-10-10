Dynamic lattice search (``methods::lattice_search``) keeps two readers of a
pair cluster. One builds hollow sites over the triangles and squares of the
current structure, grows an interior lattice that carries both stackings of
every facet, places the surface around the kept interior, and walks
occupations until no relocation lowers a point. The ``dls`` driver token
uses it, and the restoration tests cover the LJ98 and LJ104 minima.
The other is a variant of the dynamic lattice search of Shao, Cheng and Cai,
J. Comput. Chem. 25, 1693 (2004), doi 10.1002/jcc.20096. Its sites sit at the
pair-well distance over the cluster's own triangles and are not relaxed;
every atom with fewer than twelve bonds is movable; one deterministic greedy
pass weighs the highest-energy movable atoms against all sites; and one
L-BFGS quench in the step-capped form used by GMIN follows each search. The
lattice needs a pair potential with one well distance. Every value-and-gradient
call is charged as one call. Lattice pair terms are charged at their fraction
of the n(n-1)/2 pairs of a full evaluation and settled in whole calls. Under
one call per chain can be outstanding at a first hit or at the end of a run,
while choosing a move costs at most one call. ``examples/lj_lattice_probe.rs``
reports the cost and end energies of random starts taken down the descent.
