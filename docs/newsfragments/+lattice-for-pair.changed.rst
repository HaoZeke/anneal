``Lattice::for_pair`` sets bond, hollow, site, clearance and merge
distances to 1.2, 1.47, 1, 0.85 and 0.25 times ``PairKind::r_min``.
``interior_coordination`` stays a field. ``Lattice::default`` is
``for_pair(LennardJones)`` and matches the previous hard-coded well
bit for bit. ``lattice_ensemble::run`` takes that pair kind.
