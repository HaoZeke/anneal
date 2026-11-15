The C++ potentials that share the neighbour list (LJ, Morse, Sutton-Chen,
ZBL, EAM, Stillinger-Weber, Tersoff) are several times faster on a sheared
or otherwise non-orthogonal periodic cell, with bit-identical results. The
minimum image of a pair no longer searches the nearby lattice images when
the rounded image is provably the shortest, and the neighbour list drops a
pair without that search when every image lies beyond the cutoff plus
skin. Through `rgpot.ase.RgpotCalculator`, a sheared 32-atom copper cell,
which the Morse cutoff tiles into 864 atoms, went from 203 to 39 ms for
the first evaluation and from 48 to 8.5 ms for each one after a small move.
