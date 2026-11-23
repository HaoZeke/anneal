<!-- Copyright 2023--present rgpot developers -->

A variable-cell neighbour update now measures image shifts of both signs before it reuses a stored list. A lattice change whose signs differ no longer drops a pair that has moved into the cutoff. A negative, infinite or NaN skin does not count as a reuse in either `withinSkin` overload. Periodic positions are folded before the skin comparison, so adding whole lattice vectors does not move an atom.
