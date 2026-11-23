<!-- Copyright 2023--present rgpot developers -->

A periodic `Cluster4` neighbour list now holds the same rows whichever output fields it stores. Rows of one cluster and partner were merged by their stored shift, so a query without shifts merged every periodic image of a partner into one row and lost the rest. In a 3 Å cubic cell with four atoms and a 4 Å cutoff, `build`, `Walk` and `Builder` now give 74 rows for every choice of shifts, vectors and distances, where a query without shifts gave 4. Pair lists were not affected.
