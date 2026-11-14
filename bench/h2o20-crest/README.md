Copyright (c) anneal developers

(H2O)20, seeds 2000-2005.
Both arms search GFN2 plus the one-sided O-H wall, onset 1.15 angstrom,
200 eV per square angstrom.
`PAPER_BUDGETS` for this system is 28000 charged calls, the cap already
used for GFN2 (H2O)20.
Each seed starts from one random placement of rigid waters: centres drawn
uniformly in a ball of radius 9 angstrom, orientations from random
quaternions, and an intermolecular contact inside 2 angstrom refused.
Both arms receive that geometry. The start is not a motif, a lattice, or
a shell indexed by group.
The host is bench, on core 0, in one UTS namespace.
The shared rescore is
`xtb in.xyz --gfn 2 --opt vtight --acc 0.01 --iterations 500`.

The Cambridge / Wales--Hodges (H2O)20 minimum in this repository is the
TIP4P energy -872.9887544952 kJ/mol.
This panel is GFN2-xTB, so that energy is a different potential.
The reference is the lowest rescored energy of the twelve runs:
anneal seed 2003, -101.748066515747 Eh.
A hit is at or below that energy by 1e-6 Eh.
Anneal hits 1 of 6 seeds. CREST hits 0 of 6.
Fisher's exact two-sided test on those hit counts has p 1.

The two-sided Mann-Whitney test on the six rescored energies, the normal
approximation with the tie correction and the continuity correction, has
p 0.005074868098.
The paired gap, anneal minus CREST, has W+ 0, W- 21, exact two-sided p
0.03125, rank-biserial -1, median -0.007216007464 Eh.
Anneal is lower on every seed.
Anneal's lowest is the reference. CREST's lowest is seed 2002,
-101.739503833669 Eh, 0.008562682078 Eh above the reference.

Anneal's time inside the potential has median 1236.023867 s.
Time outside the potential has median 474.677276 s.
CREST's time inside the driver handle has median 1256.795367 s, and the
rest of its wall has median 78.7654125 s.

The searches ran at code revision
c2de652dca0bc75e2b0cb4afffac91fd656404e4, host bench.
Rows are appended. A pair already recorded at this budget is left as written.
