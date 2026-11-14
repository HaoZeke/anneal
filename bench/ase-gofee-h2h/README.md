# ASE basin hopping, ASE minima hopping, and AGOX GOFEE

Lennard-Jones 38, 55 and 75. Six seeds, 2000 through 2005, shared with the paper preset. A hit is a quenched energy within 0.001 of the Cambridge minimum.

The force-call budget is the paper budget and stays in the budget column: 400000 for LJ38, 1000000 for LJ55, and 4000000 for LJ75. ASE basin hopping, and every arm on LJ38 and LJ55, runs to that budget or to a hit.

ASE minima hopping and AGOX GOFEE on LJ75 also stop at a wall cap of 2 hours per seed. A seed stopped by the cap is `capped`. It is a miss: hit stays 0, and the row keeps the calls spent, the best energy, and the wall. Seed 2002 of minima hopping was stopped when the cap was introduced, at 800000 calls, best energy -393.153042, wall 34397 s.

Minima hopping compares each new minimum with every minimum already stored. The atoms are one element, so that comparison grows with the length of the list. On LJ75 seed 2002 the measured rate was about 400000 calls/hour at 180000 calls (the first half hour) and about 49000 calls/hour between 700000 and 800000 calls.
