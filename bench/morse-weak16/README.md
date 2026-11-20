Copyright (c) anneal developers

# pair_rank stops on 55D

Negative record. Nothing here is proposed for develop.
The opt-in lattice ranking is 446cd701. recommended() is
unchanged. This commit adds no search code.

## What the m55 misses are

pair_rank on m55 rho=14, seeds 2000-2011, hits 2/12.
recommended hits 9/12. Seven of the ten misses end on
55D, -220.51853338. 55D is the Cs packing whose listed
range starts at rho 15.09. At rho=14 the global minimum
is 55C, -220.646208, and a quench of 55D sits 0.1277
higher.

The first 55D record, and the fraction of the 1e6 budget
still left:

- 2001 at call 544722, 0.46 left
- 2002 at call 712736, 0.29 left
- 2005 at call 445610, 0.55 left
- 2006 at call 762709, 0.24 left
- 2007 at call 555284, 0.44 left
- 2008 at call 707920, 0.29 left
- 2009 at call 282966, 0.72 left

recommended also misses 2002 and 2009, both on 55D.
Seeds 2004 and 2011 record 55D and still hit. The other
three misses never reach 55D: 2000 stays on
-220.49847976, 2003 ends at -220.49016163, and 2010
ends at -219.94332884.

After that first 55D record the walk does not improve,
except on the two hits. Seed 2009 has 717034 calls left
and the later records are the same energy.

## Why the same score wins m38

m38 rho=14's minimum is the close-packed truncated
octahedron, -144.321054. The short-range score is
looking for that packing, and pair_rank goes from 4/12
to 10/12. m55's minimum at rho=14 is still the
icosahedron. Both cells are the same pair, so a rule
that only changes the pair cannot keep one and drop the
other. A cutoff on the number of atoms is not a rule.

On the published 55D coordinates the Morse rho=14
second shell, contacts between 1.2 and 1.6 of the median
bond, is worth about -0.48. The same contacts under
Lennard-Jones are worth about -19. The best eight
unrelaxed rises there span 0.013 and are all downhill.
The rises that break a contact sit about 0.6 higher and
fall outside that pool. Lennard-Jones on the same
geometry has no downhill rise. Its pool of eight is the
draw that leaves a shelf.

## Two draws that were measured and not kept

Both leave recommended() on the Lennard-Jones lattice.
Opening the pool to one well for every pair, including
Lennard-Jones, makes the C2v exchange stay at
-580.42126628, so that window is not the measured pool.
The runs below are the short-ranged arm only, on the
seeds that decide the question. They are not a full
gate. Each draw already loses a recommended hit, so
neither was committed.

Pool. When the second neighbour is not bound by a
twentieth of the well, the draw is every unrelaxed rise
within one well of the best. The twentieth is the
existing range gate. rho=6 and Lennard-Jones still bind,
and they keep the pool of eight.

- m38 2001 hit, call 321914
- m38 2002 hit, call 292194
- m38 2005 hit, call 190847
- m38 2008 hit, call 3387
- m55 2000 hit, call 867263
- m55 2001 miss, best -219.67621063
- m55 2006 miss, best -220.51853338
- m55 2009 miss, best -220.51853338

The probed m38 hits stay, including the gains at 2002
and 2005. m55 2000, which pair_rank left on the shelf,
hits. m55 2001 and 2006 do not, and recommended hits
both.

Rank. The exchange is ranked at the longest Morse range
whose second neighbour is still bound, just under
rho 8.95. The pool stays eight. The quench stays at
rho=14.

- m38 2001 miss, best -142.71594037
- m38 2002 hit, call 330252
- m38 2005 miss, best -142.71594037
- m38 2008 hit, call 152319
- m55 2000 miss, best -216.55297417
- m55 2001 hit, call 949953
- m55 2006 miss, best -220.49016163
- m55 2009 miss, best -220.49847976

m38 2001 is a recommended hit, and this draw misses it.
m55 2000 and 2006 miss as well.

Neither draw keeps the m38 gain and loses nowhere.
pair_rank stays off the default.
