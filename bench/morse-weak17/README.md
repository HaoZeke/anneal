Copyright (c) anneal developers

# recommended stays on 55D

Negative record. Nothing here is proposed for develop.
recommended() on m55 rho=14 misses seeds 2002 and 2009,
both at 55D, -220.51853338. The global minimum is 55C,
-220.646208. This commit adds no search code.

## Where the budget goes

Seed 2002 first records 55D at call 580683, with 419317
calls left. Seed 2009 first records it at call 826082,
with 173918 calls left. The 717034-call remainder is
the pair-rank run of seed 2009, which records 55D at
call 282966. These two are recommended().

Counts are from a trace of that code. `branch` is
`branch_on_stall`. `patience` is `restart_patience`.
A leave is the call plateau. A stall escape on this
preset is that leave's climb: the trail count stays 0.

Seed 2002, whole run and at the 55D record:

- calls 1000000, hops 14739, returns 3158, leaves 186
- at the record: call 580683, hop 7532, returns 1771,
  leaves 104, lattice draws 2355, of which 29 quenched
  at least 1e-3 below the best so far
- pool empty 0, branch false, patience 5000
- climbs that landed within 1e-4 of the best: 0
- climbs that beat the best by 1e-3: 0

After that record: 82 leaves, 1387 return-screen
hits, 2415 lattice draws, 661 of those screened as
returns, and 0 lattice quenches 1e-3 below 55D.

Seed 2009, whole run and at the 55D record:

- calls 1000000, hops 14134, returns 3551, leaves 182
- at the record: call 826082, hop 10886, returns 3036,
  leaves 148, lattice draws 3179, of which 42 quenched
  at least 1e-3 below the best so far
- pool empty 0, branch false, patience 5000
- climbs within 1e-4 of the best: 3
- climbs that beat the best by 1e-3: 0

After that record: 34 leaves, 515 return-screen hits,
905 lattice draws, 164 of those screened as returns,
and 0 lattice quenches 1e-3 below 55D.

## Which mechanism holds the record

The stall clock fires. The plateau is one leave every
`restart_patience` calls, 5000. Each leave pins the
climb, then clears the hop clock and the call clock.
The hop test, quiet longer than twice the longest
quiet stretch, therefore ends both runs at longest 0.
The trail is the other stall exit and it never runs,
because the plateau selects the climb.

The basin-repeat test in the fresh-start branch does
not run: `branch_on_stall` is false. The return screen
does run, and it skips the full quench when a short
relaxation falls back within the merge radius. On seed
2002 that is 661 of the 2415 lattice draws after the
record. The other 1754 are full quenches, and none of
them beat 55D. The screen is not what holds the record.

The exchange pool is not exhausted. No lattice draw
returned empty. After the record the draws continue,
and none of them quench below 55D.

The climb fires and does not escape. No climb on seed
2002 beat the best, and the three climbs on seed 2009
that landed on the best energy did not leave it. A
climb is adopted whatever its energy, so the chain
moves and then comes back to the same record.

The same climb is part of a hitting trajectory.
m38 rho=14 seed 2001 hits at call 219638 after 37
climbs, and none of those 37 beat the best either.
The improvements on that seed come from the other
arms. Replacing the climb changes that path.

## Two leaves that were measured and not kept

Both leave recommended() unchanged when they are off.
Each one already loses a recommended hit.

Allocator. The plateau draws from the stall allocator,
so the trail can run. The trail is the recorded basin
entry, continued outward. Seed 2002 then hits at call
485441. Seed 2009 misses at -220.49847976. m38 rho=14
seed 2001 misses at -142.69779698. recommended hits
that seed.

Late allocator. The same draw starts only once the
current silence is longer than every completed gap
between records. That becomes true inside an ordinary
gap, because the longest completed gap is still short
early in the run. m38 rho=14 seed 2001 misses at
-142.71594037. Seed 2002 misses at -219.94332884.
Seed 2009 misses at -216.05802023.

Neither leave keeps every recommended hit. The climb
stays the plateau.
