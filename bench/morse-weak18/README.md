Copyright (c) anneal developers

# The silence clock still does not leave 55D

Negative record. Nothing here is proposed for develop.

A plateau leave zeroed the hop silence and the call gap. Quiet
never outlasted the longest stretch already answered, and the
stall allocator never ran. A leave that misses the existing
improvement threshold now leaves both clocks on the last record.
The stall response records the stretch it answered, so the same
twice-longest test waits instead of firing on every later hop.
A new record is what ends the silence. No threshold was added.

## Gate

Serial, seeds 2000-2011. Budgets are the Lennard-Jones paper
list: 400000 at 38 points, 1000000 at 55, and 4000000 at 80.
Morse rows are h2h_morse on Config::recommended. LJ38 rows are
h2h_seed on the same preset, stopped at the reference plus
0.001. A hit is a quenched energy at or below the reference
plus 0.001. The references are 55C -220.646208, Morse 38
-144.321054, 80G -340.811371, and LJ38 -173.928427.

The baseline revision is
61fa67cf380e46876b569da45e7cc091647a775b.
The clock rows are
22e69120797c5aefb553f6df4996b58af17cd2d0.
A miss is charged the full budget. Fisher exact tests on hits
and Mann-Whitney tests on calls are two-sided, with the tie
and continuity corrections.

m55r14: hits 9/12 to 8/12, median calls 451561.5 to
451561.5, Fisher p 1, call p 0.930009.
m38r14: hits 4/12 to 4/12, median calls 400000 to
400000, Fisher p 1, call p 1.
LJ38: hits 12/12 to 12/12, median calls 6357 to 6357,
Fisher p 1, call p 1.
m80r14: hits 6/12 to 8/12, median calls 3901429 to
2069761, Fisher p 0.680172, call p 0.308224.

It loses hits. m55 seed 2007 was a hit at call 932023 and
finishes on 55D, -220.51853338. m80 seed 2000 was a hit at
call 3711158 and finishes at -340.80256253. m80 gains seeds
2002, 2008 and 2011. A gained hit does not cancel a lost one.

## 55D

Seed 2002 still finishes at -220.51853338. Its move draws are
not the baseline draws, so a silence was answered, and that
answer did not beat 55D. Seed 2009 leaves 55D for
-220.49847976, which is worse. Seed 2005 moves from
-220.49847976 onto 55D and is still a miss.

## What stayed put

m38r14 matches the baseline ledger on all 12 seeds. Seed 2001
still hits at call 219638. The one m38 run past 5000 hops,
seed 2004, has the same arm draws as the baseline binary.
LJ38 matches on all 12 seeds. The longest LJ38 run is 469
hops, under the stall patience, so no silence was recorded.
Those runs are bit-identical.
