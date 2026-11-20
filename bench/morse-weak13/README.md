Copyright (c) anneal developers

# Why m55 rho 14 and m80 rho 14 still miss

Negative record. Nothing here is proposed for develop.

The opt-in one-shot return-Twin search has no open lead. Its gate,
recorded in bench/morse-weak3, rejected it: m55r14 fell from 8/12
to 7/12, LJ38 from 12/12 to 8/12, and m38r10 from 10/12 to 8/12.
The hold-then-fork that passed the following gate is the population
arm closed in weak12.

## What develop actually returns

Revision 61fa67cf, serial `h2h_morse`, `Config::recommended`, seeds
2000-2011. Budgets are the Lennard-Jones paper list: 400000 at 38
points, 1000000 at 55, 4000000 at 80. A hit is a quenched energy at
or below the Cambridge reference plus 0.001. The references are 55C
-220.646208 and 80G -340.811371.

The three m55 energies -220.069, -219.739 and -216.058, and the m80
energy -339.838, are the weak11 population-record recommended arm.
That arm ran seeds 2000-2002 at 1000000 calls and seeds 2000-2001
at 2000000 calls. This binary does not finish there. m55 seeds 2000
and 2001 hit 55C, and seed 2002 finishes on 55D. m80 seed 2000 hits
80G at call 3711158, and seed 2001 hits at call 529680.

Measured on this binary: m55r14 9/12, m80r14 6/12.

## Where the misses stop

m55 misses:

| seed | best | funnel | best call | calls after | records | basins |
| --- | --- | --- | --- | --- | --- | --- |
| 2002 | -220.51853338 | 55D | 580683 | 419317 | 36 | 1084 |
| 2005 | -220.49847976 | 0.020 above 55D | 952937 | 47063 | 37 | 1110 |
| 2009 | -220.51853338 | 55D | 826082 | 173918 | 46 | 901 |

55D is tests/fixtures/morse-55D.txt. Relaxed at rho 14 it is
-220.518533, 0.1277 above 55C. Seeds 2002 and 2009 land on that
energy. Seed 2005 lands 0.020 above it and 0.1477 above 55C. The
in-tree Cambridge table has no label for that minimum. Hops are
14739, 13970 and 14134.

The incumbent stalls, and the walk does not. After the best call,
419317, 47063 and 173918 charged calls produce no lower record.
The run still ends in 900 to 1100 basins, against 36 to 46 records,
so almost every basin is not a new incumbent. Seed 2005's longest
gap between records is 591676 calls, and that gap is before the
best: the tail after call 952937 is 5% of the budget. Seeds 2002
and 2009 sit on 55D for the last 42% and 17%.

No arm holds a better minimum than the one the ledger records.
On all three seeds the six moves, single, surface, sym, burst,
soap and dls-kernel, have a best at that energy or worse.
dls-kernel takes 28% to 33% of the draws, burst 23% to 28%, and
soap 4% to 6%. The budget is not parked on one move. Draws and
per-arm energies are arms.csv.

m80 misses:

| seed | best | above 80G | best call | calls after | records | basins |
| --- | --- | --- | --- | --- | --- | --- |
| 2002 | -340.80256253 | 0.008808 | 633029 | 3366971 | 54 | 4443 |
| 2009 | -340.80256253 | 0.008808 | 3695683 | 304317 | 107 | 4268 |
| 2004 | -340.38733557 | 0.424035 | 511157 | 3488843 | 64 | 4339 |
| 2005 | -340.38733557 | 0.424035 | 161915 | 3838085 | 52 | 3174 |
| 2011 | -340.38733557 | 0.424035 | 510410 | 3489590 | 71 | 3642 |
| 2008 | -340.35713133 | 0.454240 | 2185165 | 1814835 | 54 | 3897 |

-340.802563 is the repeated shelf already noted for this cell,
0.008808 above 80G and outside the 0.001 window. The same six
moves top out at the incumbent or worse. Hops are 53000 to 63000
and basins 3200 to 4400, against 52 to 107 records. Seeds 2002,
2004, 2005, 2008 and 2011 have already stopped improving for 45%
to 96% of the budget. Seed 2009 reaches the shelf only at call
3695683, so its stall is the last 8%. dls-kernel takes 27% to
32% of the draws, burst 20% to 30%, and soap 2% to 4%.

## Switches that do not fix both cells

Each switch is already in the tree. None adds a size, a motif,
or a new constant. Seeds are 2000-2011 except the branch, which
was run only on the nine misses above. A miss that stays a miss
cannot raise the hit count.

Pair scale sets `H2H_PAIR_SCALE=1`, so Morse uses
`Config::recommended_for_pair`. The length is the pair minimum
divided by 2^(1/6), 0.8909, in place of the Lennard-Jones
diameter. m55r14 goes from 9/12 to 8/12. It gains seeds 2005 and
2009 and loses 2003, 2006 and 2008. Seed 2008 falls to
-219.94332884. m80r14 goes from 6/12 to 8/12. It gains 2002,
2004, 2008 and 2011 and loses 2007 and 2010. Seeds 2005 and 2009
stay on -340.80256253. m38r6 stays 12/12. m38r14 goes from 4/12
to 9/12 and loses seeds 2001 and 2008, which recommended hits.
m80r6 stays 12/12. The median of outside_s/charged is 0.67 to
0.81 of recommended. The m55 count falls, so the harness keeps
the Lennard-Jones diameter.

`branch_on_stall` on the nine misses converts none. The recorded
energies are the same minima. On the seeds where a branch charged
calls, the best-call clock moves and the minimum does not. m55
seed 2009's record trace is unchanged.

`plateau_heat` doubles the acceptance temperature on a stall
leave, up to the existing cap of 4, and resets it when the
incumbent improves. m55r14 goes from 9/12 to 7/12. It gains seed
2009 and loses 2004, 2008 and 2011, all three to -220.49847976.
Seed 2005 climbs from that minimum onto 55D and still misses.
Seed 2002 stays on 55D. m80r14 goes from 6/12 to 8/12. It gains
2002, 2005, 2009 and 2011 and loses 2000 and 2006. Seed 2000
ends at -339.83798212, the weak11 minimum, 0.973 above 80G.
Seeds 2004 and 2008 climb onto -340.80256253 and still miss.
m38r6 stays 12/12. m38r14 goes from 4/12 to 5/12 and loses seeds
2001 and 2008. m80r6 stays 12/12. The median of outside_s/charged
is 0.84 to 1.10 of recommended. m80r14 is 1.097.

LJ38, LJ75 and LJ98 were not run. Pair scale does not change
their configuration: `recommended_for_pair` for Lennard-Jones and
`recommended` both pass length scale 1 and energy scale 1. Branch
and plateau heat would change those runs, and both had already
lost the m55r14 hit count, so neither was a candidate.

Per-seed rows are rows.csv.
