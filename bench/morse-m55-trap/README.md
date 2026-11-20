Copyright (c) anneal developers

# Refusing a return onto the record loses hits

Negative record. Nothing here is proposed for develop.

## What the misses do

On the recommended search, m55 rho 14 seeds 2002 and 2009 finish
at -220.51853338, and seed 2005 finishes at -220.49847976. The
proposal scale stays 1. No restart runs. No trail runs. The stall
response that does run is a climb, and it does not beat the record.

Seed 2002 first records -220.49001260 at hop 2886, call 303506,
a dls-kernel quench. The exact 55D energy is hop 7532, call
580683, also dls-kernel, with 419317 calls left. The 7054 trials
after that record stay at or above 55D. Their draws are dls-kernel
2416, burst 1676, surface 1030, single 985, sym 759 and soap 188.
199 of those quenches land on the 55D energy (dls-kernel 197,
soap 1, surface 1). 82 climbs are adopted, 77 of them more than
1 above 55D. The PTM triple on the exact-energy accept is 6/5/0.

Seed 2009 records the same energy at hop 10886, call 826082,
dls-kernel, with 173918 calls left. Its PTM triple there is
6/0/0. The 3192 later trials stay at or above 55D: dls-kernel
906, burst 844, surface 526, single 431, sym 360 and soap 125.
34 climbs are adopted.

Seed 2005 never records 55D. Its best is -220.49847976 at hop
13163, call 952937, dls-kernel, with 47063 calls left. The PTM
triple is 10/0/0. 791 later trials and 9 climbs stay above 55D.

Hitting seeds that miss 55D find 55C from a chain that is not
sitting on it. Seed 2000 stops at call 103415 inside a dls-kernel
quench proposed at call 103240 from -219.67577780. Seed 2004's
last finished accept is a dls-kernel quench at -220.63363359,
and the hit at call 334221 is the next dls-kernel quench from
that structure. Seed 2008 hits by an accepted orbit-completion
at call 198984, energy -220.64620756, with no 55D record. Seed
2007 does record 55D, at hop 4998, call 409132, after
-220.51237892 at call 408211, both dls-kernel. 8110 finished
trials after that stay at or above 55D, and 101 climbs are
adopted. The hit at call 932023 is an in-flight dls-kernel
quench from -219.78938361.

## The refusal

A lattice relocation looks new to the partial return screen. The
full quench then falls into the recorded basin, and acceptance
walks back onto it. A quench at or above the record held before
that descent, whose structure the bias already places in that
basin, is refused. An improvement is kept. No threshold was
added. The comparison is the record energy, and the basin test
is the existing lookup.

## Gate

Serial, seeds 2000-2011, paper budgets. Morse rows are h2h_morse
on Config::recommended. LJ rows are h2h_seed on the same preset,
stopped at the reference plus 0.001. A hit is a quenched energy
at or below the reference plus 0.001. The references are 55C
-220.646208, Morse 38 at rho 6 -157.477108, 80G -340.811371,
LJ38 -173.928427 and LJ75 -397.492331. Budgets are 400000 at
38 points, 1000000 at 55, and 4000000 at 75 and 80.

The baseline revision is
61fa67cf380e46876b569da45e7cc091647a775b.
The return rows are
727998d9e0fc80b333eb9b052248d9566153e354.
A miss is charged the full budget. Fisher exact tests on hits
and Mann-Whitney tests on calls are two-sided, with the tie and
continuity corrections.

m55r14: hits 9/12 to 8/12, median calls 451561.5 to 365841.5,
Fisher p 1, call p 0.619307, outside-time median 1.037410.
It loses seed 2004 (hit at 334221, finishes at -220.51853338),
seed 2008 (hit at 198984, finishes at -220.51853338) and seed
2011 (hit at 55240, finishes at -220.51853338). It gains seed
2002 (hit at 394420) and seed 2009 (hit at 22954). Seed 2005
stays a miss at -220.49847976.

m38r6: hits 12/12 to 12/12, median calls 49438.5 to 49913.0,
Fisher p 1, call p 0.707454, outside-time median 0.929900.
No hit is lost.

m80r14: hits 6/12 to 7/12, median calls 3901429.0 to
3410986.5, Fisher p 1, call p 0.648867, outside-time median
1.078989. It loses seed 2000 (hit at 3711158, finishes at
-340.38733557), seed 2006 (hit at 3802858, finishes at
-340.38733557), seed 2007 (hit at 1292375, finishes at
-340.80256253) and seed 2010 (hit at 2973511, finishes at
-340.38708407). It gains seeds 2002, 2004, 2008, 2009 and
2011, at calls 3967568, 2178065, 2854405, 480568 and 1193694.
Seed 2005 stays a miss and moves from -340.38733557 to
-339.83798212.

LJ38: hits 12/12 to 12/12, median calls 6357.0 to 4733.5,
Fisher p 1, call p 0.750622, outside-time median 0.967168.
No hit is lost.

LJ75: hits 12/12 to 12/12, median calls 44176.0 to 17755.0,
Fisher p 1, call p 0.056693, outside-time median 0.985669.
No hit is lost.

A gained hit does not cancel a lost one. Over the 60 measured
rows, the median of outside time per call against the baseline
is 0.998681. The largest row is 1.356698. Every cell median is
within 1.1.

sc126-55 was not measured. librgpot.so is not on this machine,
so those 12 seeds have no rows.
