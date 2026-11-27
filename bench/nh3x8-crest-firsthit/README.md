# CREST first hit on (NH3)8

Reference −35.470769324714 Eh. Tolerance 1e-6 Eh. Threshold −35.470768324714 Eh. Budget 332675 calls. A miss is stored as 332676.

A counted call is a finished MD step or one optimization cycle, the same accounting as `bench/h2o8-crest-firsthit/run.py`. A finished MD step is `MD time /ps` times 1000 divided by `dt /fs`, and only a block that later says finished is counted. An optimization cycle is one `* total energy` line. The cycle energy is the full-precision value from the optimization log. The geometry after the final `:: total energy` line is a later point and is not a counted call. Intact means every nitrogen has three hydrogens within 1.2 Å. `first_hit_call` is the index of the first counted call whose intact geometry is at or below the threshold. The search stops at the budget.

The invocation is `crest start.xyz --gfn2 -T 1 --legacy` with the GFN2 evaluator behind `-xnam` and one thread. The start geometry is the rigid repack from `gfn2_mol_h2h nh3-8 4000 <seed>`, the same generator as `bench/nh3-octamer`. CREST is 3.0.2, revision `6914a25`. The evaluator is xtb 6.7.1, revision `edcfbbe`.

Seeds 17000–17011 ran strictly serial, one thread, no time cap. Six seeds hit. The median of the twelve `first_hit_call` values is 325303.5. Every best geometry is intact. The wall clock is longer than the cpu time; the call counts are the evaluations.

| seed | first_hit_call | total_calls | hit | best / Eh | wall / s |
| --- | --- | --- | --- | --- | --- |
| 17000 | 317931 | 318415 | 1 | −35.470769256519 | 3963.000000 |
| 17001 | 298654 | 318346 | 1 | −35.470769298282 | 3807.000000 |
| 17002 | 332676 | 332675 | 0 | −35.470768121187 | 7321.000000 |
| 17003 | 332676 | 332675 | 0 | −35.470767594682 | 11477.000000 |
| 17004 | 134643 | 274566 | 1 | −35.470769327640 | 6759.000000 |
| 17005 | 297116 | 310890 | 1 | −35.470769232779 | 9489.000000 |
| 17006 | 332676 | 332675 | 0 | −35.470768183710 | 5764.000000 |
| 17007 | 332676 | 332675 | 0 | −35.470733013142 | 5965.000000 |
| 17008 | 258650 | 275442 | 1 | −35.470769278388 | 3908.000000 |
| 17009 | 332676 | 332675 | 0 | −35.450669475781 | 9218.000000 |
| 17010 | 332676 | 332675 | 0 | −35.470765670079 | 6282.000000 |
| 17011 | 187294 | 270752 | 1 | −35.470769327605 | 3847.000000 |

## Recorded full-run totals

`bench/nh3-octamer/results.csv` records the crest-arm call totals for seeds 17000–17005. Metadynamics velocities are not pinned, so a rerun need not repeat them.

| seed | this total_calls | recorded calls | match |
| --- | --- | --- | --- |
| 17000 | 318415 | 243868 | 0 |
| 17001 | 318346 | 221579 | 0 |
| 17002 | 332675 | 332675 | 1 |
| 17003 | 332675 | 241500 | 0 |
| 17004 | 274566 | 134305 | 0 |
| 17005 | 310890 | 196621 | 0 |

Seed 17002 is the one match. Its search was still running when the counter reached the budget, and the recorded full run has that same call total.
