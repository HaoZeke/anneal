# CREST first hit on (H2O)8

Reference −40.681336013150 Eh. Tolerance 1e-6 Eh. Threshold −40.681335013150 Eh. Budget 151738 calls. A miss is stored as 151739.

A counted call is a finished MD step or one optimization cycle, the same accounting as the recorded full-run totals. A finished MD step is `MD time /ps` times 1000 divided by `dt /fs`, and only a block that later says finished is counted. An optimization cycle is one `* total energy` line. The cycle energy is the full-precision value from the optimization log. The geometry after the final `:: total energy` line is a later point and is not a counted call. Intact means every oxygen has two hydrogens within 1.2 Å. `first_hit_call` is the index of the first counted call whose intact geometry is at or below the threshold. The search stops at the budget.

The invocation is `crest start.xyz --gfn2 -T 1 --legacy` with the GFN2 evaluator behind `-xnam` and one thread. CREST is 3.0.2, revision `6914a25`. The evaluator is xtb 6.7.1, revision `edcfbbe`.

## Seed 17000

The recorded total for seed 17000 is 127141. This rerun counted 151538. The command, the binaries, the rigid-repack start, and the one-thread setting are the recorded ones. Metadynamics velocities are not pinned, so a call total is one sample and need not repeat 127141. This sample was still optimizing when the counter reached the budget. The first hit is call 131547, before that cap. On this seed the short timing trial was charged against the cap, so the counted total stops 200 calls short of 151738. Later seeds do not charge that trial.

## Seeds 17000–17011

| seed | first_hit_call | total_calls | hit | best / Eh |
| --- | --- | --- | --- | --- |
| 17000 | 131547 | 151538 | 1 | −40.681335841728 |
| 17001 | 113009 | 151738 | 1 | −40.681335622591 |
| 17002 | 66888 | 142740 | 1 | −40.681335976117 |
| 17003 | 76037 | 151738 | 1 | −40.681335914197 |
| 17004 | 81130 | 151738 | 1 | −40.681335975453 |
| 17005 | 151739 | 151738 | 0 | −40.681245390898 |
| 17006 | 91817 | 139177 | 1 | −40.681335988082 |
| 17007 | 151739 | 151738 | 0 | −40.681245466346 |
| 17008 | 151739 | 151738 | 0 | −40.681245423782 |
| 17009 | 151739 | 151738 | 0 | −40.603591290126 |
| 17010 | 151739 | 151738 | 0 | −40.681245330059 |
| 17011 | 151696 | 151738 | 1 | −40.681335265487 |

Seven of the twelve seeds hit. Seeds 17005, 17007, 17008, 17009, and 17010 ran to the budget without an intact geometry at the threshold. On seeds 17003–17006 the wall clock is longer than the cpu time; the call counts are the evaluations.
