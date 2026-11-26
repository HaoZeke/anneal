# LJ98 walklen4 gate, shard A

Arm A revision `4feb0699d4e61c9531283e82196fcb60baf66932` tree `788d36860fb10d5863299acef0d0538618267d6d`.
Arm B revision `ddd53f2f60d53aebdf2e97f03d3ade07c2905738` built tree `2fb57f0db81b5c3d1ec78077881b2982b8b78a55`.
B is `git merge-tree --write-tree` of A and that tip.
The tip's parent is `f75836be832c0f953f99c26efb9cd5c4b87565b0`.
The built tree adds `paper/related-work.md` from A; the search sources are the tip.

Workload: LJ98, `h2h_seed lj 98 0 4000000 <seed> recommended -543.664361 <prefix>`.
Budget 4000000. Seeds 2048-2083 (n=36). One core, one process, ABBA order.
A hit is energy <= -543.665361 + 0.001. A miss scores 4000001 calls.
Outside-potential CPU time is process CPU (`wait4` user+system) minus harness `pot_s`.
Ratios use seeds where both arms hit. Bootstrap 10000 resamples, seed 20261123,
each ratio with its own generator. Mann-Whitney is two-sided, asymptotic, tie-corrected.
Fisher exact is two-sided. GMIN was not run.

Host id hash `3b1b725a97e4e3fc407fe86011128eed3413f48e895d03fa0514e1a70eb8a61a`. Core `0`.

## Verdict

**FIX**

| check | value | bar | call |
| --- | --- | --- | --- |
| hits A | 33 / 36 |  |  |
| hits B | 28 / 36 | B >= A | unclear |
| Fisher two-sided p | 0.188679 | p < 0.05 and B fewer is a clear loss | unclear |
| Fisher one-sided less p | 0.0943394 | reported |  |
| MWU two-sided p | 0.057294 | p < 0.05 and B slower is a clear loss | pass |
| MWU U (B) | 817 | null center 648.0 |  |
| calls geomean B/A | 1.81004 | both-hit only, reported |  |
| calls 95% CI | 0.872761 .. 3.89794 | reported |  |
| outside geomean B/A | 1.8003 | both-hit only | clear_fail |
| outside 95% CI | 0.884205 .. 3.77749 | upper <= 1.10 | clear_fail |
| both-hit pairs | 25 |  |  |
| valid | True | complete ABBA measurement |  |
| verdict | FIX | PASS, else FIX if a bar fails clearly, else UNRESOLVED | FIX |

Seeds whose hit flags differ: 2051 2053 2054 2055 2056 2060 2066 2067 2078 2079 2083.

## Descriptive

| arm | hits | median calls (miss = 4000001) | median outside CPU on both-hit s | median wall s |
| --- | --- | --- | --- | --- |
| A | 33 | 1285888.0 | 16.762960 | 37.153367 |
| B | 28 | 1547093.0 | 19.449895 | 44.498565 |

Both-hit seeds: 25.

## Rules

PASS needs no hit loss, no significant calls-to-hit loss, and an outside-time
upper confidence bound of at most 1.10. FIX if any of those fails clearly.
UNRESOLVED otherwise. The verdict is the table value and is not relabelled.

