# Why the population arm loses LJ98

Negative record. The adaptive population search is not proposed for
develop. The measurements below are the weak11 binary, serial,
`Config::recommended` against `Config::recommended_adaptive`. Budgets
and references are the weak11 cells: LJ98 at 1500000 calls,
reference -543.665361. A hit is `lowest <= reference + 0.001`.

The two arms record the same energies through call 524288
(`Config::STREAM_HOLD_OPENING`). The second member is present and
receives none of those calls. The fork then stops the incumbent and
spends the remaining 975712 calls on a search restarted from the
caller's coordinates, on a generator copied before the opening.
Those calls are split across the two members:

| seed | member calls | restarts | recorded best |
| --- | --- | --- | --- |
| 2001 | 476587, 498142 | 4 | -543.6429572274 |
| 2003 | 479128, 495874 | 7 | -543.6429572313 |
| 2004 | 483861, 490806 | 6 | -543.6429572220 |

The recorded best sticks at the second-best minimum, -543.64296.
The members do not finish in that basin. Seed 2001 ends at
-532.956 and -541.895, seed 2003 at -535.995 and -531.540, seed
2004 at a non-finite energy and -529.155. Seed 2003's fork never
beats the second-best the opening already held at call 10617.

Recommended finds the global minimum on the continuation of that
same chain, after the cut: seed 2001 at call 682516, seed 2004 at
call 755416, seed 2003 at call 1333413. Seed 2003 sits on the
second-best from call 10617 until that descent. The continuation
needs 809125 of the calls the fork gives away.

A side quench on the copied generator, fired once each time the
open quiet exceeds the longest gap the incumbent has closed, and
barred from moving the chain, still loses seed 2003. Its energy
sequence matches recommended through the second-best, and the
shifted call clock misses the later descent. Seeds 2001 and 2004
hit under that rule. It is not a fix, and it is not in this tree.

No rule that takes calls off that chain can keep seed 2003 inside
a budget of 1500000. The population, given the whole tail, does
not find the minimum the continuation finds. The arm stays off
develop.

Per-member rows are `member-calls.csv`.
