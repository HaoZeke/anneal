# LJ75 reseed projection

Same session, hostname `bench`, working directory `/work`, `taskset -c 0`. The incumbent tip is `b76373c4`. Sizes run LJ38, LJ75, LJ98, LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is within 0.001 of the Cambridge minimum.

`incumbent` projects the live chain only after a new lowest energy and leaves the icosahedral reseed unprojected. `fix` keeps that chain rule and projects a reseed only after the best icosahedral state has stood for eight reseed intervals, and only when the plain quench did not beat it.

| Size | Incumbent | Fix |
| --- | --- | --- |
| LJ38 | 6/6 | 6/6 |
| LJ75 | 4/6 | 6/6 |
| LJ98 | 5/6 | 5/6 |
| LJ104 | 6/6 | 6/6 |

LJ75 seeds 2000 and 2001 are the two the incumbent misses. Both hit with the stall-gated reseed projection.
