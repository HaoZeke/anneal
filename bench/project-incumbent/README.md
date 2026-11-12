# Projection frequency

Same session, hostname `bench`, working directory `/work`, `taskset -c 0`. Develop is `e6a13a47`. LJ98 runs before LJ104. Inside a size the six seeds and three arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ98 −543.665361, LJ104 −582.086642). Both arms start from the same coordinates.

`develop` projects on the interval of ten quenches, including the icosahedral reseed. `accepted` keeps that interval on the live chain and does not project the reseed. `incumbent` projects only after a new lowest energy, and does not project the reseed.

| Size | Arm | Hits |
| --- | --- | --- |
| LJ98 | develop | 4/6 |
| LJ98 | accepted | 4/6 |
| LJ98 | incumbent | 5/6 |
| LJ104 | develop | 3/6 |
| LJ104 | accepted | 2/6 |
| LJ104 | incumbent | 6/6 |

`accepted` drops a hit at LJ104. `incumbent` does not, at either size. That is the policy in the search.

A cache of the projection by descriptor, reused within the merge radius, does not keep the trajectory. On 40_000 evaluations at LJ104 seed 2000, 7 of 32 projections were within 0.7 of an earlier descriptor and none had the same coordinates. Reusing the cached geometry moved the 150_000-evaluation best from −578.60157826 to −577.74065405. The 5_000-step LJ98 hash still matched, because those repeats are outside that prefix.
