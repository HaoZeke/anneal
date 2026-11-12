# Projection frequency

Same session, hostname `bench`, working directory `/work`, `taskset -c 0`. Develop is `eeed4308`. Sizes run LJ38, LJ75, LJ98, LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is a quenched energy within 0.001 of the Cambridge minimum (LJ38 −173.928427, LJ75 −397.492331, LJ98 −543.665361, LJ104 −582.086642).

`develop` projects on the interval of ten quenches, including the icosahedral reseed. `incumbent` projects only after a new lowest energy, and does not project the reseed.

| Size | Arm | Hits |
| --- | --- | --- |
| LJ38 | develop | 6/6 |
| LJ38 | incumbent | 6/6 |
| LJ75 | develop | 5/6 |
| LJ75 | incumbent | 4/6 |
| LJ98 | develop | 4/6 |
| LJ98 | incumbent | 5/6 |
| LJ104 | develop | 3/6 |
| LJ104 | incumbent | 6/6 |

LJ38 is unchanged. LJ75 drops one hit: develop misses 2003, incumbent misses 2000 and 2001. LJ98 gains 2003. LJ104 hits every seed.
