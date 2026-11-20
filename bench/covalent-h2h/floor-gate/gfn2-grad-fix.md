# GFN2 floor-gate gradient-sign replacement

The `floor-rerun`, `series` and `grad` rows in `gfn2.csv` used the
reversed xTB gradient and are superseded as evidence by
`floor-rerun-grad-fix`, `series-grad-fix` and `grad-grad-fix`,
respectively. The original rows remain in the append-only file.

The replacement rows use revision
`a37896fdf9d098a879db18f6a5d5e30b95b65da5` and hostname `bench`.
They repeat seeds 2000--2011 for Si13 and C20 at 100,000 calls, serially
on core 0 with one thread, xTB 6.7.1, electronic temperature 3000 K,
rgpot revision `8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, and the
Cooperlake OpenBLAS kernel. The example, engine and xTB library SHA-256
digests are, respectively,
`2e67177d880ee8663ea9f778d8614a32252e6d07a6ef81ac0a4e12344af3148e`,
`4d9dd694c1f861285d867eb019c95505364ae12dd251250de101922a1a4ed451`
and
`39232b70c4669a284e5136fc649520ab7239eb859df15c8d7af98ad5e3bfa7c9`.

A central finite difference on three geometries gave a maximum absolute
gradient error of `9.392943312037e-9 eV/angstrom`.

The table uses a two-sided Fisher exact test on hits and a two-sided
asymptotic Mann--Whitney test with tie correction and continuity
correction on calls to the hit. A miss counts as its 100,000-call
budget. The all-tied C20 call samples have p = 1. Inside and outside are
the medians of the per-run timing fields.

| system | old to replacement | hits old/new | median calls old/new | Fisher p | tie-corrected MW p | SCC failures old/new | median best old/new (eV) | median inside old/new (s) | median outside old/new (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Si13 | `floor-rerun` to `floor-rerun-grad-fix` | 12/12 to 12/12 | 1972.5 to 1837.5 | 1.0000 | 0.8852 | 1424 to 554 | -600.610853 to -599.345467 | 460.586 to 323.702 | 0.025 to 0.032 |
| Si13 | `series` to `series-grad-fix` | 12/12 to 12/12 | 1972.5 to 1837.5 | 1.0000 | 0.8852 | 1424 to 554 | -600.610853 to -599.345467 | 461.651 to 325.604 | 0.026 to 0.051 |
| Si13 | `grad` to `grad-grad-fix` | 12/12 to 12/12 | 1972.5 to 1837.5 | 1.0000 | 0.8852 | 1424 to 554 | -600.610853 to -599.345467 | 460.621 to 317.618 | 0.030 to 0.025 |
| C20 | `floor-rerun` to `floor-rerun-grad-fix` | 0/12 to 0/12 | 100000 to 100000 | 1.0000 | 1.0000 | 0 to 0 | -1146.550244 to -1146.550244 | 2902.738 to 2641.446 | 0.846 to 0.513 |
| C20 | `series` to `series-grad-fix` | 0/12 to 0/12 | 100000 to 100000 | 1.0000 | 1.0000 | 0 to 0 | -1146.550244 to -1146.550244 | 2930.950 to 2634.856 | 0.882 to 0.479 |
| C20 | `grad` to `grad-grad-fix` | 0/12 to 0/12 | 100000 to 100000 | 1.0000 | 1.0000 | 0 to 0 | -1146.550244 to -1146.550244 | 2640.078 to 2637.053 | 0.846 to 0.494 |
