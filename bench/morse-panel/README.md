# Morse panel

Cambridge Cluster Database Morse table, epsilon = 1. A bold entry is the global minimum at that range. The potential is `PairKind::Morse`, `exp(rho(1-r))(exp(rho(1-r))-2)`. A hit is within 0.001 of the table.

| system | structure | rho | energy | budget |
| --- | --- | --- | --- | --- |
| m55r6 | 55B | 6 | -250.286609 | 1000000 |
| m55r14 | 55C | 14 | -220.646208 | 1000000 |
| m70r3 | 70A | 3 | -577.739782 | 1000000 |
| m70r6 | 70D | 6 | -325.887749 | 1000000 |
| m70r10 | 70E | 10 | -296.412149 | 1000000 |
| m70r14 | 70F | 14 | -292.462856 | 1000000 |
| m80r3 | 80A | 3 | -690.577890 | 4000000 |
| m80r6 | 80B | 6 | -378.333471 | 4000000 |
| m80r10 | 80D | 10 | -344.911381 | 4000000 |
| m80r14 | 80G | 14 | -340.811371 | 4000000 |

Budgets follow the Lennard-Jones paper list: 1e6 below 75 atoms, 4e6 at 80. Twelve seeds, 2000-2011. `Config::recommended`. Serial, one core, hostname `bench`, working directory `/work`. `pot_s` is inside the potential. `outside_s` is the rest. Per-seed rows are `seeds.csv`. Type-7 medians are below. Anneal counts one energy-and-gradient pass and GMIN counts a potential entry, so this note prints no Fisher or Mann-Whitney value. `summary.csv` keeps the stored columns.

GMIN is the public Wales build, revision `aac1548a8aec13b53ad12a8d6904625081685d2e`. The data file is `examples/h2h_gmin_morse.data` plus `MORSE <rho>` and `RANSEED`. GMIN rows already stored at the same system, seed and budget are copied and not run again: m55r6 seeds 2000-2005 from `bench/morse-h2h`, and every m55r14 and m80r14 seed from `bench/morse14-h2h`.

The anneal walk is the recommended preset after the portable math and fused pair gradient. The acceptance dial for a steep well was checked on this preset and did not move a row: m55r6 seeds 2000 and 2006, m55r14 seeds 2000 and 2006, m70r3 seed 2000 and m80r3 seed 2000 kept the same hit call and the same best energy. A later rigid-replica start is not this search. `run_with_gradient` is one chain, and the preset is not a rigid body.

| system | anneal hits | gmin hits |
| --- | --- | --- |
| m55r6 | 12/12 | 12/12 |
| m55r14 | 6/12 | 0/12 |
| m70r3 | 11/12 | 12/12 |
| m70r6 | 9/12 | 11/12 |
| m70r10 | 2/12 | 0/12 |
| m70r14 | 3/12 | 0/12 |
| m80r3 | 12/12 | 12/12 |
| m80r6 | 12/12 | 12/12 |
| m80r10 | 2/12 | 0/12 |
| m80r14 | 10/12 | 0/12 |

Type-7 medians. Calls are `evals_to_hit`. A miss is entered at the budget. Each column is that arm's own median.

| system | anneal calls | gmin calls | anneal wall | gmin wall | anneal pot | gmin pot | anneal outside | gmin outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m55r6 | 233.5 | 27462 | 0.00717 | 0.569 | 0.00667 | 0.504 | 0.000489 | 0.0654 |
| m55r14 | 925177 | 1000000 | 33.84 | 17.25 | 26.08 | 15.22 | 7.76 | 2.02 |
| m70r3 | 301864 | 205473 | 20.00 | 6.25 | 14.02 | 5.74 | 5.98 | 0.518 |
| m70r6 | 529122 | 214479 | 34.47 | 5.96 | 25.47 | 5.48 | 8.82 | 0.482 |
| m70r10 | 1000000 | 1000000 | 62.31 | 27.23 | 47.14 | 25.07 | 15.14 | 2.16 |
| m70r14 | 1000000 | 1000000 | 59.33 | 26.76 | 46.58 | 24.82 | 12.63 | 1.96 |
| m80r3 | 542368 | 114618 | 46.77 | 4.05 | 32.60 | 3.78 | 14.17 | 0.268 |
| m80r6 | 1066991 | 346246 | 86.97 | 12.04 | 63.53 | 11.24 | 23.49 | 0.797 |
| m80r10 | 4000000 | 4000000 | 307.70 | 138.69 | 236.09 | 129.53 | 71.46 | 9.15 |
| m80r14 | 2479560 | 4000000 | 184.71 | 138.84 | 146.67 | 128.22 | 38.04 | 10.67 |

At rho = 6 and at M80 rho = 3 both arms hit every seed. At M55 rho = 6 the call medians are 233 for anneal and 27462 for GMIN. At M80 rho = 6 the call medians are 1.07e6 for anneal and 3.46e5 for GMIN. Each wall in the table is that arm's own median.

Four of the six M55 rho = 14 misses finish at -220.518533, the Cambridge 55D energy. The two M80 rho = 14 misses are seed 2009 at -340.387336 and seed 2010 at -340.802563, 0.0088 above 80G. M80 rho = 6 seed 2010 reaches -378.333465 at call 757199.
