# Population search, weak11

Push-only record. These cells are the adaptive population arm against
`Config::recommended` on one binary, serial, the same seeds. The
population search is a negative result and is not proposed for develop.

A hit is `lowest <= reference + 0.001`. The `h2h_seed` hit flag uses
tolerance 0, so a quench about 1e-9 above a printed reference is
`hit=0` there. `lowest` is the search best. `best_call` is the trace
row written when the energy last dropped by at least 1e-7. The final
lowest sits at most 2.5e-7 below that traced energy.

| system | budget | reference |
| --- | --- | --- |
| LJ98 | 1500000 | -543.665361 |
| Morse38, rho 14 | 800000 | -144.321054 |
| Morse55, rho 14 | 1000000 | -220.646208 |
| Morse80, rho 14 | 2000000 | -340.811371 |

LJ98, seeds 2000-2004: adaptive hits 1/5, recommended hits 3/5.
Adaptive misses seeds 2001, 2003 and 2004, which recommended hits,
and wins only seed 2002. Every Morse cell is identical on the two
arms. Per-seed rows are `weak11-population-rows.csv`.
