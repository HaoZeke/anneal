# Morse against GMIN

Cambridge Cluster Database, Doye and Wales, Morse table. A bold entry is the global minimum at that range. The potential is `PairKind::Morse`, `exp(rho(1-r))(exp(rho(1-r))-2)`. A hit is within 0.001 of the table. A miss is entered at the budget.

| system | structure | rho | energy | budget |
| --- | --- | --- | --- | --- |
| m30r6 | 30B | 6 | -118.432844 | 400000 |
| m30r14 | 30D' | 14 | -106.835790 | 400000 |
| m38r6 | 38E | 6 | -157.477108 | 400000 |
| m38r14 | 38D | 14 | -144.321054 | 400000 |
| m55r6 | 55B | 6 | -250.286609 | 1000000 |
| m55r14 | 55C | 14 | -220.646208 | 1000000 |
| m80r3 | 80A | 3 | -690.577890 | 4000000 |
| m80r6 | 80B | 6 | -378.333471 | 4000000 |
| m80r14 | 80G | 14 | -340.811371 | 4000000 |

Budgets follow the Lennard-Jones paper list: 4e5 up to 38 atoms, 1e6 below 75, 4e6 from 75. Twelve seeds, 2000-2011. Anneal is `Config::recommended`. Serial, one core, hostname `bench`. `pot_s` is inside the potential. `outside_s` is the rest. Per-seed rows are `seeds.csv`. Type-7 medians, the two-sided Fisher exact test on hits, and the two-sided tie- and continuity-corrected Mann-Whitney test on calls are `summary.csv`.

GMIN is the public Wales build, revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, with `scripts/morse_h2h/morse-timer.patch` on the Morse potential. The data file is `examples/h2h_gmin_morse.data` plus `MORSE <rho>` and `RANSEED`. The documented basin-hopping settings in that file are kept. Rows already stored at this revision, the same system, seed and budget were copied: M38 seeds 2000-2005 from `bench/morse-h2h`, and every M55 and M80 seed from `bench/morse-panel` and `bench/morse14-h2h`. M30 and the M38 seeds 2006-2011 were run in this session. `code_revision` is the anneal commit for the anneal rows and the GMIN revision for the GMIN rows. `host` is `bench`.

| system | anneal | gmin | Fisher p | MW calls | anneal calls | gmin calls | anneal wall | gmin wall | anneal inside | anneal outside |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m30r6 | 12/12 | 11/12 | 1 | 5.92e-4 | 5616.5 | 66113.5 | 0.040 | 0.418 | 0.022 | 0.018 |
| m30r14 | 10/12 | 1/12 | 6.44e-4 | 1.19e-3 | 242026 | 400000 | 1.359 | 2.488 | 0.928 | 0.431 |
| m38r6 | 12/12 | 11/12 | 1 | 3.66e-5 | 432.5 | 146866 | 0.003 | 1.332 | 0.003 | 0.001 |
| m38r14 | 12/12 | 0/12 | 7.40e-7 | 1.03e-5 | 17762.5 | 400000 | 0.158 | 3.591 | 0.110 | 0.053 |
| m55r6 | 12/12 | 12/12 | 1 | 3.66e-5 | 563 | 27461.5 | 0.008 | 0.569 | 0.007 | 0.001 |
| m55r14 | 7/12 | 0/12 | 4.58e-3 | 2.86e-3 | 730604 | 1000000 | 13.013 | 17.251 | 9.139 | 3.892 |
| m80r3 | 12/12 | 12/12 | 1 | 1.35e-3 | 761438.5 | 114617.5 | 29.483 | 4.047 | 19.839 | 9.644 |
| m80r6 | 2/12 | 12/12 | 6.73e-5 | 2.39e-5 | 4000000 | 346246 | 161.404 | 12.041 | 104.714 | 56.587 |
| m80r14 | 9/12 | 0/12 | 3.37e-4 | 3.83e-4 | 1338932.5 | 4000000 | 46.623 | 138.838 | 35.260 | 11.432 |

Calls, wall and the two anneal times are type-7 medians. Wall is in seconds.
