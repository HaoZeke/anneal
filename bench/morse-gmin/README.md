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

Earlier numbers were removed because their revision is not reachable. They remain on `origin/fix/morse-gmin`.
