# Morse rho = 6, Mackay cap, seeds 2000-2005

Serial, one core, hostname `bench`, working directory `/work`. The binary is the search on `cdb82168` with the long-range Mackay cap. A hit is within 0.001 of the Cambridge table. Medians are Hyndman-Fan type 7. `pot_s` is inside the potential. `outside_s` is the rest. Per-seed rows are `seeds-rho6-cdb82168.csv`.

The flag is the second-neighbour energy of the pair, not the size. At rho = 6 it is set. At rho = 14 it is not: M55 seed 2003 still stops at 245577 evaluations, the unlabelled count.

| system | hits | evals median | wall median (s) | unlabelled median | GMIN median |
| --- | --- | --- | --- | --- | --- |
| m38r6 | 6/6 | 297.5 | 0.002458 | 97681 | 88057 |
| m55r6 | 6/6 | 232.5 | 0.003845 | 24344.5 | 23758.5 |

The unlabelled medians are the recommended search on this line, six seeds, same budgets. The older `seeds.csv` medians were 103678.5 and 129013.
