# Morse rho = 14, unlabeled search at cdb82168

Same harness as `seeds.csv`: six seeds 2000-2005, serial, one core, hostname `bench`, working directory `/work`. A hit is within 0.001 of the Cambridge table. Medians are Hyndman-Fan type 7. `pot_s` is inside the potential. `outside_s` is the rest.

These rows are the recommended unlabeled search. The Mackay cut, the spine cut, and the sheet slip on this commit are labeled-only, so they do not enter. `seeds.csv` is the `fe9671c8` table. Per-seed rows of this commit are `seeds-cdb82168.csv`.

| system | hits | evals median | wall median (s) |
| --- | --- | --- | --- |
| m55r14 | 5/6 | 446461 | 10.161 |
| m80r14 | 4/6 | 2387370.5 | 110.462 |

M55 seed 2002 finishes at -220.518533, the Cambridge 55D energy. M80 seeds 2004 and 2005 finish at -340.802563, 0.0088 above 80G. Every hit stops inside the budget, M55 seeds 2000, 2001, 2003, 2004 and 2005 and M80 seeds 2000 to 2003; the charged count is the evaluations to the hit.
