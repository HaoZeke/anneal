# Morse rho = 14, unlabelled search at cdb82168

Same harness as `seeds.csv`: six seeds 2000-2005, serial, one core, hostname `bench`, working directory `/work`. A hit is within 0.001 of the Cambridge table. Medians are Hyndman-Fan type 7. `pot_s` is inside the potential. `outside_s` is the rest.

These rows are the recommended unlabelled search. The Mackay cut, the spine cut, and the sheet slip on this commit are labelled-only, so they do not enter. The earlier `seeds.csv` was measured at `fe9671c8` and is left as it is. Per-seed rows are `seeds-cdb82168.csv`.

| system | hits | evals median | wall median (s) |
| --- | --- | --- | --- |
| m55r14 | 5/6 | 446461 | 10.161 |
| m80r14 | 4/6 | 2387370.5 | 110.462 |

M55 seed 2002 finishes at -220.518533, the Cambridge 55D energy. M80 seeds 2004 and 2005 finish at -340.802563, 0.0088 above 80G. Seeds 2000 and 2005 at M55 stop inside the budget; the charged count is the evaluations to the hit.
