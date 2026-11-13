# Morse generality panel

M30, M45, M60, M70, and M80 at rho = 3, 6, 10, and 14. Six seeds, 2000-2005, both arms, serial, one core, hostname `bench`, working directory `/work`. A hit is within 0.001 of the Cambridge table. Medians are Hyndman-Fan type 7. `pot_s` is inside the potential. `outside_s` is the rest. Per-seed rows are `seeds-general.csv`. Existing `seeds.csv`, `results.csv`, and the later rho = 14 and rho = 6 files are unchanged.

Anneal is `33728752`. The recommended preset is on, the long-range Mackay cap still runs when a second neighbour is bound, and the acceptance controller is on at every range: the step and the Metropolis temperature follow the acceptance, moves are drawn from the successes they produce, and a quiet record raises the temperature. A duplicate run of pair-distance fingerprints restarts from another member of that population. GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`.

| system | anneal | gmin | anneal evals | gmin evals |
| --- | --- | --- | --- | --- |
| m30r3 | 6/6 | 6/6 | 2446 | 780 |
| m30r6 | 6/6 | 6/6 | 10162 | 44195 |
| m30r10 | 5/6 | 0/6 | 212371 | 400000 |
| m30r14 | 6/6 | 0/6 | 136043 | 400000 |
| m45r3 | 6/6 | 6/6 | 1205 | 1061 |
| m45r6 | 6/6 | 6/6 | 11043 | 63991 |
| m45r10 | 2/6 | 0/6 | 1000000 | 1000000 |
| m45r14 | 2/6 | 0/6 | 1000000 | 1000000 |
| m60r3 | 6/6 | 6/6 | 2646 | 5671 |
| m60r6 | 6/6 | 6/6 | 53574 | 78177 |
| m60r10 | 4/6 | 1/6 | 215320 | 1000000 |
| m60r14 | 1/6 | 0/6 | 1000000 | 1000000 |
| m70r3 | 5/6 | 6/6 | 476055 | 144666 |
| m70r6 | 2/6 | 5/6 | 1000000 | 173674 |
| m70r10 | 2/6 | 0/6 | 1000000 | 1000000 |
| m70r14 | 0/6 | 0/6 | 1000000 | 1000000 |
| m80r3 | 6/6 | 6/6 | 332716 | 72146 |
| m80r6 | 4/6 | 6/6 | 1256314 | 251954 |
| m80r10 | 1/6 | 0/6 | 4000000 | 4000000 |
| m80r14 | 0/6 | 0/6 | 4000000 | 4000000 |

At rho = 6 anneal leads on evaluations through 60 atoms and is behind on hits at 70 (2/6 against 5/6) and 80 (4/6 against 6/6). At rho = 10 and 14 anneal has the hits GMIN does not, except the two sizes where both miss. At rho = 3 both hit, and GMIN's wall is lower once the cluster is large: the anneal time there is outside the potential.
