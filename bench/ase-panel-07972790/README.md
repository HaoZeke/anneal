Copyright (c) anneal developers

ASE panel at 07972790505f9609902dd503da78b8331dd0f3a8.

Anneal is the cluster preset. ASE 3.29.0 basin hopping and minima hopping use constructor defaults. The wheel sha256 is 7b9dd103f007810339c24acfee2f6b677c0c48443b21d3c98e52959246cf4ebf. Seeds 2000-2011, twelve runs an arm, one core. Budgets are 400000, 1000000, 4000000, 4000000, 4000000, and 1000000 force calls. A hit is within 0.001 of the reference. The ASE wall cap is three times the slowest cluster-preset wall of that cell in this run. A capped run is a miss counted at its wall. No cluster-preset run was capped. No run failed.

`analyze.py` reads `rows.csv` only. Medians of wall time, and of the time inside and outside each force evaluation, are in `summary.csv`.

| cell | anneal | basin hopping | minima hopping | basin stops | minima stops | p basin | p minima |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 10/12 | 0/12 | 0/12 | 12 | 12 | 6.73038e-05 | 6.73038e-05 |
| lj55 | 12/12 | 1/12 | 0/12 | 11 | 12 | 9.61483e-06 | 7.39602e-07 |
| lj98 | 4/12 | 0/12 | 0/12 | 0 | 12 | 0.0931677 | 0.0931677 |
| lj104 | 0/12 | 0/12 | 0/12 | 0 | 12 | 1 | 1 |
| lj150 | 0/12 | 0/12 | 0/12 | 0 | 12 | 1 | 1 |
| au55 | 0/12 | 0/12 | 0/12 | 12 | 12 | 1 | 1 |

Cluster-preset hits are 10, 12, 4, 0, 0, and 0 of 12. Basin hopping hit one LJ55 seed and no other seed. It spent the full budget on LJ98, LJ104, and LJ150, and stopped at the cap on every LJ38 and Au55 seed and on 11 LJ55 seeds. Minima hopping stopped at the cap on all 72 seeds and hit none. Fisher's exact test on hits is below 0.001 against both ASE arms on LJ38 and LJ55, 0.093 on LJ98, and 1 on LJ104, LJ150, and Au55, where no arm hit.
