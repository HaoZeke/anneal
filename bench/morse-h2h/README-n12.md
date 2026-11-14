# M70 and M80, twelve seeds

Seeds 2000-2011, serial, one core, hostname `bench`, working directory `/work`. Budgets are the paper budgets: 1e6 at 70 and 4e6 at 80. A hit is within 0.001 of the Cambridge table. Medians are Hyndman-Fan type 7. Fisher p-values are two-sided exact tests on the hit counts. Anneal with the acceptance controller on is `f562fac4`. The same binary with `H2H_ACCEPTANCE=0` is the flag-off arm. GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`.

Raw traces are not in git. Dropped from `/work/morse-general/runs`: 1093 files (240 logs, 240 row dumps, 120 GMIN `data`, 120 `coords`, 120 `lowest`, 73 `.dump`, 60 `.save`). The same classes of file under `/work/morse-vnext/runs-on` and `runs-off` are not committed.

## Flag on against GMIN

| system | anneal | GMIN | p | anneal evals | GMIN evals | anneal outside (s) | GMIN outside (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| m70r3 | 11/12 | 12/12 | 1 | 243607 | 205472 | 73.55 | 0.44 |
| m70r6 | 10/12 | 11/12 | 1 | 32782 | 214478 | 0.49 | 0.46 |
| m70r10 | 1/12 | 0/12 | 1 | 1000000 | 1000000 | 10.40 | 2.13 |
| m70r14 | 0/12 | 0/12 | 1 | 1000000 | 1000000 | 9.82 | 1.95 |
| m80r3 | 12/12 | 12/12 | 1 | 398124 | 114618 | 145.53 | 0.28 |
| m80r6 | 6/12 | 12/12 | 0.014 | 3181349 | 346246 | 44.62 | 0.84 |
| m80r10 | 0/12 | 0/12 | 1 | 4000000 | 4000000 | 60.57 | 9.52 |
| m80r14 | 1/12 | 0/12 | 1 | 4000000 | 4000000 | 39.06 | 8.73 |

Rho = 6 at 70 atoms is 10/12 against GMIN 11/12 (was 2/6 against 5/6 on six seeds). The median anneal cost there is 33k evaluations against GMIN's 214k. At 80 atoms and rho = 6, GMIN is 12/12 and anneal is 6/12.

## Flag on against flag off

| system | on | off | p |
| --- | --- | --- | --- |
| m70r3 | 11/12 | 8/12 | 0.32 |
| m70r6 | 10/12 | 7/12 | 0.37 |
| m70r10 | 1/12 | 3/12 | 0.59 |
| m70r14 | 0/12 | 5/12 | 0.037 |

Flag-off M80 did not finish. M80 rho = 3 had completed 5 of 12 seeds (3 hits) when this note was written. Each full-budget miss there takes about half an hour, and almost all of that is outside the potential.

## Rho = 3 outside the potential

On a flag-off M80 rho = 3 run of 20,000 evaluations the wall was 10.57 s, of which 0.60 s was inside the potential and 0.62 s was inside the quencher. Building the sorted-pair descriptor was 0.11 s (1725 calls). Comparing basins was 0.002 s (695 calls). The other ~9.4 s is the hop between quenches. A four-slice norm filter and a long-pair partial sum did not reduce that remainder, so they are not in the tree. At the paper budget the median outside time is 74 s at M70 rho = 3 and 146 s at M80 rho = 3, against 0.44 s and 0.28 s for GMIN. The potential time is the same order as GMIN's (5.8 s and 11.6 s).
