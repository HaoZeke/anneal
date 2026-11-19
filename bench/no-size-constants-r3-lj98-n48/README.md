# LJ98 n=48 paired comparison

n=48 outcome: `FAIL`. n=60 outcome: `FAIL`.

## Pre-set rule

PASS if the two-sided Mann-Whitney p is at least 0.05 and B has no fewer hits than A, or B is better. B is better when it has strictly more hits, or the same number of hits and a strictly lower median evals_to_hit. Otherwise FAIL. A hit drop is not treated as B being better. The outside_s/charged ratio is reported and is not part of the rule.

## Cell

System `lj98`. Budget `4000000`. Reference `-543.6653610000`. Tolerance `0.001`. These three values are the lj98 cell of the gate rows, and every n=48 row carries the same three.

Arm A revision `dd054749d4703923f4cf4c34f50589c574ee1276`, tree `f1705d6f02f7f257f50e831c16057740283efea5`. Arm B revision `33c799031109a32ff9f3e61e9cdd94774adf54c6`, tree `3fc07c20d323a7ae5b34304720da7d7656309ba4`. Hashed host id `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

## Protocol

- One serial session. No warm-up run and no second pass over a finished seed.
- Seeds 2012-2059 (48 values), disjoint from the gate seeds 2000-2011.
- Even seeds run A then B. Odd seeds run B then A. From seed 2012 that sequence is A, B, B, A, and it repeats.
- Each arm and seed is a fresh process. OpenMP, OpenBLAS, MKL, BLIS, VECLIB, NUMBA and Rayon threads are 1. The process is pinned to CPU core 0.
- Both arms are release builds. The preset is `recommended`. The stop level is the reference plus the tolerance.
- Median is the type-7 sample quantile. Fisher exact is two-sided. Mann-Whitney is two-sided with tie and continuity corrections; U is arm B against arm A.
- outside_s/charged is computed per row. The arm value is the median of those ratios.

## n=48

Hits 45/48 (A) and 38/48 (B), Fisher exact p = 0.0704367. Median evals_to_hit 1317269.5 (A) and 1451825.0 (B), Mann-Whitney U (B vs A) = 1387.5, two-sided p = 0.0846825. Median outside_s/charged 1.66285e-05 (A) and 1.71005e-05 (B), ratio B/A = 1.02839. Hit drop = 1. B better = 0. Outcome = FAIL.

## Pooled n=60

The 48 rows above plus the 12 lj98 rows of the gate rows file, seeds 2000-2011, unchanged.

Hits 55/60 (A) and 47/60 (B), Fisher exact p = 0.0713813. Median evals_to_hit 1321410.0 (A) and 1588222.0 (B), Mann-Whitney U (B vs A) = 2232.0, two-sided p = 0.0232915. Median outside_s/charged 1.61954e-05 (A) and 1.67051e-05 (B), ratio B/A = 1.03148. Hit drop = 1. B better = 0. Outcome = FAIL.

## Per-seed evals_to_hit

Columns are cohort, seed, evals_to_hit A, evals_to_hit B. The same table is `per-seed.csv`.

| cohort | seed | evals A | evals B |
| --- | --- | --- | --- |
| gate | 2000 | 2011832 | 1530581 |
| gate | 2001 | 682516 | 1415892 |
| gate | 2002 | 4000000 | 4000000 |
| gate | 2003 | 1333413 | 510294 |
| gate | 2004 | 755416 | 4000000 |
| gate | 2005 | 1309407 | 3639629 |
| gate | 2006 | 4000000 | 838485 |
| gate | 2007 | 2263275 | 2767086 |
| gate | 2008 | 644482 | 4000000 |
| gate | 2009 | 2940139 | 2327304 |
| gate | 2010 | 79220 | 3401981 |
| gate | 2011 | 217513 | 2356138 |
| n48 | 2012 | 1371987 | 368132 |
| n48 | 2013 | 119346 | 3018686 |
| n48 | 2014 | 6862 | 6862 |
| n48 | 2015 | 2802242 | 4000000 |
| n48 | 2016 | 3365167 | 4000000 |
| n48 | 2017 | 218435 | 484854 |
| n48 | 2018 | 1620329 | 985299 |
| n48 | 2019 | 2654469 | 711474 |
| n48 | 2020 | 1446646 | 1450991 |
| n48 | 2021 | 1941207 | 1548397 |
| n48 | 2022 | 692187 | 886729 |
| n48 | 2023 | 1076900 | 4000000 |
| n48 | 2024 | 26436 | 1981476 |
| n48 | 2025 | 155462 | 4000000 |
| n48 | 2026 | 2901491 | 889154 |
| n48 | 2027 | 213863 | 425504 |
| n48 | 2028 | 2695025 | 679087 |
| n48 | 2029 | 1745021 | 1117126 |
| n48 | 2030 | 7842 | 7842 |
| n48 | 2031 | 2376636 | 870552 |
| n48 | 2032 | 105397 | 3578049 |
| n48 | 2033 | 1682441 | 4000000 |
| n48 | 2034 | 1784783 | 586819 |
| n48 | 2035 | 379631 | 4000000 |
| n48 | 2036 | 266339 | 1016292 |
| n48 | 2037 | 1472151 | 814813 |
| n48 | 2038 | 1217587 | 2784038 |
| n48 | 2039 | 414856 | 1899184 |
| n48 | 2040 | 90736 | 4000000 |
| n48 | 2041 | 2449783 | 1990998 |
| n48 | 2042 | 834595 | 4000000 |
| n48 | 2043 | 4000000 | 4000000 |
| n48 | 2044 | 533324 | 7877 |
| n48 | 2045 | 19713 | 19713 |
| n48 | 2046 | 1787976 | 2442604 |
| n48 | 2047 | 2593934 | 1628047 |
| n48 | 2048 | 1544152 | 4000000 |
| n48 | 2049 | 199704 | 447873 |
| n48 | 2050 | 1262552 | 2713403 |
| n48 | 2051 | 4000000 | 3088527 |
| n48 | 2052 | 2088461 | 1342859 |
| n48 | 2053 | 4000000 | 510915 |
| n48 | 2054 | 69255 | 225663 |
| n48 | 2055 | 2022010 | 2318687 |
| n48 | 2056 | 251655 | 744560 |
| n48 | 2057 | 40472 | 3095872 |
| n48 | 2058 | 2083343 | 1452659 |
| n48 | 2059 | 397057 | 883703 |
