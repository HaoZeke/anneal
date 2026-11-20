# Morse rho = 14, pair-scaled preset

Arm A is commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`, the default preset, with no H2H variable set.
Arm B is tree `70871f7307271806926b2aec9870e838dffbe924` with `H2H_PAIR_SCALE=1` and no other H2H variable. That switch selects the pair-scaled preset.
Seeds 2100-2123 are twenty-four fresh seeds per arm per cell. Both arms of a seed run the geometry arm A writes for that seed.
m55r14 has budget 1000000 and reference -220.646208. m80r14 has budget 4000000 and reference -340.811371.
A hit is a finite best energy at or below the reference plus 0.001.
Inside each cell the order is ABBA. An even offset from 2100 runs A then B, and an odd offset runs B then A. Slot 0 is the first arm of the seed. m55r14 runs before m80r14.
The measured process tree is pinned to core 0. Every thread-pool variable is 1. The binaries are release builds. No run uses a shortened budget, and there is no warm-up search.
Medians are Hyndman-Fan type 7. The all-seeds call median uses the recorded calls, and a miss is recorded at its budget.
Fisher is the two-sided exact test on the hit table. Mann-Whitney is two-sided, using the normal approximation with the tie correction and the continuity correction.
For that ranking a hit keeps its call count and a miss scores one past the budget, so every miss ranks worse than every hit.
Bonferroni uses m = 2 inside the Fisher family and m = 2 inside the Mann-Whitney family. The threshold in each family is 0.025.
The recorded best is the example's printed energy, eight digits after the decimal, reparsed to binary64 and written so the decimal round-trips. best_bits is the IEEE-754 layout of that reparsed value.

## Balance

Counts of (cell, slot, arm) in the executed sessions:

| cell | slot | arm | count |
| --- | --- | --- | --- |
| m55r14 | 0 | A | 12 |
| m55r14 | 0 | B | 12 |
| m55r14 | 1 | A | 12 |
| m55r14 | 1 | B | 12 |
| m80r14 | 0 | A | 12 |
| m80r14 | 0 | B | 12 |
| m80r14 | 1 | A | 12 |
| m80r14 | 1 | B | 12 |

Pooled over both cells:

- slot 0, arm A: 24
- slot 0, arm B: 24
- slot 1, arm A: 24
- slot 1, arm B: 24

Hashed machine id `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

## m55r14

Budget 1000000. Reference -220.646208. n = 24 per arm.

| quantity | A | B | B minus A | direction |
| --- | --- | --- | --- | --- |
| hits | 15 | 17 | 2 | B more hits |
| median calls, hits only | 357025 | 474458 | 117433 | A lower |
| median calls, all seeds | 714754.5 | 605087.5 | -109667 | B lower |
| median wall, seconds | 12.831637 | 10.235633 | -2.596004 | B lower |

Fisher p = 1713931321806/2254848913647 = 0.76010916. The hit difference is B more hits. The Fisher test does not clear the Bonferroni threshold 0.025.

Mann-Whitney U of A = 288.5, p = 1. Mean rank A = 24.5208, mean rank B = 24.4792, B minus A = -0.041667. The call-rank difference is B lower. The Mann-Whitney test does not clear the Bonferroni threshold 0.025.

## m80r14

Budget 4000000. Reference -340.811371. n = 24 per arm.

| quantity | A | B | B minus A | direction |
| --- | --- | --- | --- | --- |
| hits | 16 | 14 | -2 | A more hits |
| median calls, hits only | 2421686.5 | 733314.5 | -1688372 | B lower |
| median calls, all seeds | 3076442.5 | 3149163 | 72720.5 | A lower |
| median wall, seconds | 116.270224 | 111.603913 | -4.666312 | B lower |

Fisher p = 5600270291088/7309837001104 = 0.76612793. The hit difference is A more hits. The Fisher test does not clear the Bonferroni threshold 0.025.

Mann-Whitney U of A = 307, p = 0.69512554. Mean rank A = 25.2917, mean rank B = 23.7083, B minus A = -1.583333. The call-rank difference is B lower. The Mann-Whitney test does not clear the Bonferroni threshold 0.025.

## Per-seed hits

### m55r14

| seed | A hit | B hit | A calls | B calls |
| --- | --- | --- | --- | --- |
| 2100 | 1 | 0 | 21417 | 1000000 |
| 2101 | 1 | 1 | 447094 | 223557 |
| 2102 | 1 | 1 | 357025 | 925745 |
| 2103 | 0 | 1 | 1000000 | 277721 |
| 2104 | 1 | 1 | 75923 | 401175 |
| 2105 | 1 | 1 | 190133 | 472125 |
| 2106 | 0 | 1 | 1000000 | 706593 |
| 2107 | 1 | 1 | 697687 | 34666 |
| 2108 | 1 | 1 | 247629 | 474458 |
| 2109 | 0 | 1 | 1000000 | 43944 |
| 2110 | 1 | 1 | 348304 | 195937 |
| 2111 | 1 | 0 | 731822 | 1000000 |
| 2112 | 1 | 0 | 211276 | 1000000 |
| 2113 | 0 | 0 | 1000000 | 1000000 |
| 2114 | 1 | 1 | 963899 | 674440 |
| 2115 | 0 | 1 | 1000000 | 916964 |
| 2116 | 0 | 0 | 1000000 | 1000000 |
| 2117 | 0 | 1 | 1000000 | 511284 |
| 2118 | 1 | 1 | 957949 | 535735 |
| 2119 | 1 | 0 | 463260 | 1000000 |
| 2120 | 1 | 1 | 270821 | 497137 |
| 2121 | 0 | 0 | 1000000 | 1000000 |
| 2122 | 1 | 1 | 431702 | 469565 |
| 2123 | 0 | 1 | 1000000 | 815416 |

### m80r14

| seed | A hit | B hit | A calls | B calls |
| --- | --- | --- | --- | --- |
| 2100 | 1 | 0 | 3307751 | 4000000 |
| 2101 | 0 | 0 | 4000000 | 4000000 |
| 2102 | 0 | 1 | 4000000 | 269315 |
| 2103 | 1 | 1 | 2397983 | 750172 |
| 2104 | 1 | 0 | 197909 | 4000000 |
| 2105 | 1 | 1 | 1641720 | 541567 |
| 2106 | 1 | 1 | 2473044 | 3795921 |
| 2107 | 1 | 1 | 2445390 | 3112218 |
| 2108 | 0 | 0 | 4000000 | 4000000 |
| 2109 | 0 | 1 | 4000000 | 1580422 |
| 2110 | 1 | 0 | 3476455 | 4000000 |
| 2111 | 0 | 1 | 4000000 | 279324 |
| 2112 | 1 | 1 | 2555891 | 563910 |
| 2113 | 0 | 0 | 4000000 | 4000000 |
| 2114 | 1 | 1 | 1884998 | 716457 |
| 2115 | 0 | 1 | 4000000 | 887324 |
| 2116 | 1 | 0 | 870082 | 4000000 |
| 2117 | 1 | 1 | 1601348 | 495793 |
| 2118 | 1 | 1 | 2889540 | 1639063 |
| 2119 | 0 | 0 | 4000000 | 4000000 |
| 2120 | 1 | 0 | 3263345 | 4000000 |
| 2121 | 1 | 1 | 43468 | 3186108 |
| 2122 | 1 | 0 | 3516385 | 4000000 |
| 2123 | 1 | 1 | 869874 | 608432 |

## Identity

- base commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`
- base tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- arm A commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- arm B commit `f14713c3ad2115ba3134dc7b04de553b9695b946` tree `70871f7307271806926b2aec9870e838dffbe924`
- rustc `rustc 1.88.0 (6b00bc388 2025-06-23)`
