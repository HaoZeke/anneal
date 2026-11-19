# Shelf quarantine paired gate, Bonferroni addendum

This addendum does not replace the paired-gate verdict. It records a fresh LJ77 confirmation and the Bonferroni adjustment of the main table.

## LJ77 confirmation

Seeds 2012 through 2035, twenty-four seeds, budget 4000000, the same arm A and arm B builds and the same hit rule as the main comparison. One run at a time on core 0. Every row recorded affinity 0.

ABBA was checked by counting the arm that ran first on each seed, in the order the rows were written. The first-arm sequence is ABBAABBAABBAABBAABBAABBA: A opens 12 seeds and B opens 12 seeds.

| hits A | hits B | Fisher p | median calls A (95% CI) | median calls B (95% CI) | MW p | total calls A | total calls B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8/24 | 12/24 | 0.3801 | 3311177 [1818132, 3747331] | 2350329 [1528995, 3076264] | 0.2083 | 85894750 | 74472006 |

Fisher exact p is 0.3801. Mann-Whitney p on energy calls to a hit is 0.2083. Arm B recorded more hits than arm A, and the median calls to a hit are lower for B. The fewer-hits result on LJ77 in the main table is not repeated on these seeds.

## Bonferroni adjustment of the main table

The adjustment is over 56 systems x 2 tests. Adjusted p = min(1, raw p x 112). Raw p-values are the two-sided Fisher exact test on hits and the two-sided exact Mann-Whitney test on energy calls to a hit. A blank Mann-Whitney cell means neither arm hit, so that test is not defined. The main table has 57 systems. The smallest raw p is the LJ77 Fisher p, 0.0361. That p times 112 is greater than 1, and that p times 57 x 2 is also greater than 1, so every adjusted p below is 1.

| system | raw Fisher p | Bonferroni Fisher p | raw MW p | Bonferroni MW p |
| --- | ---: | ---: | ---: | ---: |
| lj38 | 1.0000 | 1.0000 | 0.4428 | 1.0000 |
| lj75 | 1.0000 | 1.0000 | 0.5556 | 1.0000 |
| lj76 | 1.0000 | 1.0000 | 0.6111 | 1.0000 |
| lj77 | 0.0361 | 1.0000 | 0.7111 | 1.0000 |
| lj78 | 1.0000 | 1.0000 | 0.2472 | 1.0000 |
| lj79 | 1.0000 | 1.0000 | 0.9228 | 1.0000 |
| lj80 | 1.0000 | 1.0000 | 0.9238 | 1.0000 |
| lj81 | 1.0000 | 1.0000 | 0.4859 | 1.0000 |
| lj82 | 1.0000 | 1.0000 | 0.4873 | 1.0000 |
| lj83 | 1.0000 | 1.0000 | 0.4521 | 1.0000 |
| lj84 | 1.0000 | 1.0000 | 0.8874 | 1.0000 |
| lj85 | 1.0000 | 1.0000 | 0.2969 | 1.0000 |
| lj86 | 1.0000 | 1.0000 | 0.6826 | 1.0000 |
| lj87 | 1.0000 | 1.0000 | 0.6075 | 1.0000 |
| lj88 | 1.0000 | 1.0000 | 0.9908 | 1.0000 |
| lj89 | 1.0000 | 1.0000 | 0.8428 | 1.0000 |
| lj90 | 1.0000 | 1.0000 | 0.6947 | 1.0000 |
| lj91 | 1.0000 | 1.0000 | 0.0934 | 1.0000 |
| lj92 | 1.0000 | 1.0000 | 0.4874 | 1.0000 |
| lj93 | 1.0000 | 1.0000 | 0.6539 | 1.0000 |
| lj94 | 1.0000 | 1.0000 | 0.5365 | 1.0000 |
| lj95 | 0.1930 | 1.0000 | 0.4278 | 1.0000 |
| lj96 | 1.0000 | 1.0000 | 0.8633 | 1.0000 |
| lj97 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| lj98 | 0.6668 | 1.0000 | 0.6806 | 1.0000 |
| lj99 | 0.1930 | 1.0000 | 0.3822 | 1.0000 |
| lj100 | 0.6802 | 1.0000 | 0.7546 | 1.0000 |
| lj101 | 0.6668 | 1.0000 | 1.0000 | 1.0000 |
| lj102 | 1.0000 | 1.0000 | 0.3095 | 1.0000 |
| lj103 | 0.6802 | 1.0000 | 0.5138 | 1.0000 |
| lj104 | 0.6843 | 1.0000 | 1.0000 | 1.0000 |
| m30r6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| m30r14 | 0.0932 | 1.0000 | 0.7777 | 1.0000 |
| m38r6 | 1.0000 | 1.0000 | 0.7662 | 1.0000 |
| m38r14 | 0.6802 | 1.0000 | 0.8762 | 1.0000 |
| m55r6 | 1.0000 | 1.0000 | 0.9664 | 1.0000 |
| m55r14 | 1.0000 | 1.0000 | 0.4967 | 1.0000 |
| m70r3 | 1.0000 | 1.0000 | 0.3397 | 1.0000 |
| m70r6 | 1.0000 | 1.0000 | 0.1944 | 1.0000 |
| m70r10 | 0.6802 | 1.0000 | 0.6095 | 1.0000 |
| m70r14 | 0.6404 | 1.0000 | 0.6190 | 1.0000 |
| m75r14 | 1.0000 | 1.0000 | 0.8095 | 1.0000 |
| m80r3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| m80r6 | 1.0000 | 1.0000 | 0.5995 | 1.0000 |
| m80r10 | 0.6802 | 1.0000 | 0.7546 | 1.0000 |
| m80r14 | 1.0000 | 1.0000 | 0.8182 | 1.0000 |
| au38 | 0.6802 | 1.0000 | 0.6104 | 1.0000 |
| cu6au32 | 1.0000 | 1.0000 |  |  |
| au55 | 1.0000 | 1.0000 | 0.2286 | 1.0000 |
| cu27au28 | 1.0000 | 1.0000 |  |  |
| cu19au19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| tip4p21 | 1.0000 | 1.0000 |  |  |

LJ77 raw Fisher p 0.0361 adjusts to 1.0000 and does not survive Bonferroni.
