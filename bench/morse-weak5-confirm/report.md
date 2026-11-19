# Held-stream confirmation run

Every number below is computed by `scripts/morse_weak5_confirm/harness.py report` from
`rows.csv`, `sessions.jsonl` and `hosts.csv` in this directory.

- Arm A: `Config::recommended(n)`. Arm B: `Config::recommended_adaptive(n)`, the held-stream fork.
  Arm A': `Config::recommended(n)` again on an independent seed set, the A/A control.
- A and B share the seed and the start geometry of each run. A' uses seeds of its own.
- One release binary, one process per run, one thread, pinned to one core, one run at a time.
  No warm-up, no retry, no replacement run, no reduced budget.
- Round `r` runs A, B, A' when `r` is even and A', B, A when `r` is odd.
- A hit is a quenched energy at or below the Cambridge minimum plus 0.001. A miss is charged the full budget.
- Median and quartiles are Hyndman-Fan type 7. Fisher is the two-sided exact test on hit counts.
  Mann-Whitney is two-sided, normal approximation with tie and continuity corrections, on calls with misses at the budget.
  McNemar is the exact test on the seeds where exactly one of A and B hit.
- The A/A band pools the 12 runs of A and the 12 of A' and splits them at random into halves of that size.
  The hit-count band is exact. The median-ratio bands are from 20000 draws of a fixed generator. A band is the 2.5th to 97.5th percentile of the first half against the second.

## Provenance

| field | value |
| --- | --- |
| base commit (patch under test) | `f1de90cba54a84ff9974deed7010ca2dce1bf4cb` |
| base tree | `d659af0159ea8554ebbf1033aa1310edf6f7dced` |
| measured commit | `bd97755f4ff979f4a3bbc137ce2b834664708381` |
| measured tree | `b76a219eb92918d63837cc3705399779529d6d85` |
| paths changed from base | `examples/morse_weak5_confirm.rs, scripts/morse_weak5_confirm/harness.py` |
| git status --porcelain at start | `empty` |
| binary sha256 | `1328cab29fd37fb6ae242e05ab91c5179d98727d85554b2228e048e7416d5750` |
| harness sha256 | `4c613cc6738749ebe325e00d30d5eb3cacbd1cbdcb74dd88335238d092bcabaa` |
| rustc | `rustc 1.88.0 (6b00bc388 2025-06-23)` |
| cargo | `cargo 1.88.0 (873a06493 2025-05-10)` |
| build | `cargo +1.88.0 build --release --locked --example morse_weak5_confirm --target-dir <target-dir>` |
| taskset | `taskset from util-linux 2.39.3` |
| python | `3.12.3` |
| environment given to each run | `MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1` |
| pinned core | `3` |
| harness affinity | `4,5,6` |
| host id (hash of the machine id) | `35967fedaf541948` |
| cpu | `Intel(R) Xeon(R) Processor` |
| logical cpus | `8` |
| memory kB | `49342816` |
| kernel | `6.12.113+` |
| os | `Ubuntu 24.04.5 LTS` |
| arch | `x86_64` |

Sessions:

| session | start | end | rows added | load average at start | smoke |
| --- | --- | --- | --- | --- | --- |
| 20261119T004600Z | 2026-11-19T00:49:49Z | 2026-11-19T03:54:25Z | 288 | 0.31 0.84 0.78 | no |

## Per cell and arm

| cell | arm | hits/n | calls, median [Q1, Q3] | calls to hit, hit runs only | wall s, median [Q1, Q3] | us per call, median [Q1, Q3] |
| --- | --- | --- | --- | --- | --- | --- |
| lj38 | A | 12/12 | 127840 [107994, 247239] | 127840 [107994, 247239] (n=12) | 0.64 [0.53, 1.31] | 5.03 [4.93, 5.14] |
|  | B | 12/12 | 127840 [107994, 247239] | 127840 [107994, 247239] (n=12) | 0.65 [0.53, 1.27] | 5.06 [5.01, 5.16] |
|  | A' | 11/12 | 80202 [33757, 153268] | 65653 [26942, 122786] (n=11) | 0.43 [0.19, 0.79] | 5.28 [5.06, 5.74] |
| m38r10 | A | 6/12 | 354138 [179025, 400000] | 170600 [104566, 202210] (n=6) | 3.37 [1.76, 3.92] | 9.69 [9.44, 9.85] |
|  | B | 6/12 | 354138 [179025, 400000] | 170600 [104566, 202210] (n=6) | 3.37 [1.71, 3.91] | 9.59 [9.44, 9.77] |
|  | A' | 5/12 | 400000 [206096, 400000] | 150489 [131805, 224631] (n=5) | 3.83 [1.97, 3.85] | 9.60 [9.56, 9.68] |
| lj75 | A | 5/12 | 4000000 [3037223, 4000000] | 2721351 [976196, 3142514] (n=5) | 58.8 [46.4, 62.1] | 15.4 [15.2, 15.7] |
|  | B | 12/12 | 1793308 [939516, 2262412] | 1793308 [939516, 2262412] (n=12) | 28.9 [16.2, 37.9] | 16.1 [15.5, 16.8] |
|  | A' | 5/12 | 4000000 [2072528, 4000000] | 1701232 [814750, 2196293] (n=5) | 57.9 [32.1, 59.8] | 15.0 [14.6, 15.8] |
| lj98 | A | 7/12 | 3166812 [1273270, 4000000] | 1430688 [582133, 2186692] (n=7) | 75.3 [30.6, 92.8] | 23.9 [23.4, 24.5] |
|  | B | 12/12 | 1298332 [758128, 2282344] | 1298332 [758128, 2282344] (n=12) | 33.6 [18.4, 58.8] | 25.5 [25.0, 26.2] |
|  | A' | 8/12 | 2460732 [1137898, 4000000] | 1302410 [943148, 2434920] (n=8) | 59.0 [27.6, 93.1] | 23.8 [23.4, 24.3] |
| m75r14 | A | 9/12 | 963277 [229274, 2014408] | 838982 [201620, 1077703] (n=9) | 30.4 [6.96, 62.8] | 31.2 [30.0, 32.4] |
|  | B | 12/12 | 779278 [229274, 1120324] | 779278 [229274, 1120324] (n=12) | 23.6 [6.96, 34.0] | 30.4 [29.9, 30.6] |
|  | A' | 10/12 | 1044010 [834707, 1871505] | 958480 [745482, 1321034] (n=10) | 32.1 [25.9, 58.6] | 30.8 [30.5, 31.5] |
| m80r14 | A | 6/12 | 3901429 [2553227, 4000000] | 2132943 [1242945, 3526746] (n=6) | 139.1 [91.1, 141.2] | 35.5 [35.3, 35.9] |
|  | B | 7/12 | 2529674 [2212186, 4000000] | 2305686 [1376910, 2384216] (n=7) | 85.7 [75.7, 137.0] | 34.3 [34.0, 34.4] |
|  | A' | 8/12 | 2849382 [2282084, 4000000] | 2722496 [957066, 2809545] (n=8) | 102.8 [80.8, 138.2] | 35.4 [34.8, 35.6] |
| m80r6 | A | 12/12 | 62677 [31442, 606910] | 62677 [31442, 606910] (n=12) | 2.25 [1.14, 22.8] | 37.6 [36.9, 37.9] |
|  | B | 12/12 | 62677 [31442, 536638] | 62677 [31442, 536638] (n=12) | 2.26 [1.15, 20.2] | 37.5 [36.8, 37.8] |
|  | A' | 12/12 | 75254 [43192, 328989] | 75254 [43192, 328989] (n=12) | 2.65 [1.50, 12.5] | 36.3 [34.9, 37.7] |
| m75r6 | A | 11/12 | 854804 [602440, 1415516] | 822536 [465934, 1145378] (n=11) | 28.6 [20.0, 50.0] | 33.5 [33.2, 33.9] |
|  | B | 11/12 | 736084 [474892, 1561116] | 723381 [380902, 1316356] (n=11) | 24.3 [15.7, 50.9] | 32.8 [32.4, 33.0] |
|  | A' | 12/12 | 981885 [282057, 1351852] | 981885 [282057, 1351852] (n=12) | 33.4 [9.37, 45.9] | 33.7 [33.4, 34.1] |

Calls count every run, a miss at its budget. Calls to hit count the runs that hit.

## B against A

| cell | A hits | B hits | Fisher p | hit only A / only B | McNemar p | Mann-Whitney p, calls | median calls B/A | median us per call B/A | same walk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 12/12 | 12/12 | 1.000 | 0 / 0 | 1.000 | 1.000 | 1.00 | 1.01 | 12/12 |
| m38r10 | 6/12 | 6/12 | 1.000 | 0 / 0 | 1.000 | 1.000 | 1.00 | 0.99 | 12/12 |
| lj75 | 5/12 | 12/12 | 0.005 | 0 / 7 | 0.016 | 0.007 | 0.45 | 1.04 | 1/12 |
| lj98 | 7/12 | 12/12 | 0.037 | 0 / 5 | 0.062 | 0.098 | 0.41 | 1.07 | 2/12 |
| m75r14 | 9/12 | 12/12 | 0.217 | 0 / 3 | 0.250 | 0.506 | 0.81 | 0.98 | 4/12 |
| m80r14 | 6/12 | 7/12 | 1.000 | 4 / 5 | 1.000 | 0.564 | 0.65 | 0.96 | 1/12 |
| m80r6 | 12/12 | 12/12 | 1.000 | 0 / 0 | 1.000 | 0.707 | 1.00 | 1.00 | 8/12 |
| m75r6 | 11/12 | 11/12 | 1.000 | 1 / 1 | 1.000 | 0.795 | 0.86 | 0.98 | 3/12 |

Same walk counts the seeds on which A and B charged the same calls and ended on the same energy bits.

## A against A', the A/A control

| cell | A hits | A' hits | Fisher p, A vs A' | Mann-Whitney p, A vs A' | hits A' - A | hits B - A | A/A band of the hit difference | B - A against the band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 12/12 | 11/12 | 1.000 | 0.069 | -1 | +0 | [-1, +1] | inside |
| m38r10 | 6/12 | 5/12 | 1.000 | 0.729 | -1 | +0 | [-5, +5] | inside |
| lj75 | 5/12 | 5/12 | 1.000 | 0.797 | +0 | +7 | [-4, +4] | OUTSIDE |
| lj98 | 7/12 | 8/12 | 1.000 | 0.836 | +1 | +5 | [-5, +5] | inside |
| m75r14 | 9/12 | 10/12 | 1.000 | 0.622 | +1 | +3 | [-3, +3] | inside |
| m80r14 | 6/12 | 8/12 | 0.680 | 0.294 | +2 | +1 | [-4, +4] | inside |
| m80r6 | 12/12 | 12/12 | 1.000 | 0.931 | +0 | +0 | [+0, +0] | inside |
| m75r6 | 11/12 | 12/12 | 1.000 | 0.931 | +1 | +0 | [-1, +1] | inside |

## Median ratios against the A/A band

| cell | calls A'/A | calls B/A | A/A band | B/A against the band | us per call A'/A | us per call B/A | A/A band | B/A against the band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | 0.63 | 1.00 | [0.53, 1.91] | inside | 1.05 | 1.01 | [0.94, 1.06] | inside |
| m38r10 | 1.13 | 1.00 | [0.54, 1.85] | inside | 0.99 | 0.99 | [0.98, 1.02] | inside |
| lj75 | 1.00 | 0.45 | [0.76, 1.32] | OUTSIDE | 0.98 | 1.04 | [0.96, 1.04] | OUTSIDE |
| lj98 | 0.78 | 0.41 | [0.37, 2.72] | inside | 0.99 | 1.07 | [0.97, 1.03] | OUTSIDE |
| m75r14 | 1.08 | 0.81 | [0.58, 1.74] | inside | 0.99 | 0.98 | [0.97, 1.04] | inside |
| m80r14 | 0.73 | 0.65 | [0.69, 1.45] | OUTSIDE | 1.00 | 0.96 | [0.99, 1.01] | OUTSIDE |
| m80r6 | 1.20 | 1.00 | [0.14, 7.10] | inside | 0.97 | 1.00 | [0.95, 1.05] | inside |
| m75r6 | 1.15 | 0.86 | [0.52, 1.89] | inside | 1.01 | 0.98 | [0.98, 1.02] | OUTSIDE |

Ratios are of medians. A band is what a split of A's own runs does by chance. A difference inside it is not separated from seed luck.

## Per seed

### lj38

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | hit | 124155 | hit | 124155 | yes |
| 2001 | hit | 111509 | hit | 111509 | yes |
| 2002 | hit | 293837 | hit | 293837 | yes |
| 2003 | hit | 174442 | hit | 174442 | yes |
| 2004 | hit | 131524 | hit | 131524 | yes |
| 2005 | hit | 69728 | hit | 69728 | yes |
| 2006 | hit | 99994 | hit | 99994 | yes |
| 2007 | hit | 110660 | hit | 110660 | yes |
| 2008 | hit | 325364 | hit | 325364 | yes |
| 2009 | hit | 317864 | hit | 317864 | yes |
| 2010 | hit | 231706 | hit | 231706 | yes |
| 2011 | hit | 49500 | hit | 49500 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | miss | 400000 |
| 6001 | hit | 13311 |
| 6002 | hit | 4923 |
| 6003 | hit | 166828 |
| 6004 | hit | 94751 |
| 6005 | hit | 40572 |
| 6006 | hit | 96823 |
| 6007 | hit | 148748 |
| 6008 | hit | 64144 |
| 6009 | hit | 12955 |
| 6010 | hit | 65653 |
| 6011 | hit | 191829 |

### m38r10

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | miss | 400000 | miss | 400000 | yes |
| 2001 | hit | 308277 | hit | 308277 | yes |
| 2002 | hit | 207130 | hit | 207130 | yes |
| 2003 | miss | 400000 | miss | 400000 | yes |
| 2004 | miss | 400000 | miss | 400000 | yes |
| 2005 | miss | 400000 | miss | 400000 | yes |
| 2006 | hit | 153749 | hit | 153749 | yes |
| 2007 | hit | 72458 | hit | 72458 | yes |
| 2008 | miss | 400000 | miss | 400000 | yes |
| 2009 | miss | 400000 | miss | 400000 | yes |
| 2010 | hit | 187450 | hit | 187450 | yes |
| 2011 | hit | 88171 | hit | 88171 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 131805 |
| 6001 | miss | 400000 |
| 6002 | hit | 224631 |
| 6003 | miss | 400000 |
| 6004 | miss | 400000 |
| 6005 | miss | 400000 |
| 6006 | miss | 400000 |
| 6007 | hit | 83381 |
| 6008 | hit | 297173 |
| 6009 | hit | 150489 |
| 6010 | miss | 400000 |
| 6011 | miss | 400000 |

### lj75

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | miss | 4000000 | hit | 983966 | no |
| 2001 | hit | 3327845 | hit | 1777397 | no |
| 2002 | miss | 4000000 | hit | 3387932 | no |
| 2003 | miss | 4000000 | hit | 2227827 | no |
| 2004 | miss | 4000000 | hit | 806164 | no |
| 2005 | hit | 2721351 | hit | 2244503 | no |
| 2006 | miss | 4000000 | hit | 529823 | no |
| 2007 | miss | 4000000 | hit | 2316140 | no |
| 2008 | hit | 976196 | hit | 1331842 | no |
| 2009 | miss | 4000000 | hit | 1809218 | no |
| 2010 | hit | 3142514 | hit | 3226304 | no |
| 2011 | hit | 290074 | hit | 290074 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 2196293 |
| 6001 | miss | 4000000 |
| 6002 | miss | 4000000 |
| 6003 | miss | 4000000 |
| 6004 | miss | 4000000 |
| 6005 | miss | 4000000 |
| 6006 | miss | 4000000 |
| 6007 | hit | 1701232 |
| 6008 | hit | 814750 |
| 6009 | hit | 46837 |
| 6010 | miss | 4000000 |
| 6011 | hit | 3029173 |

### lj98

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | hit | 363251 | hit | 363251 | yes |
| 2001 | hit | 801015 | hit | 2823612 | no |
| 2002 | hit | 1888987 | hit | 1106537 | no |
| 2003 | hit | 103862 | hit | 103862 | yes |
| 2004 | miss | 4000000 | hit | 1490127 | no |
| 2005 | hit | 1430688 | hit | 1653231 | no |
| 2006 | hit | 3849225 | hit | 3169899 | no |
| 2007 | miss | 4000000 | hit | 3312151 | no |
| 2008 | hit | 2484398 | hit | 801498 | no |
| 2009 | miss | 4000000 | hit | 761181 | no |
| 2010 | miss | 4000000 | hit | 748971 | no |
| 2011 | miss | 4000000 | hit | 2101921 | no |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | miss | 4000000 |
| 6001 | hit | 2543465 |
| 6002 | hit | 1450422 |
| 6003 | miss | 4000000 |
| 6004 | hit | 2512355 |
| 6005 | miss | 4000000 |
| 6006 | miss | 4000000 |
| 6007 | hit | 1154399 |
| 6008 | hit | 507411 |
| 6009 | hit | 2409108 |
| 6010 | hit | 1088393 |
| 6011 | hit | 170689 |

### m75r14

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | miss | 4000000 | hit | 3669515 | no |
| 2001 | hit | 1077703 | hit | 691903 | no |
| 2002 | hit | 1112778 | hit | 1334759 | no |
| 2003 | hit | 177756 | hit | 177756 | yes |
| 2004 | hit | 51299 | hit | 51299 | yes |
| 2005 | hit | 201620 | hit | 201620 | yes |
| 2006 | miss | 4000000 | hit | 1460206 | no |
| 2007 | hit | 848851 | hit | 866652 | no |
| 2008 | hit | 838982 | hit | 999882 | no |
| 2009 | miss | 4000000 | hit | 575390 | no |
| 2010 | hit | 238492 | hit | 238492 | yes |
| 2011 | hit | 1352544 | hit | 1048845 | no |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 134553 |
| 6001 | hit | 896118 |
| 6002 | hit | 1067176 |
| 6003 | miss | 4000000 |
| 6004 | hit | 2531897 |
| 6005 | miss | 4000000 |
| 6006 | hit | 546500 |
| 6007 | hit | 879319 |
| 6008 | hit | 1651374 |
| 6009 | hit | 1405654 |
| 6010 | hit | 1020843 |
| 6011 | hit | 700870 |

### m80r14

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | hit | 3711158 | hit | 2305686 | no |
| 2001 | hit | 529680 | miss | 4000000 | no |
| 2002 | miss | 4000000 | hit | 2354570 | no |
| 2003 | hit | 1226468 | miss | 4000000 | no |
| 2004 | miss | 4000000 | hit | 1931688 | no |
| 2005 | miss | 4000000 | hit | 2413861 | no |
| 2006 | hit | 3802858 | miss | 4000000 | no |
| 2007 | hit | 1292375 | hit | 822133 | no |
| 2008 | miss | 4000000 | hit | 2645488 | no |
| 2009 | miss | 4000000 | hit | 751720 | no |
| 2010 | hit | 2973511 | miss | 4000000 | no |
| 2011 | miss | 4000000 | miss | 4000000 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 1076167 |
| 6001 | hit | 2929058 |
| 6002 | hit | 599761 |
| 6003 | hit | 2760936 |
| 6004 | miss | 4000000 |
| 6005 | hit | 463533 |
| 6006 | miss | 4000000 |
| 6007 | hit | 2684057 |
| 6008 | miss | 4000000 |
| 6009 | hit | 2769707 |
| 6010 | hit | 3421204 |
| 6011 | miss | 4000000 |

### m80r6

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | hit | 37038 | hit | 37038 | yes |
| 2001 | hit | 14653 | hit | 14653 | yes |
| 2002 | hit | 673934 | hit | 536262 | no |
| 2003 | hit | 1461617 | hit | 539962 | no |
| 2004 | hit | 7052 | hit | 7052 | yes |
| 2005 | hit | 9273 | hit | 9273 | yes |
| 2006 | hit | 53696 | hit | 53696 | yes |
| 2007 | hit | 584568 | hit | 537764 | no |
| 2008 | hit | 1216668 | hit | 607702 | no |
| 2009 | hit | 407293 | hit | 407293 | yes |
| 2010 | hit | 71658 | hit | 71658 | yes |
| 2011 | hit | 50314 | hit | 50314 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 97585 |
| 6001 | hit | 21762 |
| 6002 | hit | 87339 |
| 6003 | hit | 50912 |
| 6004 | hit | 341687 |
| 6005 | hit | 42266 |
| 6006 | hit | 43501 |
| 6007 | hit | 324756 |
| 6008 | hit | 912643 |
| 6009 | hit | 63168 |
| 6010 | hit | 744784 |
| 6011 | hit | 5310 |

### m75r6

| seed | A | A calls | B | B calls | same walk |
| --- | --- | --- | --- | --- | --- |
| 2000 | hit | 1824606 | hit | 1793719 | no |
| 2001 | hit | 738946 | hit | 723381 | no |
| 2002 | miss | 4000000 | hit | 748787 | no |
| 2003 | hit | 769569 | hit | 2322365 | no |
| 2004 | hit | 1722822 | hit | 1149130 | no |
| 2005 | hit | 1313081 | miss | 4000000 | no |
| 2006 | hit | 977675 | hit | 568881 | no |
| 2007 | hit | 109238 | hit | 109238 | yes |
| 2008 | hit | 822536 | hit | 610662 | no |
| 2009 | hit | 887072 | hit | 1483582 | no |
| 2010 | hit | 162399 | hit | 162399 | yes |
| 2011 | hit | 192923 | hit | 192923 | yes |

| A' seed | A' | A' calls |
| --- | --- | --- |
| 6000 | hit | 867639 |
| 6001 | hit | 138591 |
| 6002 | hit | 302632 |
| 6003 | hit | 3720233 |
| 6004 | hit | 1186159 |
| 6005 | hit | 159769 |
| 6006 | hit | 1503778 |
| 6007 | hit | 1096131 |
| 6008 | hit | 793427 |
| 6009 | hit | 220333 |
| 6010 | hit | 1301210 |
| 6011 | hit | 3723755 |

## Noise diagnostics

Switches, wait time and cpu time are the process's own account over the search. Core busy time is what the
pinned core did while the process ran, from `/proc/stat` in 10 ms ticks, and steal is what the hypervisor took.
A core busy time near the process cpu time means nothing else ran on the core.

| cell | arm | involuntary switches, total | involuntary switches, worst run | wait / cpu time, total, % | cpu time / wall, total | core busy time / cpu time, total | steal ticks, total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lj38 | A | 62 | 10 | 0.0168 | 0.9990 | 1.000 | 0 |
|  | B | 61 | 10 | 0.0070 | 0.9978 | 1.007 | 0 |
|  | A' | 47 | 12 | 0.0089 | 0.9977 | 1.008 | 0 |
| m38r10 | A | 176 | 23 | 0.0090 | 0.9993 | 1.000 | 1 |
|  | B | 179 | 24 | 0.0059 | 0.9995 | 1.001 | 0 |
|  | A' | 199 | 23 | 0.0061 | 0.9996 | 1.001 | 0 |
| lj75 | A | 3159 | 367 | 0.0076 | 0.9998 | 1.000 | 7 |
|  | B | 1753 | 265 | 0.0078 | 0.9998 | 1.000 | 4 |
|  | A' | 2888 | 350 | 0.0076 | 0.9998 | 1.001 | 5 |
| lj98 | A | 3940 | 579 | 0.0065 | 0.9998 | 1.000 | 4 |
|  | B | 2509 | 455 | 0.0076 | 0.9998 | 1.000 | 3 |
|  | A' | 3459 | 510 | 0.0064 | 0.9999 | 1.000 | 6 |
| m75r14 | A | 2998 | 719 | 0.0061 | 0.9999 | 1.000 | 1 |
|  | B | 1805 | 592 | 0.0161 | 0.9998 | 1.000 | 2 |
|  | A' | 3140 | 693 | 0.0058 | 0.9998 | 1.000 | 6 |
| m80r14 | A | 6993 | 765 | 0.0056 | 0.9999 | 1.000 | 5 |
|  | B | 5917 | 729 | 0.0052 | 0.9999 | 1.000 | 8 |
|  | A' | 6005 | 757 | 0.0054 | 0.9999 | 1.000 | 2 |
| m80r6 | A | 913 | 288 | 0.0059 | 0.9998 | 1.001 | 1 |
|  | B | 578 | 119 | 0.0054 | 0.9998 | 1.001 | 1 |
|  | A' | 522 | 178 | 0.0036 | 0.9999 | 1.000 | 0 |
| m75r6 | A | 2383 | 712 | 0.0055 | 0.9999 | 1.000 | 3 |
|  | B | 2363 | 702 | 0.0050 | 0.9999 | 1.000 | 2 |
|  | A' | 2690 | 702 | 0.0056 | 0.9999 | 1.000 | 1 |

## Checks

- PASS: every session started and ended (1 start, 1 end, 0 abort)
- PASS: no smoke session in the data
- PASS: worktree clean at every start and end
- PASS: one code_revision in the rows (bd97755f4ff979f4a3bbc137ce2b834664708381)
- PASS: one code_tree in the rows (b76a219eb92918d63837cc3705399779529d6d85)
- PASS: one binary_sha256 in the rows (1328cab29fd37fb6ae242e05ab91c5179d98727d85554b2228e048e7416d5750)
- PASS: one host_id in the rows (35967fedaf541948)
- PASS: one base_revision across sessions (f1de90cba54a84ff9974deed7010ca2dce1bf4cb)
- PASS: one base_tree across sessions (d659af0159ea8554ebbf1033aa1310edf6f7dced)
- PASS: one toolchain across sessions (1.88.0)
- PASS: one core across sessions (3)
- PASS: one harness_sha256 across sessions (4c613cc6738749ebe325e00d30d5eb3cacbd1cbdcb74dd88335238d092bcabaa)
- PASS: every run pinned to the one core, one thread
- PASS: one row per cell, arm and seed
- PASS: the same number of runs in every arm and cell (12)
- PASS: A and B share seeds and starts
- PASS: A' seeds are disjoint from A seeds
- PASS: A and A' share one config hash per cell and B has another
- PASS: arm A ran the preset it names ([(0, 0, 1)])
- PASS: arm B ran the preset it names ([(1, 1, 2)])
- PASS: arm A' ran the preset it names ([(0, 0, 1)])
- PASS: interleave: even rounds A>B>AA, odd rounds AA>B>A
- PASS: the table of cells matches the driver's report
- PASS: below the fork allowance B is the same walk as A, seed by seed (lj38 m38r10)
- PASS: where A hit inside the fork allowance, B is the same walk (36 seeds)
