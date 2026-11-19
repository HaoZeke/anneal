# Morse weak8 paired gate

Verdict: **PASS**

PASS only when the paired runs are complete and B has no cell with fewer hits at two-sided Fisher p<0.05 and no cell with a higher hit-call median at two-sided Mann-Whitney p<0.05 above the A/A' bootstrap band.

Fresh-ledger claim: **HOLDS**.

Config::recommended is one replica with the population search off, and each run starts from Ledger::new. All eight cells use that preset, so all eight are fresh-ledger default trajectories. Only an adaptive, replica-id filtered search is allowed to move.

- A and A' are separate release builds of the base tree. B is a release build of the patched tree, recorded as the base revision plus that tree.
- All arms use Config::recommended, the paper budgets, and the same 12 seeds and starts per cell.
- Arm order is a balanced flip: rotations of A, A', B, then those orders reversed, repeated across the twelve seeds.
- Runs are serial, one thread, pinned to one core, with no warm-up, retry, replacement, quick mode, or reduced budget.
- Calls-to-hit statistics include hitting runs only. Confidence intervals are 95% percentile bootstrap intervals with 20000 fixed-stream draws.
- The calls gate compares B/A's hit-call median ratio with the 95% bootstrap band of A'/A.
- Resume: no.

## Provenance

| field | value |
| --- | --- |
| base revision | `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` |
| base tree | `7a2ba49b37241a8fe42b2c45c226374de6d92274` |
| harness revision | `6bdaa33ca7442d0bf25a51314627d73cb245db38` |
| harness tree | `e789ff1bcff7ce5e34d6fd01da9627544963377c` |
| host id | `417ad3ecfabd2aa8` |
| cpu | `Intel(R) Xeon(R) Processor` |
| logical cpus | `8` |
| memory kB | `49342816` |
| kernel | `6.12.113+` |
| os | `Ubuntu 24.04.5 LTS` |
| arch | `x86_64` |
| rustc | `rustc 1.88.0 (6b00bc388 2025-06-23)` |
| cargo | `cargo 1.88.0 (873a06493 2025-05-10)` |
| python | `3.12.3` |
| taskset | `taskset from util-linux 2.39.3` |
| pinned core | `0` |
| run environment | `LJOS_SEAT=benchbot MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1` |

Arm builds:

| arm | source revision | source tree | binary sha256 | porcelain before | porcelain after |
| --- | --- | --- | --- | --- | --- |
| A | 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 | 7a2ba49b37241a8fe42b2c45c226374de6d92274 | 22226deaaf3f48d52c2caa6d2f386a936558e825af1908e219a9c9fe63829b83 | empty | empty |
| A' | 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 | 7a2ba49b37241a8fe42b2c45c226374de6d92274 | b8c2826673b227632a82686670c787fcf623525d8836c3fcb3e452ddc3310ae9 | empty | empty |
| B | 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 | e45fa8e672477cad76454b61c36498ee027d7340 | d6c14e8ed4f2ee1f4fbb11590c156f221aeaa0a305dc15ccaee252a4afcb45cf | empty | empty |

## Order balance

| slot | A | A' | B |
| --- | --- | --- | --- |
| 0 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 |
| 2 | 32 | 32 | 32 |

## Per-cell results

| cell | A hits | A' hits | B hits | Fisher B/A | Fisher A'/A | A hit calls median [boot CI] | A' hit calls median [boot CI] | B hit calls median [boot CI] | MW B/A | MW A'/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | 5/12 | 5/12 | 5/12 | 1.000 | 1.000 | 2721351 [290074, 3327845] (n=5) | 2721351 [290074, 3327845] (n=5) | 2721351 [290074, 3327845] (n=5) | 1.000 | 1.000 |
| LJ98 | 7/12 | 7/12 | 7/12 | 1.000 | 1.000 | 1430688 [363251, 2484398] (n=7) | 1430688 [363251, 2484398] (n=7) | 1430688 [363251, 2484398] (n=7) | 1.000 | 1.000 |
| M75 rho 6 | 11/12 | 11/12 | 11/12 | 1.000 | 1.000 | 822536 [192923, 1313081] (n=11) | 822536 [192923, 1313081] (n=11) | 822536 [192923, 1313081] (n=11) | 1.000 | 1.000 |
| M75 rho 14 | 9/12 | 9/12 | 9/12 | 1.000 | 1.000 | 838982 [177756, 1112778] (n=9) | 838982 [177756, 1112778] (n=9) | 838982 [177756, 1112778] (n=9) | 1.000 | 1.000 |
| M80 rho 6 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 62677 [25846, 629251] (n=12) | 62677 [25846, 629251] (n=12) | 62677 [25846, 629251] (n=12) | 1.000 | 1.000 |
| M80 rho 14 | 6/12 | 6/12 | 6/12 | 1.000 | 1.000 | 2132943 [878074, 3757008] (n=6) | 2132943 [878074, 3757008] (n=6) | 2132943 [878074, 3757008] (n=6) | 1.000 | 1.000 |
| LJ38 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 1.000 | 1.000 |
| m38r10 | 6/12 | 6/12 | 6/12 | 1.000 | 1.000 | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 1.000 | 1.000 |

Totals, wall medians, A/A' bands, and bit identity:

| cell | total charged calls A / A' / B | wall median s A / A' / B | hit-call median B/A | A/A' bootstrap band | A/B energy bits and call counts identical |
| --- | --- | --- | --- | --- | --- |
| LJ75 | 38457980 / 38457980 / 38457980 | 56.76 / 56.79 / 56.84 | 1.000 | [0.107, 9.382] | 12/12 |
| LJ98 | 30921843 / 30921843 / 30921843 | 74.37 / 74.48 / 74.37 | 1.000 | [0.192, 5.200] | 12/12 |
| M75 rho 6 | 13520867 / 13520867 / 13520867 | 28.81 / 28.68 / 28.75 | 1.000 | [0.235, 4.264] | 12/12 |
| M75 rho 14 | 17900025 / 17900025 / 17900025 | 30.25 / 30.36 / 29.86 | 1.000 | [0.209, 4.775] | 12/12 |
| M80 rho 6 | 4587764 / 4587764 / 4587764 | 2.49 / 2.60 / 2.42 | 1.000 | [0.077, 12.507] | 12/12 |
| M80 rho 14 | 37536050 / 37536050 / 37536050 | 141.95 / 139.91 / 139.76 | 1.000 | [0.330, 3.026] | 12/12 |
| LJ38 | 2040283 / 2040283 / 2040283 | 0.62 / 0.65 / 0.60 | 1.000 | [0.479, 2.086] | 12/12 |
| m38r10 | 3417294 / 3417294 / 3417294 | 3.10 / 3.10 / 3.09 | 1.000 | [0.428, 2.247] | 12/12 |

## Gate decision

Significant hit losses: none.

Significant hit-call losses outside the A/A' band: none.

Bonferroni wins over eight cells: none.

Marginal losses (0.05 <= p < 0.1): none.

Any hit drop, including non-significant: none.

Any call increase, including non-significant: none.

## Fresh-ledger bit identity

Per seed, A against B, for every cell that ran on a fresh ledger. A seed is identical when the final-energy bits, the charged call count, and the calls to hit all match.

| cell | seed | energy bits | charged calls | calls to hit | identical |
| --- | --- | --- | --- | --- | --- |
| LJ75 | 2000 | yes | yes | yes | yes |
| LJ75 | 2001 | yes | yes | yes | yes |
| LJ75 | 2002 | yes | yes | yes | yes |
| LJ75 | 2003 | yes | yes | yes | yes |
| LJ75 | 2004 | yes | yes | yes | yes |
| LJ75 | 2005 | yes | yes | yes | yes |
| LJ75 | 2006 | yes | yes | yes | yes |
| LJ75 | 2007 | yes | yes | yes | yes |
| LJ75 | 2008 | yes | yes | yes | yes |
| LJ75 | 2009 | yes | yes | yes | yes |
| LJ75 | 2010 | yes | yes | yes | yes |
| LJ75 | 2011 | yes | yes | yes | yes |
| LJ98 | 2000 | yes | yes | yes | yes |
| LJ98 | 2001 | yes | yes | yes | yes |
| LJ98 | 2002 | yes | yes | yes | yes |
| LJ98 | 2003 | yes | yes | yes | yes |
| LJ98 | 2004 | yes | yes | yes | yes |
| LJ98 | 2005 | yes | yes | yes | yes |
| LJ98 | 2006 | yes | yes | yes | yes |
| LJ98 | 2007 | yes | yes | yes | yes |
| LJ98 | 2008 | yes | yes | yes | yes |
| LJ98 | 2009 | yes | yes | yes | yes |
| LJ98 | 2010 | yes | yes | yes | yes |
| LJ98 | 2011 | yes | yes | yes | yes |
| M75 rho 6 | 2000 | yes | yes | yes | yes |
| M75 rho 6 | 2001 | yes | yes | yes | yes |
| M75 rho 6 | 2002 | yes | yes | yes | yes |
| M75 rho 6 | 2003 | yes | yes | yes | yes |
| M75 rho 6 | 2004 | yes | yes | yes | yes |
| M75 rho 6 | 2005 | yes | yes | yes | yes |
| M75 rho 6 | 2006 | yes | yes | yes | yes |
| M75 rho 6 | 2007 | yes | yes | yes | yes |
| M75 rho 6 | 2008 | yes | yes | yes | yes |
| M75 rho 6 | 2009 | yes | yes | yes | yes |
| M75 rho 6 | 2010 | yes | yes | yes | yes |
| M75 rho 6 | 2011 | yes | yes | yes | yes |
| M75 rho 14 | 2000 | yes | yes | yes | yes |
| M75 rho 14 | 2001 | yes | yes | yes | yes |
| M75 rho 14 | 2002 | yes | yes | yes | yes |
| M75 rho 14 | 2003 | yes | yes | yes | yes |
| M75 rho 14 | 2004 | yes | yes | yes | yes |
| M75 rho 14 | 2005 | yes | yes | yes | yes |
| M75 rho 14 | 2006 | yes | yes | yes | yes |
| M75 rho 14 | 2007 | yes | yes | yes | yes |
| M75 rho 14 | 2008 | yes | yes | yes | yes |
| M75 rho 14 | 2009 | yes | yes | yes | yes |
| M75 rho 14 | 2010 | yes | yes | yes | yes |
| M75 rho 14 | 2011 | yes | yes | yes | yes |
| M80 rho 6 | 2000 | yes | yes | yes | yes |
| M80 rho 6 | 2001 | yes | yes | yes | yes |
| M80 rho 6 | 2002 | yes | yes | yes | yes |
| M80 rho 6 | 2003 | yes | yes | yes | yes |
| M80 rho 6 | 2004 | yes | yes | yes | yes |
| M80 rho 6 | 2005 | yes | yes | yes | yes |
| M80 rho 6 | 2006 | yes | yes | yes | yes |
| M80 rho 6 | 2007 | yes | yes | yes | yes |
| M80 rho 6 | 2008 | yes | yes | yes | yes |
| M80 rho 6 | 2009 | yes | yes | yes | yes |
| M80 rho 6 | 2010 | yes | yes | yes | yes |
| M80 rho 6 | 2011 | yes | yes | yes | yes |
| M80 rho 14 | 2000 | yes | yes | yes | yes |
| M80 rho 14 | 2001 | yes | yes | yes | yes |
| M80 rho 14 | 2002 | yes | yes | yes | yes |
| M80 rho 14 | 2003 | yes | yes | yes | yes |
| M80 rho 14 | 2004 | yes | yes | yes | yes |
| M80 rho 14 | 2005 | yes | yes | yes | yes |
| M80 rho 14 | 2006 | yes | yes | yes | yes |
| M80 rho 14 | 2007 | yes | yes | yes | yes |
| M80 rho 14 | 2008 | yes | yes | yes | yes |
| M80 rho 14 | 2009 | yes | yes | yes | yes |
| M80 rho 14 | 2010 | yes | yes | yes | yes |
| M80 rho 14 | 2011 | yes | yes | yes | yes |
| LJ38 | 2000 | yes | yes | yes | yes |
| LJ38 | 2001 | yes | yes | yes | yes |
| LJ38 | 2002 | yes | yes | yes | yes |
| LJ38 | 2003 | yes | yes | yes | yes |
| LJ38 | 2004 | yes | yes | yes | yes |
| LJ38 | 2005 | yes | yes | yes | yes |
| LJ38 | 2006 | yes | yes | yes | yes |
| LJ38 | 2007 | yes | yes | yes | yes |
| LJ38 | 2008 | yes | yes | yes | yes |
| LJ38 | 2009 | yes | yes | yes | yes |
| LJ38 | 2010 | yes | yes | yes | yes |
| LJ38 | 2011 | yes | yes | yes | yes |
| m38r10 | 2000 | yes | yes | yes | yes |
| m38r10 | 2001 | yes | yes | yes | yes |
| m38r10 | 2002 | yes | yes | yes | yes |
| m38r10 | 2003 | yes | yes | yes | yes |
| m38r10 | 2004 | yes | yes | yes | yes |
| m38r10 | 2005 | yes | yes | yes | yes |
| m38r10 | 2006 | yes | yes | yes | yes |
| m38r10 | 2007 | yes | yes | yes | yes |
| m38r10 | 2008 | yes | yes | yes | yes |
| m38r10 | 2009 | yes | yes | yes | yes |
| m38r10 | 2010 | yes | yes | yes | yes |
| m38r10 | 2011 | yes | yes | yes | yes |

Fresh-ledger seeds that contradict the unchanged-trajectory claim: none.

## Checks

- PASS: latest session ended without abort (1 start, 1 end, 0 abort)
- PASS: complete schedule (288 of 288 rows)
- PASS: one row per sequence
- PASS: same run count in every cell and arm
- PASS: same seeds and starts in every arm
- PASS: balanced slot and arm counts (slot 0 A=32, slot 0 A'=32, slot 0 B=32, slot 1 A=32, slot 1 A'=32, slot 1 B=32, slot 2 A=32, slot 2 A'=32, slot 2 B=32)
- PASS: one pinned core and one thread
- PASS: arm source identities and binaries are stable
- PASS: B records the base revision and the patched tree
- PASS: A and A' are separate release builds of the base tree
- PASS: every cell ran on a fresh ledger (LJ75 LJ98 M75 rho 6 M75 rho 14 M80 rho 6 M80 rho 14 LJ38 m38r10)
