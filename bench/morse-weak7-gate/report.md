# Morse weak7 paired gate

Verdict: **PASS**

PASS only when B has no cell with fewer hits at two-sided Fisher p<0.05 and no cell with a higher hit-call median at two-sided Mann-Whitney p<0.05 above the A/A' bootstrap band.

- A and A' are separate release builds of the base source tree. B is the base tree plus the weak7 patch.
- All arms use the same 12 seeds and starts per cell. The A/B order follows ABBA across successive seeds, with A' interleaved.
- Runs are serial, one thread, pinned to one core, with no warm-up, retry, replacement, quick mode, or reduced budget.
- Calls-to-hit statistics include hitting runs only. Confidence intervals are 95% percentile bootstrap intervals with 20000 fixed-stream draws.
- The calls gate compares B/A's hit-call median ratio with the 95% bootstrap band of A'/A.
- Resume: no.

## Provenance

| field | value |
| --- | --- |
| base revision | `5eb79cdcb0a986984dc7b487e5a3cc93224246f9` |
| base tree | `7a2ba49b37241a8fe42b2c45c226374de6d92274` |
| harness revision | `3d7a6511b24c6aaef6296bcab5ddab284c8aa556` |
| harness tree | `c04324c520a138513ee6eeff311c6df451658d15` |
| harness sha256 | `6216a0e1baf22189186596e890b55bcc48e03b64c65e2383433e174ec60d71e2` |
| driver sha256 | `17be7aefb93e18bedc7b1b7e7f3c8558cc8adff64bf6880d8706dba2b6fcddbe` |
| host id (machine-id hash) | `698d22b051113ba2` |
| host label | `anneal-ci` |
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
| run environment | `LJOS_SEAT=benchbot MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1` |

Arm builds:

| arm | source revision | source tree | binary sha256 | porcelain before | porcelain after |
| --- | --- | --- | --- | --- | --- |
| A | 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 | 7a2ba49b37241a8fe42b2c45c226374de6d92274 | 8677b02a1a237cf946667a55915a81ccb605ee90c4d8bca9a5a5bc6a9cf705a7 | empty | empty |
| A' | 5eb79cdcb0a986984dc7b487e5a3cc93224246f9 | 7a2ba49b37241a8fe42b2c45c226374de6d92274 | 4d00fdc1839f0dafbc9f0bc6be8740ff8ec804406c5e34528b1221635e80ef0e | empty | empty |
| B | f7f6c4198119cb1c5559a44b6603c54b4071cc5d | ac274887eb9de3e1e177d1a6c7d8a6ec2b1d16d4 | af23ac34c7bd7fb9f1da2ae1c40aa9fe5c65a9fcb91294ace9517abefa5d86b1 | empty | empty |

## Per-cell results

| cell | A hits | A' hits | B hits | Fisher B/A | Fisher A'/A | A hit calls median [boot CI] | A' hit calls median [boot CI] | B hit calls median [boot CI] | MW B/A | MW A'/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | 5/12 | 5/12 | 11/12 | 0.027 | 1.000 | 2721351 [290074, 3327845] (n=5) | 2721351 [290074, 3327845] (n=5) | 2602476 [1244856, 3420800] (n=11) | 0.692 | 1.000 |
| LJ98 | 7/12 | 7/12 | 11/12 | 0.155 | 1.000 | 1430688 [363251, 2484398] (n=7) | 1430688 [363251, 2484398] (n=7) | 1406415 [973038, 2045601] (n=11) | 1.000 | 1.000 |
| M75 rho 6 | 11/12 | 11/12 | 12/12 | 1.000 | 1.000 | 822536 [192923, 1313081] (n=11) | 822536 [192923, 1313081] (n=11) | 1295603 [398206, 1527484] (n=12) | 0.538 | 1.000 |
| M75 rho 14 | 9/12 | 9/12 | 12/12 | 0.217 | 1.000 | 838982 [177756, 1112778] (n=9) | 838982 [177756, 1112778] (n=9) | 936525 [220056, 1166532] (n=12) | 0.455 | 1.000 |
| M80 rho 6 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 62677 [25846, 629251] (n=12) | 62677 [25846, 629251] (n=12) | 62677 [25846, 596242] (n=12) | 0.840 | 1.000 |
| M80 rho 14 | 6/12 | 6/12 | 8/12 | 0.680 | 1.000 | 2132943 [878074, 3757008] (n=6) | 2132943 [878074, 3757008] (n=6) | 3022796 [1295569, 3533826] (n=8) | 0.846 | 1.000 |
| LJ38 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 1.000 | 1.000 |
| m38r10 | 6/12 | 6/12 | 6/12 | 1.000 | 1.000 | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 1.000 | 1.000 |

Totals, wall medians, A/A' bands, and bit identity:

| cell | total charged calls A / A' / B | wall median s A / A' / B | hit-call median B/A | A/A' bootstrap band | A/B final-energy bits + calls identical |
| --- | --- | --- | --- | --- | --- |
| LJ75 | 38457980 / 38457980 / 29490443 | 59.12 / 60.31 / 48.23 | 0.956 | [0.107, 9.382] | 1/12 |
| LJ98 | 30921843 / 30921843 / 19707649 | 79.43 / 80.84 / 43.56 | 0.983 | [0.192, 5.200] | 2/12 |
| M75 rho 6 | 13520867 / 13520867 / 13986933 | 33.53 / 32.54 / 45.79 | 1.575 | [0.235, 4.264] | 3/12 |
| M75 rho 14 | 17900025 / 17900025 / 14016239 | 33.80 / 34.74 / 30.83 | 1.116 | [0.209, 4.775] | 4/12 |
| M80 rho 6 | 4587764 / 4587764 / 3230024 | 2.41 / 2.44 / 2.42 | 1.000 | [0.077, 12.507] | 8/12 |
| M80 rho 14 | 37536050 / 37536050 / 36261319 | 145.36 / 145.46 / 128.44 | 1.417 | [0.330, 3.026] | 0/12 |
| LJ38 | 2040283 / 2040283 / 2040283 | 0.67 / 0.66 / 0.65 | 1.000 | [0.479, 2.086] | 12/12 |
| m38r10 | 3417294 / 3417294 / 3417294 | 3.17 / 3.14 / 3.18 | 1.000 | [0.428, 2.247] | 12/12 |

Cells bit-identical in final energy and charged calls for every A/B seed: LJ38, m38r10.

## Gate decision

Significant hit losses: none.

Significant hit-call losses outside the A/A' band: none.

Bonferroni wins over eight cells: none.

Marginal losses (0.05 <= p < 0.1): none.

## Order counts

Counts of (slot, arm) over the session. The A/B order is ABBA, so A-before-B equals B-before-A and each arm has the same number of runs. A' is interleaved on that pattern.

| slot | A | A' | B |
| --- | --- | --- | --- |
| 0 | 48 | 24 | 24 |
| 1 | 0 | 48 | 48 |
| 2 | 48 | 24 | 24 |

A before B: 48. B before A: 48.

## Cross-session context, not a gate

Weak7 arm B compared with weak6 arm B. The weak6 runs were recorded on base `dd054749d4703923f4cf4c34f50589c574ee1276`, a different base from this gate. This is a cross-session comparison and not a gate. It does not enter the verdict.

The weak7 patch keeps the incumbent and its encounters across the fork and anchors a restart by replica id, so trajectories can change.

Weak6 rows were read from the weak6 gate mbox at report time. Mbox sha256: `5343eddcb67b81f13224945ec306571952e29b9510d744645ef043885cfa95dd`. Those rows are not part of this tree.

| cell | weak7 B hits | weak6 B hits | Fisher p | weak7 B hit-call median | weak6 B hit-call median | MW p | total calls weak7 B / weak6 B | wall median s weak7 B / weak6 B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | 11/12 | 12/12 | 1.000 | 2602476 (n=11) | 1793308 (n=12) | 0.218 | 29490443 / 20931190 | 48.23 / 29.16 |
| LJ98 | 11/12 | 12/12 | 1.000 | 1406415 (n=11) | 1298332 (n=12) | 0.975 | 19707649 / 18436241 | 43.56 / 32.47 |
| M75 rho 6 | 12/12 | 11/12 | 1.000 | 1295603 (n=12) | 723381 (n=11) | 0.498 | 13986933 / 13865067 | 45.79 / 24.84 |
| M75 rho 14 | 12/12 | 12/12 | 1.000 | 936525 (n=12) | 779278 (n=12) | 0.795 | 14016239 / 11316319 | 30.83 / 23.63 |
| M80 rho 6 | 12/12 | 12/12 | 1.000 | 62677 (n=12) | 62677 (n=12) | 0.707 | 3230024 / 2872667 | 2.42 / 2.26 |
| M80 rho 14 | 8/12 | 7/12 | 1.000 | 3022796 (n=8) | 2305686 (n=7) | 0.183 | 36261319 / 33225146 | 128.44 / 86.36 |
| LJ38 | 12/12 | 12/12 | 1.000 | 127840 (n=12) | 127840 (n=12) | 1.000 | 2040283 / 2040283 | 0.65 / 0.60 |
| m38r10 | 6/12 | 6/12 | 1.000 | 170600 (n=6) | 170600 (n=6) | 1.000 | 3417294 / 3417294 | 3.18 / 3.16 |

## Checks

- PASS: every session ended without abort (1 start, 1 end, 0 abort)
- PASS: complete schedule (288 of 288 rows)
- PASS: one row per sequence
- PASS: same run count in every cell and arm
- PASS: same seeds and starts in every arm
- PASS: ABBA order with A' interleaved
- PASS: balanced order-flipped (slot, arm) counts (slot 0 A=48 A'=24 B=24; slot 1 A=0 A'=48 B=48; slot 2 A=48 A'=24 B=24; A-before-B=48; B-before-A=48)
- PASS: A and A' share a source and B is a different tree
- PASS: arm modes match the build
- PASS: host label is anneal-ci
- PASS: one pinned core and one thread
- PASS: arm source identities and binaries are stable
- PASS: all arm worktree status records are empty
