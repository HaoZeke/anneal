# Morse weak6 paired gate

Verdict: **PASS**

PASS only when B has no cell with fewer hits at two-sided Fisher p<0.05 and no cell with a higher hit-call median at two-sided Mann-Whitney p<0.05 above the A/A' bootstrap band.

- A and A' are separate release builds of the base source tree. B is the two-patch weak6 tree.
- All arms use the same 12 seeds and starts per cell. The A/B order follows ABBA across successive seeds, with A' interleaved.
- Runs are serial, one thread, pinned to one core, with no warm-up, retry, replacement, quick mode, or reduced budget.
- Calls-to-hit statistics include hitting runs only. Confidence intervals are 95% percentile bootstrap intervals with 20000 fixed-stream draws.
- The calls gate compares B/A's hit-call median ratio with the 95% bootstrap band of A'/A.
- Resume: no.

## Provenance

| field | value |
| --- | --- |
| base revision | `dd054749d4703923f4cf4c34f50589c574ee1276` |
| base tree | `f1705d6f02f7f257f50e831c16057740283efea5` |
| harness revision | `660ed71612bd21d77a9dff56963697624d885b68` |
| harness tree | `35ebc30b5f8fb9809f4ea621533b8624b31efa86` |
| harness sha256 | `9b0c8abb2379a2e4d99259b8bb916c59bce517fbd729935067f9c65e46e4e3e6` |
| driver sha256 | `858657408ddf040e12fc0b99719352b97f4ce247de0b5f60253b76771481b66e` |
| host id (machine-id hash) | `57b494e29cd1fd6b` |
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
| A | dd054749d4703923f4cf4c34f50589c574ee1276 | f1705d6f02f7f257f50e831c16057740283efea5 | cc5bb9dda93a300902c01d8983982a4f0579f5a85765d963c5993f5744628ff1 | empty | empty |
| A' | dd054749d4703923f4cf4c34f50589c574ee1276 | f1705d6f02f7f257f50e831c16057740283efea5 | 82cdaa1916b434befda0fddda341980728af8ce05259fbfd88211373945b6028 | empty | empty |
| B | b5d24dc1e34bb5c2666fc4f966ad2bbf456133a3 | 90b19586099d060640a6487391b23bccf0ad1172 | 1a4a1cea231d8115cc954d556d3a22d75558ed9fa802e4a55302457eac94dacd | empty | empty |

## Per-cell results

| cell | A hits | A' hits | B hits | Fisher B/A | Fisher A'/A | A hit calls median [boot CI] | A' hit calls median [boot CI] | B hit calls median [boot CI] | MW B/A | MW A'/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | 5/12 | 5/12 | 12/12 | 0.005 | 1.000 | 2721351 [290074, 3327845] (n=5) | 2721351 [290074, 3327845] (n=5) | 1793308 [895065, 2280322] (n=12) | 0.673 | 1.000 |
| LJ98 | 7/12 | 7/12 | 12/12 | 0.037 | 1.000 | 1430688 [363251, 2484398] (n=7) | 1430688 [363251, 2484398] (n=7) | 1298332 [755076, 2462766] (n=12) | 0.966 | 1.000 |
| M75 rho 6 | 11/12 | 11/12 | 11/12 | 1.000 | 1.000 | 822536 [192923, 1313081] (n=11) | 822536 [192923, 1313081] (n=11) | 723381 [192923, 1483582] (n=11) | 0.767 | 1.000 |
| M75 rho 14 | 9/12 | 9/12 | 12/12 | 0.217 | 1.000 | 838982 [177756, 1112778] (n=9) | 838982 [177756, 1112778] (n=9) | 779278 [220056, 1191802] (n=12) | 0.696 | 1.000 |
| M80 rho 6 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 62677 [25846, 629251] (n=12) | 62677 [25846, 629251] (n=12) | 62677 [25846, 537013] (n=12) | 0.707 | 1.000 |
| M80 rho 14 | 6/12 | 6/12 | 7/12 | 1.000 | 1.000 | 2132943 [878074, 3757008] (n=6) | 2132943 [878074, 3757008] (n=6) | 2305686 [822133, 2413861] (n=7) | 0.617 | 1.000 |
| LJ38 | 12/12 | 12/12 | 12/12 | 1.000 | 1.000 | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 127840 [105327, 262772] (n=12) | 1.000 | 1.000 |
| m38r10 | 6/12 | 6/12 | 6/12 | 1.000 | 1.000 | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 170600 [80314, 257704] (n=6) | 1.000 | 1.000 |

Totals, wall medians, A/A' bands, and bit identity:

| cell | total charged calls A / A' / B | wall median s A / A' / B | hit-call median B/A | A/A' bootstrap band | A/B final-energy bits + calls identical |
| --- | --- | --- | --- | --- | --- |
| LJ75 | 38457980 / 38457980 / 20931190 | 60.59 / 61.63 / 29.16 | 0.659 | [0.107, 9.382] | 1/12 |
| LJ98 | 30921843 / 30921843 / 18436241 | 77.14 / 75.75 / 32.47 | 0.907 | [0.192, 5.200] | 2/12 |
| M75 rho 6 | 13520867 / 13520867 / 13865067 | 28.80 / 28.66 / 24.84 | 0.879 | [0.235, 4.264] | 3/12 |
| M75 rho 14 | 17900025 / 17900025 / 11316319 | 29.60 / 29.54 / 23.63 | 0.929 | [0.209, 4.775] | 4/12 |
| M80 rho 6 | 4587764 / 4587764 / 2872667 | 2.26 / 2.26 / 2.26 | 1.000 | [0.077, 12.507] | 8/12 |
| M80 rho 14 | 37536050 / 37536050 / 33225146 | 136.84 / 136.76 / 86.36 | 1.081 | [0.330, 3.026] | 1/12 |
| LJ38 | 2040283 / 2040283 / 2040283 | 0.60 / 0.60 / 0.60 | 1.000 | [0.479, 2.086] | 12/12 |
| m38r10 | 3417294 / 3417294 / 3417294 | 3.15 / 3.16 / 3.16 | 1.000 | [0.428, 2.247] | 12/12 |

Cells bit-identical in final energy and charged calls for every A/B seed: LJ38, m38r10.

## Gate decision

Significant hit losses: none.

Significant hit-call losses outside the A/A' band: none.

Bonferroni wins over eight cells: [{"bonferroni_p": 0.036613272311212815, "cell": "lj75", "metric": "hits", "p": 0.004576659038901602}].

Marginal losses (0.05 <= p < 0.1): none.

## Checks

- PASS: every session ended without abort (1 start, 1 end, 0 abort)
- PASS: complete schedule (288 of 288 rows)
- PASS: one row per sequence
- PASS: same run count in every cell and arm
- PASS: same seeds and starts in every arm
- PASS: ABBA order with A' interleaved
- PASS: one pinned core and one thread
- PASS: arm source identities and binaries are stable
- PASS: all arm worktree status records are empty
