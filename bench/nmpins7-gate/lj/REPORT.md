# nmpins7 Lennard-Jones gate

Cells LJ38, LJ75, LJ98 and LJ104. Seeds 2000-2011. Arms A, B and C ran in one session on core 0.

B loses nowhere.

Calls are evaluations to the minimum, and a miss is entered at the budget. Outside time per call is `(wall - potential time) / charged calls`. The outside-time loss is a median ratio above 1.1. Hit and call losses use a one-sided test and Bonferroni m = 30 at alpha 0.05. C is the same comparison with no pass or fail.

| cell | arm | hits | median calls | IQR | median outside/call (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| lj38 | A | 12/12 | 127839.50 | 139245.25 | 2.843008e-06 |
| lj38 | B | 12/12 | 127839.50 | 139245.25 | 2.761933e-06 |
| lj38 | C | 12/12 | 127839.50 | 139245.25 | 2.794523e-06 |
| lj75 | A | 5/12 | 4000000.00 | 962776.75 | 7.032217e-06 |
| lj75 | B | 5/12 | 4000000.00 | 962776.75 | 6.917017e-06 |
| lj75 | C | 5/12 | 4000000.00 | 962776.75 | 6.994637e-06 |
| lj98 | A | 7/12 | 3166811.50 | 2726730.25 | 9.759106e-06 |
| lj98 | B | 7/12 | 3166811.50 | 2726730.25 | 9.712284e-06 |
| lj98 | C | 7/12 | 3166813.50 | 2726730.25 | 9.655638e-06 |
| lj104 | A | 5/12 | 4000000.00 | 2271734.00 | 1.071968e-05 |
| lj104 | B | 5/12 | 4000000.00 | 2271734.00 | 1.081235e-05 |
| lj104 | C | 5/12 | 4000000.00 | 2244528.25 | 1.078969e-05 |

| cell | contrast | Fisher raw | Fisher Bonferroni | calls U | calls p raw | calls p Bonferroni | outside ratio | outside U | outside p raw | outside p Bonferroni | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lj38 | B vs A | 1 | 1 | 72.0 | 0.511545 | 1 | 0.9715 | 59.0 | 0.782135 | 1 | pass |
| lj38 | C vs A | 1 | 1 | 72.0 | 0.511545 | 1 | 0.9829 | 62.0 | 0.727815 | 1 | not gated |
| lj75 | B vs A | 0.659914 | 1 | 72.0 | 0.512874 | 1 | 0.9836 | 80.0 | 0.332503 | 1 | pass |
| lj75 | C vs A | 0.659914 | 1 | 72.5 | 0.5 | 1 | 0.9947 | 76.0 | 0.41993 | 1 | not gated |
| lj98 | B vs A | 0.659914 | 1 | 72.0 | 0.511971 | 1 | 0.9952 | 68.0 | 0.602494 | 1 | pass |
| lj98 | C vs A | 0.659914 | 1 | 72.5 | 0.5 | 1 | 0.9894 | 66.0 | 0.646273 | 1 | not gated |
| lj104 | B vs A | 0.659914 | 1 | 72.0 | 0.512874 | 1 | 1.0086 | 79.0 | 0.353727 | 1 | pass |
| lj104 | C vs A | 0.659914 | 1 | 73.0 | 0.487133 | 1 | 1.0065 | 77.0 | 0.397506 | 1 | not gated |

A/A on LJ38, same binary, n = 12, ABBA. Replicas agree on hit, charged calls, calls to the hit and best energy on 12/12 seeds. Hits 12/12 and 12/12. Median outside time per call 2.807311e-06 and 2.795191e-06 (ratio of medians 0.9957). Per-seed replica-1/replica-0 ratios have median 1.0052, IQR 0.0534, range 0.9333 to 1.0766. Two-sided Mann-Whitney p on outside time per call is 0.83986.

| cell | B/A outside ratio | inside A/A per-seed range |
| --- | ---: | --- |
| lj38 | 0.9715 | yes |
| lj75 | 0.9836 | yes |
| lj98 | 0.9952 | yes |
| lj104 | 1.0086 | yes |

Slot counts for the gate, each (slot, arm) pair:

| cell | slot | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| lj38 | 0 | 4 | 4 | 4 |
| lj38 | 1 | 4 | 4 | 4 |
| lj38 | 2 | 4 | 4 | 4 |
| lj75 | 0 | 4 | 4 | 4 |
| lj75 | 1 | 4 | 4 | 4 |
| lj75 | 2 | 4 | 4 | 4 |
| lj98 | 0 | 4 | 4 | 4 |
| lj98 | 1 | 4 | 4 | 4 |
| lj98 | 2 | 4 | 4 | 4 |
| lj104 | 0 | 4 | 4 | 4 |
| lj104 | 1 | 4 | 4 | 4 |
| lj104 | 2 | 4 | 4 | 4 |

## Methods

Release binaries, `cargo build --release --locked`. The whole process tree was pinned with `taskset -c 0`. OpenMP, OpenBLAS, MKL, Rayon, NumExpr and the other thread-count variables were 1, and the dynamic thread switches were off. No MPI launcher was used. The wall cap was 7200 s per seed. No run was discarded and none was a warm-up. The hit tolerance is 0.001. Medians and IQRs are Hyndman-Fan type 7. Fisher's exact test is the one-sided hypergeometric tail for fewer hits. The Mann-Whitney test is the normal approximation with the tie correction and the continuity correction, upper tail, alternative that the named arm is higher. Bonferroni uses m = 30. A call or hit loss requires the adjusted p to be below 0.05. An outside-time loss requires the ratio of median outside time per call to exceed 1.1. Arm C sets `H2H_GENERAL_SYMMETRY=1`, which sets `Config::general_symmetry`. Arm B leaves that variable unset, so the field stays at its default. Arm A does not have the field.

## Provenance

- Base commit `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`, tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`.
- Kit commit `4cd811004c67f26b0e216664cb51352f04654c09`, tree `51c1a7e0180585d078c3457b5550b2963d1ab752`.
- Arm A build `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`, tree `7a2ba49b37241a8fe42b2c45c226374de6d92274`, binary sha256 `3506346703a8825a47cf8d76ce696fc11ae13bf8ca76302ab592da7f1053545b`.
- Arms B and C build `1492b78a188066888f0227fab9fc615c3c056a30`, tree `c925b681f2c22795e2aeac8ab0121c92746f4909`, binary sha256 `f8a0ba25e90a34ad3d7352788e1623571b8c04f684606560fabe4e83e4043673`. Parent is the kit commit. The only change is the environment read of the flag.
- Harness commit `3ef60c456b6025feef1698ed76a830c8630f6a92`.
- Measurement commit `cd9450ccdf2c78997159143d84ca998d46f28e4e`.
- Session `20261119T134957Z` at `2026-11-19T13:49:57Z`.
- Machine id sha256 `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`. Core 0, affinity 0.
- Compiler `rustc 1.88.0 (6b00bc388 2025-06-23)`. Cargo `cargo 1.88.0 (873a06493 2025-05-10)`.
- Lock sha256 `bf488f7aea4edcec4dbf10cf72a5368dbbab9d492869b03c7adab57de0608110`.
- `anneal-core` 0.12.0.
- `rgmin` 0.5.0.
- `rgpot-core` 3.5.1.
- `rgpot-pair` 3.7.0@8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e.
- `readcon-core` 0.14.8.
- Budgets: lj38 400000, lj75 4000000, lj98 4000000, lj104 4000000.
- References: lj38 -173.928427, lj75 -397.492331, lj98 -543.665361, lj104 -582.086642.
