# sc-shelf-land4 Gate A Report

**Verdict: FAIL**

## Failure Reasons

- lj75: hit loss B=4 < A=8
- m30r14: hit loss B=9 < A=10
- sc96-55: B/A outside/charged 1.033133 > band upper 1.033093
- sc96-80: hit loss B=5 < A=7
- sc96-80: B/A outside/charged 1.085862 > band upper 1.033093

## Provenance & Protocol

- Arm A: `61fa67cf380e46876b569da45e7cc091647a775b` (tree `cff796c507a8fa1d0bfb28888c3570b61988055e`)
- Arm B: `3381b56419aeeece37ff4b13d370967b0b2ab6d0` (tree after `git am -3 --keep-cr` of 9-commit series on A)
- Hashed machine ID: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- Toolchain: `rustc 1.88.0 (6b00bc388 2025-06-23)` (`cargo build --release --locked`)
- rgpot revision: `672aef430855036125c902ace024756d682e180a`
- librgpot SHA-256: `bc447c976cd868ce8adda9bb20619f58c88635d0566044ab1fa94b5ccb4e49a8`
- Core pinning: `taskset -c 0` across entire process tree; all thread env vars = 1
- Seeds: `2100..2111` (`n=12`), balanced ABBA order (`6` `(slot=0, A)`, `6` `(slot=0, B)`, `6` `(slot=1, A)`, `6` `(slot=1, B)` per cell)
- Hit rule: finite `best <= reference + 0.001` (re-derived from `best` column)
- Outside time per call: `outside_s / charged` computed per row, median per arm, ratio `B/A` of medians

### Binary SHA-256 Registry

| phase | arm | driver | code_revision | tree | binary_sha256 |
|---|---|---|---|---|---|
| ab | A | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `722c98a410b8b33ebd83754509860cf4487a687fd497b65935ae3100d80ea567` |
| ab | A | `h2h_seed` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `12900bc2c547d29514c84b862d694a9f3427760becc2e251a29657281f04ffd9` |
| ab | A | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `f468e60b171bacf6e346ef1c060ffb9f89d462fe4099f3d476551866447a5cef` |
| ab | A | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `36811abea808d1f25c17e53bf3e55f2fbea05e8112d22f8d4deea7ff2c2d21ec` |
| ab | B | `h2h_gmin_proof` | `fd258d01902217f912f8ef89b3997064de018ad3` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `ce9475f9dc0b5dde8c55620ca37e2d478db3a3079001d1bbf28160c8b60901cb` |
| ab | B | `h2h_seed` | `fd258d01902217f912f8ef89b3997064de018ad3` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `d78654c92273e7f56035175b3f021266c7aa1db379bdd7a865f8fd51f9f4b3f4` |
| ab | B | `h2h_morse` | `fd258d01902217f912f8ef89b3997064de018ad3` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `850f3ef42bd59d13b648c23acbad28b074e0ef39d9bc074c098faca8ab14d2b6` |
| ab | B | `h2h_sc` | `fd258d01902217f912f8ef89b3997064de018ad3` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `c8f0fea35254e17f9582e831847ab93032874582fd6bf4ee94e1ce7cafa33f76` |
| aa | A | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `722c98a410b8b33ebd83754509860cf4487a687fd497b65935ae3100d80ea567` |
| aa | A | `h2h_seed` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `12900bc2c547d29514c84b862d694a9f3427760becc2e251a29657281f04ffd9` |
| aa | A | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `f468e60b171bacf6e346ef1c060ffb9f89d462fe4099f3d476551866447a5cef` |
| aa | A | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `36811abea808d1f25c17e53bf3e55f2fbea05e8112d22f8d4deea7ff2c2d21ec` |
| aa | A_prime | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `0a705acd8c130e30b38b333a3a62db9028007493a827726fc6bc75212aa2f865` |
| aa | A_prime | `h2h_seed` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `a9227d2ecbea35014663fddd59fd62851b5cd4b610e3b2ad1ab0d5b6d1e417de` |
| aa | A_prime | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `f6a189662612036bb0530c777adbb0ad3294879f305eb5e772b922687d3a167c` |
| aa | A_prime | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `a477bc2f22102ca6bce417965f19f3aec2e0c8fb4d1de428972a51bfece41fbc` |
| placement | default | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `722c98a410b8b33ebd83754509860cf4487a687fd497b65935ae3100d80ea567` |
| placement | default | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `f468e60b171bacf6e346ef1c060ffb9f89d462fe4099f3d476551866447a5cef` |
| placement | default | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `36811abea808d1f25c17e53bf3e55f2fbea05e8112d22f8d4deea7ff2c2d21ec` |
| placement | sort_section | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `5e0d057987c838539f8e6ca0fc83511001fd86422803ac5b194f9f6eaa44a976` |
| placement | sort_section | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `81258281809ef27b06d6d50a7b2ca0954e575dd6972adf79c1bab8c3387d1a11` |
| placement | sort_section | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `4b6876a3d0051ee1d1fa4c142b46073f201d121ec98d596ce9c682a139042ec4` |
| placement | unused_obj | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `3d44a35638b4da9d5f99b2874c6c90ee39eb516fdc5b4e5448c59d833537b43d` |
| placement | unused_obj | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `c41fab65dd54d94a4a6cf333ae767bb3496b6c16eafaaeacb1b512dc77734245` |
| placement | unused_obj | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `1f0eb054b9aebf5c488ecb38cc6ac27b448cf5b9f07c5692ab09d10f071907c3` |

## Declared Rule Evaluation (14 Cells)

| system | driver | hits A/B | Fisher p | Fisher Bonf(14) | med calls A/B | MW exact p | MW Bonf(14) | wall med s A/B | pot med s A/B | out/chg med us A/B | B/A out/chg | band upper | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj38 | `h2h_gmin_proof` | 10/11 | 1 | 1 | 134705.5/50048.0 | 0.400075 | 1 | 0.6428/0.2750 | 0.3096/0.1198 | 2.811/2.755 | 0.980261 | 1.033093 | PASS |
| lj75 | `h2h_gmin_proof` | 8/4 | 0.220347 | 1 | 2901899.5/4000000.0 | 0.217491 | 1 | 44.0339/58.4492 | 25.2502/34.1507 | 6.432/6.259 | 0.973142 | 1.033093 | FAIL |
| lj77 | `h2h_seed` | 10/10 | 1 | 1 | 1058330.0/1058330.0 | 0.988112 | 1 | 20.7993/20.6926 | 9.5334/9.6055 | 10.923/11.058 | 1.012346 | 1.033093 | PASS |
| lj98 | `h2h_gmin_proof` | 10/11 | 1 | 1 | 1906163.0/1432728.5 | 0.486423 | 1 | 49.1022/33.6610 | 28.7162/20.7992 | 9.998/9.967 | 0.996946 | 1.033093 | PASS |
| m30r14 | `h2h_morse` | 10/9 | 1 | 1 | 155647.0/190855.5 | 0.658594 | 1 | 0.8710/1.0744 | 0.6035/0.7441 | 1.762/1.786 | 1.014068 | 1.033093 | FAIL |
| sc96-38 | `h2h_sc` | 12/12 | 1 | 1 | 45150.0/54593.5 | 0.599319 | 1 | 0.4901/0.5749 | 0.4007/0.4684 | 2.274/2.081 | 0.915148 | 1.033093 | PASS |
| sc96-55 | `h2h_sc` | 12/12 | 1 | 1 | 17008.0/17008.0 | 0.981548 | 1 | 0.3754/0.3878 | 0.3021/0.3097 | 4.044/4.178 | 1.033133 | 1.033093 | FAIL |
| sc96-80 | `h2h_sc` | 7/5 | 0.684273 | 1 | 924997.5/1000000.0 | 0.482525 | 1 | 43.5800/47.4341 | 34.4292/36.2592 | 10.069/10.934 | 1.085862 | 1.033093 | FAIL |
| sc108-38 | `h2h_sc` | 6/6 | 1 | 1 | 345983.5/345983.5 | 1 | 1 | 3.6933/3.6925 | 2.8194/2.8028 | 2.676/2.654 | 0.991883 | 1.033093 | PASS |
| sc108-55 | `h2h_sc` | 3/3 | 1 | 1 | 1000000.0/1000000.0 | 1 | 1 | 22.6285/22.3835 | 16.8832/16.6846 | 5.649/5.621 | 0.995125 | 1.033093 | PASS |
| sc108-80 | `h2h_sc` | 2/2 | 1 | 1 | 1000000.0/1000000.0 | 1 | 1 | 46.8696/45.3547 | 35.4527/34.7221 | 11.127/11.072 | 0.995010 | 1.033093 | PASS |
| sc126-38 | `h2h_sc` | 12/12 | 1 | 1 | 23557.5/24297.0 | 0.920758 | 1 | 0.2528/0.2443 | 0.2007/0.1927 | 2.375/2.365 | 0.995920 | 1.033093 | PASS |
| sc126-55 | `h2h_sc` | 12/12 | 1 | 1 | 14998.5/14998.5 | 0.943938 | 1 | 0.3513/0.3512 | 0.2854/0.2872 | 4.489/4.625 | 1.030316 | 1.033093 | PASS |
| sc126-80 | `h2h_sc` | 5/5 | 1 | 1 | 1000000.0/1000000.0 | 0.846749 | 1 | 44.8723/43.3761 | 33.7932/32.7157 | 11.407/11.651 | 1.021331 | 1.033093 | PASS |

### Per-Seed Hit Strings (seeds 2100..2111)

| system | A hits | B hits |
|---|---|---|
| lj38 | `111110111110` (10/12) | `111111111101` (11/12) |
| lj75 | `001110111110` (8/12) | `110010010000` (4/12) |
| lj77 | `111111001111` (10/12) | `111111001111` (10/12) |
| lj98 | `111010111111` (10/12) | `111111101111` (11/12) |
| m30r14 | `110111101111` (10/12) | `100110111111` (9/12) |
| sc96-38 | `111111111111` (12/12) | `111111111111` (12/12) |
| sc96-55 | `111111111111` (12/12) | `111111111111` (12/12) |
| sc96-80 | `000011110111` (7/12) | `000011100110` (5/12) |
| sc108-38 | `010001110101` (6/12) | `010001110101` (6/12) |
| sc108-55 | `010010010000` (3/12) | `010010010000` (3/12) |
| sc108-80 | `000000010100` (2/12) | `000000010100` (2/12) |
| sc126-38 | `111111111111` (12/12) | `111111111111` (12/12) |
| sc126-55 | `111111111111` (12/12) | `111111111111` (12/12) |
| sc126-80 | `110011000010` (5/12) | `110011000001` (5/12) |

## Noise Bands

- `sc108-55` A/A outside-time-per-charged-call ratio (`A'/A`): `1.002269` (symmetric `1.002269`)
- Pooled A/A upper edge across all 14 cells: `1.033093` (directional max `A'/A`: `1.021291`)
- Pooled A/A + placement band upper edge: `1.033093`

### A/A Band (All 14 Cells, A vs A')

| system | med out/chg A (us) | med out/chg A' (us) | ratio A'/A | symmetric max(r, 1/r) |
|---|---:|---:|---:|---:|
| lj38 | 2.531 | 2.536 | 1.001873 | 1.001873 |
| lj75 | 6.405 | 6.199 | 0.967967 | 1.033093 |
| lj77 | 10.622 | 10.585 | 0.996424 | 1.003589 |
| lj98 | 9.369 | 9.275 | 0.989970 | 1.010132 |
| m30r14 | 1.738 | 1.739 | 1.000712 | 1.000712 |
| sc96-38 | 2.163 | 2.163 | 0.999648 | 1.000352 |
| sc96-55 | 3.962 | 3.972 | 1.002351 | 1.002351 |
| sc96-80 | 9.623 | 9.828 | 1.021291 | 1.021291 |
| sc108-38 | 2.560 | 2.576 | 1.005923 | 1.005923 |
| sc108-55 | 5.480 | 5.493 | 1.002269 | 1.002269 |
| sc108-80 | 10.661 | 10.702 | 1.003868 | 1.003868 |
| sc126-38 | 2.528 | 2.464 | 0.974603 | 1.026059 |
| sc126-55 | 4.479 | 4.352 | 0.971567 | 1.029266 |
| sc126-80 | 11.156 | 11.168 | 1.001071 | 1.001071 |

### Placement Band (Arm A Rebuilt in 3 Link Variants)

| system | default (us) | sort_section (us) | unused_obj (us) | sort/default | unused/default | spread max/min |
|---|---:|---:|---:|---:|---:|---:|
| sc108-55 | 5.496 | 5.499 | 5.464 | 1.000526 | 0.994275 | 1.006287 |
| sc96-38 | 2.180 | 2.163 | 2.163 | 0.991840 | 0.991917 | 1.008227 |
| lj38 | 2.504 | 2.527 | 2.520 | 1.009129 | 1.006524 | 1.009129 |
| m30r14 | 1.739 | 1.735 | 1.740 | 0.997584 | 1.000206 | 1.002629 |
