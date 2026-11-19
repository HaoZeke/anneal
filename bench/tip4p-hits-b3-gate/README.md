# tip4p-hits-b3 Regression and Performance Gate

## Verdict: `FAIL`

- Base ref: `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`
- Base tree: `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- B kit sha256: `a983aa97a5a3f152cbdc21c44e0cc4fa0d9613d73298d1ac336abec08f7a5312`
- B kit tree: `3eb47d63c8de0251b69abe821b3c2ca747434ed6`

- Overall outside-potential wall ratio (B / A): `0.9399` (`522.045 s` vs `555.421 s`, limit `<= 1.10x`)
- A/A calibration band (5 cells, including `tip4p15` and `water_hexamer`, second release build): max cell outside ratio `1.0021`, overall A/A outside ratio `0.9981`, min exact Mann-Whitney outside p `0.551167`
- Test suite parity (`tests.txt`): `PASS (zero B-only failures)`

## Test Suite Results

```text
suite=cargo_test_default arm=A cmd="cargo +1.88.0 test --locked" status=ok suites=188 passed=2250 failed=0 ignored=0
suite=cargo_test_default arm=B cmd="cargo +1.88.0 test --locked" status=ok suites=188 passed=2254 failed=0 ignored=0
suite=cargo_test_ci_features arm=A cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=build_failed detail=vesin_src_and_libira_absent
suite=cargo_test_ci_features arm=B cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=build_failed detail=vesin_src_and_libira_absent
suite=pytest arm=A cmd="pytest pytest/" status=ok passed=1609 failed=0 skipped=166
suite=pytest arm=B cmd="pytest pytest/" status=ok passed=1609 failed=0 skipped=166
b_only_failures=0
```

## A/B Panel (12 seeds, 2000-2011, serial core 0, ABBA per-seed order)

| Panel | System | Budget | Hits A/B | Fisher p | Med Calls A/B | MW p (calls) | Out Med A/B (s) | Ratio B/A | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.007895 / 0.007879 | 0.9980 | 0.977402 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 12/12 | 0.0372671 | 1481619.5 / 1314291.5 | 0.413966 | 2.740981 / 2.305666 | 0.8412 | 0.241524 | 7/12 | ok |
| tip4p | tip4p21 | 2000000 | 0/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 5.950604 / 4.905063 | 0.8243 | 7.17414e-05 | 3/12 | ok |
| tip4p | tip4p25 | 2000000 | 1/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 8.214558 / 6.444756 | 0.7846 | 0.000274392 | 2/12 | ok |
| molecular | water_hexamer | 4000 | 12/12 vs 12/12 | 1 | 3026.5 / 3026.5 | 1 | 2.282215 / 2.560810 | 1.1221 | 7.39602e-07 | 12/12 | FAIL |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 5897.0 / 5897.0 | 1 | 1.080475 / 1.082534 | 1.0019 | 0.932301 | 12/12 | ok |
| conformer | xld | 44988 | 8/12 vs 8/12 | 1 | 21981.0 / 21981.0 | 1 | 1.065958 / 1.066175 | 1.0002 | 0.932301 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.052264 / 0.037693 | 0.7212 | 0.67066 | 12/12 | ok |
| lj | lj75 | 4000000 | 12/12 vs 12/12 | 1 | 44176.0 / 44176.0 | 1 | 0.461840 / 0.459732 | 0.9954 | 0.755285 | 12/12 | ok |
| lj | lj98 | 4000000 | 10/12 vs 10/12 | 1 | 1321410.0 / 1321410.0 | 1 | 18.103527 / 18.001925 | 0.9944 | 0.842836 | 12/12 | ok |

## A/A Calibration Band (12 seeds, 2000-2011, serial core 0, A1 vs A2)

| Panel | System | Budget | Hits A1/A2 | Fisher p | Med Calls A1/A2 | MW p (calls) | Out Med A1/A2 (s) | Ratio A2/A1 | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.008330 / 0.008180 | 0.9820 | 0.977402 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 7/12 | 1 | 1481619.5 / 1481619.5 | 1 | 2.758919 / 2.741171 | 0.9936 | 0.887386 | 12/12 | ok |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 5897.0 / 5897.0 | 1 | 1.075576 / 1.073869 | 0.9984 | 0.551167 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.030246 / 0.030239 | 0.9998 | 1 | 12/12 | ok |
| molecular | water_hexamer | 4000 | 12/12 vs 12/12 | 1 | 3026.5 / 3026.5 | 1 | 2.254220 / 2.258860 | 1.0021 | 0.842836 | 12/12 | ok |

## Placement Band (water_hexamer)

Five arm A layout variants change only object size or section alignment of `anneal._core`. Each variant is run against arm A. Their median outside ratios are pooled with the second-build A/A on `water_hexamer`. Seeds 2000-2011, serial core 0, ABBA order, one thread.

| Series | Kind | Out med left/right (s) | Ratio | Exact MW p | Bit-ID |
| --- | --- | --- | ---: | ---: | --- |
| AA | second_build | 2.254220 / 2.258860 | 1.0021 | 0.842836 | 12/12 |
| PL1 | pad32 | 2.258996 / 2.262147 | 1.0014 | 0.977402 | 12/12 |
| PL2 | pad96 | 2.262600 / 2.282569 | 1.0088 | 0.755285 | 12/12 |
| PL3 | pad400 | 2.290458 / 2.280401 | 0.9956 | 0.755285 | 12/12 |
| PL4 | pad1200 | 2.277369 / 2.298571 | 1.0093 | 0.887386 | 12/12 |
| PL5 | pad700-pagesize65536 | 2.258779 / 2.267015 | 1.0036 | 0.551167 | 12/12 |

- Pooled placement band (min and max of the six median ratios): `0.9956` to `1.0093`
- `water_hexamer` B/A median outside ratio `1.1221` is outside the placement band
- Exact Mann-Whitney of arm B outside times (n=12) against the pooled variant outside times (n=60): U=`720.0`, p=`1.3018e-13`, Bonferroni m=`5`, p_adj=`6.50902e-13`
- `water_hexamer` loses on outside time only when its ratio is above the band and p_adj < 0.05
- Untouched cells (`rq3`, `xld`, `lj38`, `lj75`, `lj98`) lose on outside time only when their ratio is above that band and their exact Mann-Whitney p < 0.05
- `tip4p21` outside ratio `0.8243` is outside the placement band
- `tip4p25` outside ratio `0.7846` is outside the placement band
- Variant and A/A trajectories match arm A: `1`

## Hexamer hop-phase breakdown

A separate instrumented build times each water_hexamer search. The TRIC step is `TricOpt::minimize_watched` outside the potential. Book handling is the stall archive and, on arm B, the stalled-walk book. Acceptance is the decision that adopts or declines the quenched trial. Bookkeeping is the rest of the search outside the potential. Potential time is recorded beside those phases. These times are not the gate verdict.

| Arm | Seeds | Tric step (s) | Book (s) | Accept (s) | Bookkeeping (s) | Potential (s) | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 12 | 1.811116 | 0.000076 | 0.000683 | 0.030842 | 16.292639 | 3997 |
| B | 12 | 2.020307 | 0.000070 | 0.000704 | 0.031233 | 16.316445 | 3997 |

## Gate Failure Details

- AB/water_hexamer: outside_placement_band:ratio=1.1221,band=0.9956-1.0093,p=1.3018e-13,p_adj=6.50902e-13
