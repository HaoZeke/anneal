# tip4p-hits-b2 Regression and Performance Gate

## Verdict: `FAIL`

- Base ref: `5eb79cdcb0a986984dc7b487e5a3cc93224246f9`
- Base tree: `7a2ba49b37241a8fe42b2c45c226374de6d92274`
- B kit sha256: `0c5607a5a1be12612d660271cc22d4957bec89742969e872e54ef58399961ad0`
- B kit tree: `bf5617649af17a19b5a3cd704c676014055a689b`

- Overall outside-potential wall ratio (B / A): `0.9499` (`556.666 s` vs `586.047 s`, limit `<= 1.10x`)
- A/A calibration band (5 cells, including `tip4p15` and `water_hexamer`, second release build): max cell outside ratio `1.0071`, overall A/A outside ratio `0.9905`, min exact Mann-Whitney outside p `0.218921`
- Test suite parity (`tests.txt`): `PASS (zero B-only failures)`

## Test Suite Results

```text
suite=cargo_test_default arm=A cmd="cargo +1.88.0 test --locked" status=ok suites=188 passed=2250 failed=0 ignored=0
suite=cargo_test_default arm=B cmd="cargo +1.88.0 test --locked" status=ok suites=188 passed=2253 failed=0 ignored=0
suite=cargo_test_ci_features arm=A cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=build_failed detail=vesin_src_and_libira_absent
suite=cargo_test_ci_features arm=B cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=build_failed detail=vesin_src_and_libira_absent
suite=pytest arm=A cmd="pytest pytest/" status=ok passed=1609 failed=0 skipped=166
suite=pytest arm=B cmd="pytest pytest/" status=ok passed=1609 failed=0 skipped=166
b_only_failures=0
```

## A/B Panel (12 seeds, 2000-2011, serial core 0, ABBA per-seed order)

| Panel | System | Budget | Hits A/B | Fisher p | Med Calls A/B | MW p (calls) | Out Med A/B (s) | Ratio B/A | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.009194 / 0.009290 | 1.0105 | 0.977402 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 12/12 | 0.0372671 | 1481619.5 / 1314291.5 | 0.413966 | 3.117540 / 2.667423 | 0.8556 | 0.347358 | 7/12 | ok |
| tip4p | tip4p21 | 2000000 | 0/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 6.284633 / 5.156204 | 0.8204 | 0.00037128 | 3/12 | ok |
| tip4p | tip4p25 | 2000000 | 1/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 8.551343 / 6.834104 | 0.7992 | 0.00291403 | 2/12 | ok |
| molecular | water_hexamer | 4000 | 12/12 vs 12/12 | 1 | 3026.5 / 3026.5 | 1 | 2.412770 / 2.697471 | 1.1180 | 0.00363662 | 12/12 | FAIL |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 5897.0 / 5897.0 | 1 | 1.139994 / 1.130335 | 0.9915 | 0.755285 | 12/12 | ok |
| conformer | xld | 44988 | 8/12 vs 8/12 | 1 | 21981.0 / 21981.0 | 1 | 1.100544 / 1.104893 | 1.0040 | 0.842836 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.031469 / 0.031827 | 1.0114 | 0.798745 | 12/12 | ok |
| lj | lj75 | 4000000 | 12/12 vs 12/12 | 1 | 44176.0 / 44176.0 | 1 | 0.473889 / 0.471948 | 0.9959 | 0.977402 | 12/12 | ok |
| lj | lj98 | 4000000 | 10/12 vs 10/12 | 1 | 1321410.0 / 1321410.0 | 1 | 18.899359 / 19.479659 | 1.0307 | 0.842836 | 12/12 | ok |

## A/A Calibration Band (12 seeds, 2000-2011, serial core 0, A1 vs A2)

| Panel | System | Budget | Hits A1/A2 | Fisher p | Med Calls A1/A2 | MW p (calls) | Out Med A1/A2 (s) | Ratio A2/A1 | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.007981 / 0.008038 | 1.0071 | 0.887386 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 7/12 | 1 | 1481619.5 / 1481619.5 | 1 | 2.788901 / 2.785299 | 0.9987 | 0.977402 | 12/12 | ok |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 5897.0 / 5897.0 | 1 | 1.112656 / 1.110090 | 0.9977 | 0.629726 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.032196 / 0.030795 | 0.9565 | 0.842836 | 12/12 | ok |
| molecular | water_hexamer | 4000 | 12/12 vs 12/12 | 1 | 3026.5 / 3026.5 | 1 | 2.543518 / 2.514590 | 0.9886 | 0.218921 | 12/12 | ok |

## Placement Band (water_hexamer)

Five arm A layout variants change only object size or section alignment of `anneal._core`. Each variant is run against arm A. Their median outside ratios are pooled with the second-build A/A on `water_hexamer`. Seeds 2000-2011, serial core 0, ABBA order, one thread.

| Series | Kind | Out med left/right (s) | Ratio | Exact MW p | Bit-ID |
| --- | --- | --- | ---: | ---: | --- |
| AA | second_build | 2.543518 / 2.514590 | 0.9886 | 0.218921 | 12/12 |
| PL1 | pad32 | 2.446681 / 2.407241 | 0.9839 | 0.477575 | 12/12 |
| PL2 | pad96 | 2.437060 / 2.483889 | 1.0192 | 0.887386 | 12/12 |
| PL3 | pad400 | 2.393967 / 2.399561 | 1.0023 | 0.712535 | 12/12 |
| PL4 | pad1200 | 2.347968 / 2.345975 | 0.9992 | 0.798745 | 12/12 |
| PL5 | pad700-pagesize65536 | 2.320880 / 2.359653 | 1.0167 | 0.477575 | 12/12 |

- Pooled placement band (min and max of the six median ratios): `0.9839` to `1.0192`
- `water_hexamer` B/A median outside ratio `1.1180` is outside the placement band
- Exact Mann-Whitney of arm B outside times (n=12) against the pooled variant outside times (n=60): U=`662.0`, p=`2.81181e-07`, Bonferroni m=`5`, p_adj=`1.4059e-06`
- `water_hexamer` loses on outside time only when its ratio is above the band and p_adj < 0.05
- Untouched cells (`rq3`, `xld`, `lj38`, `lj75`, `lj98`) lose on outside time only when their ratio is above that band and their exact Mann-Whitney p < 0.05
- `tip4p21` outside ratio `0.8204` is outside the placement band
- `tip4p25` outside ratio `0.7992` is outside the placement band
- Variant and A/A trajectories match arm A: `1`

## Hexamer hop-phase breakdown

A separate instrumented build times each water_hexamer search. The TRIC step is `TricOpt::minimize_watched` outside the potential. Book handling is the stall archive and, on arm B, the stalled-walk book. Acceptance is the decision that adopts or declines the quenched trial. Bookkeeping is the rest of the search outside the potential. Potential time is recorded beside those phases. These times are not the gate verdict.

| Arm | Seeds | Tric step (s) | Book (s) | Accept (s) | Bookkeeping (s) | Potential (s) | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 12 | 1.854748 | 0.000099 | 0.000971 | 0.034320 | 16.793806 | 3997 |
| B | 12 | 1.951237 | 0.000072 | 0.000932 | 0.034790 | 16.742534 | 3997 |

## Gate Failure Details

- AB/water_hexamer: outside_placement_band:ratio=1.1180,band=0.9839-1.0192,p=2.81181e-07,p_adj=1.4059e-06
