# tip4p-hits-b Regression and Performance Gate

## Verdict: `FAIL`

- Overall outside-potential wall ratio (B / A): `0.9562` (`571.768 s` vs `597.966 s`, limit `<= 1.10x`)
- A/A calibration band (4 cells, including `tip4p15`): max cell outside ratio `1.0026`, overall A/A outside ratio `1.0103`, min exact Mann-Whitney outside p `0.589876`
- Test suite parity (`tests.txt`): `PASS (zero B-only failures)`

## Test Suite Results

```text
suite=cargo_test_default arm=A cmd="cargo +1.88.0 test --locked" status=ok suites=187 passed=2247 failed=0 ignored=0
suite=cargo_test_default arm=B cmd="cargo +1.88.0 test --locked" status=ok suites=187 passed=2249 failed=0 ignored=0
suite=cargo_test_ci_features arm=A cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=ok suites=187 passed=2532 failed=0 ignored=0
suite=cargo_test_ci_features arm=B cmd="cargo +1.88.0 test --locked --features ira,graphkey,featomic,vesin-nl,bank-rpc,capi,python,rgpot-ex" status=ok suites=187 passed=2534 failed=0 ignored=0
suite=pytest arm=A cmd="python -m pytest pytest/" status=ok passed=1688 failed=0 skipped=87
suite=pytest arm=B cmd="python -m pytest pytest/" status=ok passed=1688 failed=0 skipped=87
b_only_failures=0
verdict=PASS
```

## A/B Panel (12 seeds, 2000-2011, serial core 0, ABBA per-seed order)

| Panel | System | Budget | Hits A/B | Fisher p | Med Calls A/B | MW p (calls) | Out Med A/B (s) | Ratio B/A | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.008304 / 0.008240 | 0.9923 | 0.887386 | 12/12 | ok |
| tip4p | tip4p12 | 2000000 | 12/12 vs 12/12 | 1 | 41418.5 / 41418.5 | 1 | 0.072350 / 0.073291 | 1.0130 | 0.887386 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 12/12 | 0.0372671 | 1481619.5 / 1314291.5 | 0.413966 | 2.755449 / 2.345944 | 0.8514 | 0.347358 | 7/12 | ok |
| tip4p | tip4p20 | 2000000 | 6/12 vs 6/12 | 1 | 1686249.5 / 1686249.5 | 1 | 4.260037 / 4.032924 | 0.9467 | 0.67066 | 8/12 | ok |
| tip4p | tip4p21 | 2000000 | 0/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 6.185219 / 5.282321 | 0.8540 | 0.00291403 | 3/12 | ok |
| tip4p | tip4p25 | 2000000 | 1/12 vs 0/12 | 1 | 2000000.0 / 2000000.0 | 1 | 8.260717 / 6.531123 | 0.7906 | 0.00231643 | 2/12 | ok |
| molecular | water_hexamer | 4000 | 12/12 vs 12/12 | 1 | 3026.5 / 3026.5 | 1 | 2.316045 / 2.603662 | 1.1242 | 5.17722e-06 | 12/12 | FAIL |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 12738.0 / 12738.0 | 1 | 1.060285 / 1.078326 | 1.0170 | 0.513723 | 12/12 | ok |
| conformer | xld | 44988 | 10/12 vs 10/12 | 1 | 19412.5 / 19412.5 | 1 | 0.972585 / 0.974589 | 1.0021 | 0.842836 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.030406 / 0.030148 | 0.9915 | 0.887386 | 12/12 | ok |
| lj | lj75 | 4000000 | 12/12 vs 12/12 | 1 | 44176.0 / 44176.0 | 1 | 0.464682 / 0.463020 | 0.9964 | 0.842836 | 12/12 | ok |
| lj | lj98 | 4000000 | 10/12 vs 10/12 | 1 | 1321410.0 / 1321410.0 | 1 | 18.138179 / 18.173401 | 1.0019 | 0.887386 | 12/12 | ok |

## A/A Calibration Band (12 seeds, 2000-2011, serial core 0, A1 vs A2)

| Panel | System | Budget | Hits A1/A2 | Fisher p | Med Calls A1/A2 | MW p (calls) | Out Med A1/A2 (s) | Ratio A2/A1 | MW p (out) | Bit-ID | Gate |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| tip4p | tip4p8 | 2000000 | 12/12 vs 12/12 | 1 | 6339.0 / 6339.0 | 1 | 0.008608 / 0.007957 | 0.9244 | 0.842836 | 12/12 | ok |
| tip4p | tip4p15 | 2000000 | 7/12 vs 7/12 | 1 | 1481619.5 / 1481619.5 | 1 | 2.747809 / 2.755072 | 1.0026 | 0.842836 | 12/12 | ok |
| conformer | rq3 | 56914 | 12/12 vs 12/12 | 1 | 12738.0 / 12738.0 | 1 | 1.000142 / 1.001857 | 1.0017 | 0.589876 | 12/12 | ok |
| lj | lj38 | 400000 | 12/12 vs 12/12 | 1 | 6357.0 / 6357.0 | 1 | 0.030685 / 0.030680 | 0.9998 | 0.932301 | 12/12 | ok |

## Gate Failure Details

- AB/water_hexamer: outside_regression_beyond_AA:ratio=1.124,p=5.17722e-06
