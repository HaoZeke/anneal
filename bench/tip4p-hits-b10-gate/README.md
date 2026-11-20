# tip4p-hits-b10 independent gate

Verdict: **FIX**.

## Arms

| Arm | Commit | Tree | Role |
| --- | --- | --- | --- |
| `D` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `develop` baseline |
| `B9` | `b58dd00910d10cc6cc75ac1a61427730234cac0d` | `40238b525d795bf99fdf22a6ee66a4e238e4782e` | b9 baseline |
| `B10` | `65c293a5fa202f5549799591a78a9d6588ee366e` | `27132421b07aa2d58c88f13ff9a9c94bb58380a3` | b10 candidate (kit tip) |
| `B9'` | `b58dd00910d10cc6cc75ac1a61427730234cac0d` | `40238b525d795bf99fdf22a6ee66a4e238e4782e` | second independent release build of b9 (A/A band) |

## Protocol

- Panel: all 14 rigid TIP4P cells `n=8..21` with Cambridge Cluster Database references in `data/tip4p_ccd.csv`.
- Budget: `2000000` evaluations per run (`PAPER_BUDGETS`, no `H2H_*` environment variables, budget-only stop, no warm-up).
- Seeds: `2000..2011` (`n=12` per arm per cell, `672` rows total).
- Execution: single serial session (`20261120T100400Z-tip4p-hits-b10-gate`), whole process tree pinned via `taskset -c 0`, all thread environment variables set to `1`, identical starting geometries verified by SHA-256 across all four arms for every `(system, seed)`.
- Hashed machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.
- `rows.csv` sha256: `0f1b7dda5431ace9a1f1785e7ef85582f3de463cf798c8a5dcf07d5bb7aabaad`.

### Balanced ABBA rotation check

Williams-balanced 4-arm ABBA cycle across 4-seed blocks (`(D, B9, B9', B10)`, `(B10, B9', B9, D)`, `(B9, B10, D, B9')`, `(B9', D, B10, B9)`). Global `(slot, arm)` counts:

| Slot | `D` | `B9` | `B10` | `B9'` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 42 | 42 | 42 | 42 |
| 1 | 42 | 42 | 42 | 42 |
| 2 | 42 | 42 | 42 | 42 |
| 3 | 42 | 42 | 42 | 42 |

## Declared gate rules

1. **Bit-identity (`B10` vs `B9`)**: identical `best_bits` (`f64::to_bits`), `outcome_best_bits`, `calls`, and `charged` on every row (`168/168` pairs).
   - Result: **PASS** (`0` mismatches in `B10` vs `B9`; `0` mismatches in `B9` vs `B9'`).
2. **Outside time per charged call**: per-row `outside_s / charged`, Hyndman-Fan type-7 median per cell; require `B10/B9 <= 1.10` or `<= A/A spread`, and `B10/D <= 1.10` or `<= A/A spread` in every cell, where `A/A spread = max(B9/B9', B9'/B9)`.
   - Result: **FIX** (`14` failing cells).

### Rule (2) failing cells

- tip4p8: B10/D=1.1981>max(1.10,1.0045)
- tip4p9: B10/D=1.2717>max(1.10,1.0003)
- tip4p10: B10/D=1.3279>max(1.10,1.0046)
- tip4p11: B10/D=1.2811>max(1.10,1.0048)
- tip4p12: B10/D=1.3916>max(1.10,1.0019)
- tip4p13: B10/D=1.3399>max(1.10,1.0006)
- tip4p14: B10/D=1.3680>max(1.10,1.0083)
- tip4p15: B10/D=1.1787>max(1.10,1.0000)
- tip4p16: B10/D=1.2236>max(1.10,1.0199)
- tip4p17: B10/D=1.3631>max(1.10,1.0103)
- tip4p18: B10/D=1.3460>max(1.10,1.0036)
- tip4p19: B10/D=1.3444>max(1.10,1.0100)
- tip4p20: B10/D=1.2604>max(1.10,1.0057)
- tip4p21: B10/D=1.2575>max(1.10,1.0092)

## Per-cell timing (outside time per charged call)

Medians are Hyndman-Fan type-7 in microseconds per charged call (`outside_s / charged * 1e6`).

| Cell | `D` (us/call) | `B9` (us/call) | `B10` (us/call) | `B9'` (us/call) | `B10/B9` | `B10/D` | `B9/B9'` | A/A spread | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tip4p8 | 1.0757 | 1.3307 | 1.2888 | 1.3248 | 0.9685 | 1.1981 | 1.0045 | 1.0045 | FIX (B10/D=1.1981>max(1.10,1.0045)) |
| tip4p9 | 1.1215 | 1.4484 | 1.4262 | 1.4489 | 0.9847 | 1.2717 | 0.9997 | 1.0003 | FIX (B10/D=1.2717>max(1.10,1.0003)) |
| tip4p10 | 1.2475 | 1.6825 | 1.6566 | 1.6901 | 0.9846 | 1.3279 | 0.9955 | 1.0046 | FIX (B10/D=1.3279>max(1.10,1.0046)) |
| tip4p11 | 1.3012 | 1.6908 | 1.6670 | 1.6988 | 0.9859 | 1.2811 | 0.9953 | 1.0048 | FIX (B10/D=1.2811>max(1.10,1.0048)) |
| tip4p12 | 1.4368 | 2.0472 | 1.9995 | 2.0512 | 0.9767 | 1.3916 | 0.9981 | 1.0019 | FIX (B10/D=1.3916>max(1.10,1.0019)) |
| tip4p13 | 1.5342 | 2.0623 | 2.0557 | 2.0636 | 0.9968 | 1.3399 | 0.9994 | 1.0006 | FIX (B10/D=1.3399>max(1.10,1.0006)) |
| tip4p14 | 1.6656 | 2.3051 | 2.2784 | 2.3243 | 0.9884 | 1.3680 | 0.9917 | 1.0083 | FIX (B10/D=1.3680>max(1.10,1.0083)) |
| tip4p15 | 2.0092 | 2.3941 | 2.3681 | 2.3940 | 0.9892 | 1.1787 | 1.0000 | 1.0000 | FIX (B10/D=1.1787>max(1.10,1.0000)) |
| tip4p16 | 2.1446 | 2.6188 | 2.6240 | 2.6709 | 1.0020 | 1.2236 | 0.9805 | 1.0199 | FIX (B10/D=1.2236>max(1.10,1.0199)) |
| tip4p17 | 2.1580 | 2.9077 | 2.9416 | 2.9376 | 1.0117 | 1.3631 | 0.9898 | 1.0103 | FIX (B10/D=1.3631>max(1.10,1.0103)) |
| tip4p18 | 2.2939 | 3.0783 | 3.0875 | 3.0895 | 1.0030 | 1.3460 | 0.9964 | 1.0036 | FIX (B10/D=1.3460>max(1.10,1.0036)) |
| tip4p19 | 2.4508 | 3.3660 | 3.2949 | 3.3328 | 0.9789 | 1.3444 | 1.0100 | 1.0100 | FIX (B10/D=1.3444>max(1.10,1.0100)) |
| tip4p20 | 2.7151 | 3.4318 | 3.4221 | 3.4515 | 0.9972 | 1.2604 | 0.9943 | 1.0057 | FIX (B10/D=1.2604>max(1.10,1.0057)) |
| tip4p21 | 3.0403 | 3.8520 | 3.8231 | 3.8874 | 0.9925 | 1.2575 | 0.9909 | 1.0092 | FIX (B10/D=1.2575>max(1.10,1.0092)) |

Across all 14 cells: `B10/B9` median `0.9888` (range `0.9685`..`1.0117`), `B10/D` median `1.3045` (range `1.1787`..`1.3916`), A/A spread median `1.0047` (max `1.0199`).

## Informational hit counts and calls-to-hit (`B10` vs `D`)

Hits are scored at `best <= reference + 0.001` kJ/mol within the 2,000,000-call budget. Fisher p is two-sided exact on `B10` vs `D`. Mann-Whitney U and two-sided asymptotic p (tie- and continuity-corrected) compare calls-to-hit for `B10` vs `D` with misses ranked worst at `2_000_001`.

| Cell | Ref (kJ/mol) | Hits `D` | Hits `B9` | Hits `B10` | Hits `B9'` | Fisher p (`B10` vs `D`) | Med calls `D` | Med calls `B10` | MWU U | MWU p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tip4p8 | -305.5183221676 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 6339.0 | 6339.0 | 72.0 | 1.0000 |
| tip4p9 | -344.4352092012 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 5839.5 | 5839.5 | 72.0 | 1.0000 |
| tip4p10 | -391.0226842125 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 13138.5 | 13138.5 | 72.0 | 1.0000 |
| tip4p11 | -431.4890723318 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 109288.0 | 109288.0 | 68.5 | 0.8622 |
| tip4p12 | -492.9082357122 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 41418.5 | 41418.5 | 72.0 | 1.0000 |
| tip4p13 | -532.9705363445 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 216689.5 | 256396.5 | 68.0 | 0.8397 |
| tip4p14 | -582.9911036211 | 12/12 | 12/12 | 12/12 | 12/12 | 1.0000 | 135853.0 | 135853.0 | 71.5 | 1.0000 |
| tip4p15 | -628.3722525077 | 7/12 | 5/12 | 5/12 | 5/12 | 0.6843 | 1481619.5 | 2000001.0 | 81.0 | 0.5999 |
| tip4p16 | -681.1925945898 | 2/12 | 1/12 | 1/12 | 1/12 | 1.0000 | 2000001.0 | 2000001.0 | 77.5 | 0.6153 |
| tip4p17 | -723.8077531390 | 5/12 | 3/12 | 3/12 | 3/12 | 0.6668 | 2000001.0 | 2000001.0 | 81.0 | 0.5584 |
| tip4p18 | -773.2318693143 | 2/12 | 3/12 | 3/12 | 3/12 | 1.0000 | 2000001.0 | 2000001.0 | 65.0 | 0.5972 |
| tip4p19 | -821.0369191145 | 7/12 | 7/12 | 7/12 | 7/12 | 1.0000 | 1060845.0 | 1081780.5 | 72.0 | 1.0000 |
| tip4p20 | -872.9887544952 | 6/12 | 4/12 | 4/12 | 4/12 | 0.6802 | 1686250.0 | 2000001.0 | 81.5 | 0.5615 |
| tip4p21 | -916.7074302940 | 0/12 | 0/12 | 0/12 | 0/12 | 1.0000 | 2000001.0 | 2000001.0 | 72.0 | 1.0000 |

Total hits across all 14 cells (168 runs/arm): `D` `113/168`, `B9` `107/168`, `B10` `107/168`, `B9'` `107/168`.
