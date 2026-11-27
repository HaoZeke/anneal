# Added trajectory and three-rule analysis

No search was rerun or replaced for this analysis.

Three-rule verdict: **PASS**.
McNemar no-loss rule: **PASS**.
Rebuild-band outside rule: **PASS**.
Non-rigid identity rule: **PASS**.

## TIP4P trajectory identity

All requested trajectories identical: **yes**.

| Cell | A==B |
| --- | ---: |
| tip4p15 | 24/24 |
| tip4p16 | 24/24 |
| tip4p17 | 24/24 |
| tip4p18 | 24/24 |
| tip4p19 | 24/24 |
| tip4p20 | 24/24 |
| tip4p21 | 24/24 |

Non-identical seeds: 0.
Their first observable trace differences are recorded in
`trajectory-divergence.csv`.

## Three-rule cell results

| Cell | b | c | McNemar p | B/A outside | A2/A | Limit | A2 identity | Non-rigid identity | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tip4p9 | 0 | 0 | 1 | 1.004011 | 1.001886 | 1.1 | 24/24 | N/A | PASS |
| tip4p10 | 0 | 0 | 1 | 1.0003514 | 0.98641296 | 1.1 | 24/24 | N/A | PASS |
| tip4p11 | 0 | 0 | 1 | 0.99434661 | 0.99798011 | 1.1 | 24/24 | N/A | PASS |
| tip4p12 | 0 | 0 | 1 | 0.98967022 | 1.0073946 | 1.1 | 24/24 | N/A | PASS |
| tip4p13 | 0 | 0 | 1 | 0.99593513 | 1.0013711 | 1.1 | 24/24 | N/A | PASS |
| tip4p14 | 0 | 0 | 1 | 0.98695357 | 0.9953318 | 1.1 | 24/24 | N/A | PASS |
| tip4p15 | 0 | 0 | 1 | 1.0037552 | 1.0035892 | 1.1 | 24/24 | N/A | PASS |
| tip4p16 | 0 | 0 | 1 | 1.0080209 | 0.99549443 | 1.1 | 24/24 | N/A | PASS |
| tip4p17 | 0 | 0 | 1 | 1.0091764 | 1.0125964 | 1.1 | 24/24 | N/A | PASS |
| tip4p18 | 0 | 0 | 1 | 0.98534932 | 0.99425611 | 1.1 | 24/24 | N/A | PASS |
| tip4p19 | 0 | 0 | 1 | 1.012467 | 1.0025792 | 1.1 | 24/24 | N/A | PASS |
| tip4p20 | 0 | 0 | 1 | 0.99446808 | 0.99384873 | 1.1 | 24/24 | N/A | PASS |
| tip4p21 | 0 | 0 | 1 | 0.99250519 | 0.99738483 | 1.1 | 24/24 | N/A | PASS |
| lj38 | 0 | 0 | 1 | 1.0035675 | 1.0011929 | 1.1 | 12/12 | 12/12 | PASS |
| lj55 | 0 | 0 | 1 | 1.0074064 | 1.0165893 | 1.1 | 12/12 | 12/12 | PASS |
| lj75 | 0 | 0 | 1 | 0.98904149 | 0.99828283 | 1.1 | 12/12 | 12/12 | PASS |
| lj98 | 0 | 0 | 1 | 0.98857845 | 0.98735724 | 1.1 | 12/12 | 12/12 | PASS |
| lj102 | 0 | 0 | 1 | 1.0004107 | 1.0043191 | 1.1 | 12/12 | 12/12 | PASS |
| lj103 | 0 | 0 | 1 | 1.0105459 | 1.016167 | 1.1 | 12/12 | 12/12 | PASS |
| lj104 | 0 | 0 | 1 | 1.00267 | 1.0004961 | 1.1 | 12/12 | 12/12 | PASS |
| lj150 | 0 | 0 | 1 | 0.99320213 | 1.0017098 | 1.1 | 12/12 | 12/12 | PASS |
| lj75-seed | 0 | 0 | 1 | 0.99057897 | 1.0011728 | 1.1 | 12/12 | 12/12 | PASS |
| m55r6 | 0 | 0 | 1 | 0.98858907 | 0.99644413 | 1.1 | 12/12 | 12/12 | PASS |
| m55r14 | 0 | 0 | 1 | 0.9912485 | 1.0041856 | 1.1 | 12/12 | 12/12 | PASS |
| m80r6 | 0 | 0 | 1 | 0.99863685 | 1.0086867 | 1.1 | 12/12 | 12/12 | PASS |
| m80r14 | 0 | 0 | 1 | 0.99499314 | 1.0069792 | 1.1 | 12/12 | 12/12 | PASS |

`b` is A-only hits and `c` is B-only hits. The no-loss rule fails
only when `b > c` and the exact two-sided McNemar p-value is below
0.05. The outside limit is `max(1.10, A2/A)` per cell.
