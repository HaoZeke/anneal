Copyright (c) anneal developers

# Same-session baseline vs adaptive population (12 Morse + 5 LJ clusters)

Baseline is `Config::recommended_with_scales` (`recommended`).
Fix is `Config::recommended_adaptive_with_scales` (`recommended_adaptive`): a two-replica fingerprint-diverse non-reversible population (`ladder_top = 2.0`) with adaptive orbit-completion throttling, acceptance-coupled step/temperature scaling, and fingerprint-diverse plateau restarts.
Seeds 2000-2011, host `bench`, serial one core (`taskset -c 0`), no wall cap.
code_revision 3f6480e72a88293b167575714abf1eec5efc83c5.
Hits reach `reference + 0.001`. Misses are censored at the paper budget.

| system | budget | base hits | fix hits | base med calls | fix med calls | base wall s | fix wall s | Fisher p | U | MW p | wall ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m38r3 | 400,000 | 12/12 | 12/12 | 735.5 | 558.5 | 0.010 | 0.006 | 1 | 60.5 | 0.524922 | 0.620 |
| m38r6 | 400,000 | 12/12 | 12/12 | 28925.0 | 26422.0 | 0.317 | 0.292 | 1 | 62 | 0.58336 | 0.921 |
| m38r10 | 400,000 | 12/12 | 12/12 | 11856.5 | 12558.0 | 0.134 | 0.148 | 1 | 81 | 0.623605 | 1.106 |
| m38r14 | 400,000 | 12/12 | 12/12 | 19529.5 | 16203.0 | 0.227 | 0.183 | 1 | 48 | 0.174853 | 0.807 |
| m55r3 | 1,000,000 | 12/12 | 12/12 | 2040.5 | 4248.0 | 0.044 | 0.087 | 1 | 95 | 0.193931 | 1.975 |
| m55r6 | 1,000,000 | 12/12 | 12/12 | 12285.0 | 11107.5 | 0.285 | 0.290 | 1 | 88 | 0.370844 | 1.019 |
| m55r10 | 1,000,000 | 12/12 | 12/12 | 104281.0 | 60232.0 | 1.980 | 1.089 | 1 | 61 | 0.54437 | 0.550 |
| m55r14 | 1,000,000 | 5/12 | 7/12 | 1000000.0 | 686999.0 | 16.887 | 12.881 | 0.684273 | 55.5 | 0.323558 | 0.763 |
| m75r6 | 4,000,000 | 11/12 | 11/12 | 777421.0 | 810548.5 | 26.635 | 25.613 | 1 | 66.5 | 0.772782 | 0.962 |
| m75r14 | 4,000,000 | 2/12 | 11/12 | 4000000.0 | 650553.0 | 126.154 | 19.484 | 0.000644194 | 21 | 0.0021699 | 0.154 |
| m80r6 | 4,000,000 | 7/12 | 12/12 | 3348190.5 | 101615.5 | 123.387 | 4.102 | 0.0372671 | 3 | 7.12247e-05 | 0.033 |
| m80r14 | 4,000,000 | 10/12 | 9/12 | 2525571.5 | 1467187.0 | 100.158 | 49.874 | 1 | 63 | 0.622086 | 0.498 |
| lj38 | 400,000 | 12/12 | 12/12 | 10615.5 | 10549.5 | 0.076 | 0.072 | 1 | 78 | 0.750832 | 0.954 |
| lj55 | 1,000,000 | 12/12 | 12/12 | 12242.0 | 11957.0 | 0.140 | 0.156 | 1 | 77 | 0.795012 | 1.111 |
| lj75 | 4,000,000 | 12/12 | 12/12 | 1409229.5 | 1186259.0 | 26.260 | 18.446 | 1 | 80 | 0.665006 | 0.702 |
| lj98 | 4,000,000 | 10/12 | 12/12 | 1511030.5 | 850801.5 | 40.430 | 22.509 | 0.478261 | 37 | 0.0463391 | 0.557 |
| lj104 | 4,000,000 | 9/12 | 11/12 | 2328723.0 | 1790122.0 | 66.596 | 52.172 | 0.590062 | 66.5 | 0.772348 | 0.783 |

Medians are Hyndman-Fan type 7. Wall ratio is fix median wall over base median wall.
Fisher is two-sided exact on hits. Mann-Whitney is two-sided asymptotic on calls
with tie and continuity corrections (fix against base; misses censored at budget).
