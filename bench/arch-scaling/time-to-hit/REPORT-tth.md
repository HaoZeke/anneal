# Time-to-hit worker scaling

Shared communicating ensemble, replicas as operating-system threads,
seeds 2000-2011, and p = 1, 2, 4, 8. Runs stop at the first objective
energy at or below the reference plus the panel tolerance of 0.001.
Misses enter calls-to-hit as budget + 1. Wall medians use hits only.
Intervals on calls, geometric means, wall time, and efficiency are
bootstrap percentile 95% intervals from 10000 resamples with seed
20261123. Hit intervals are Wilson 95% intervals. Tests are two-sided
against p=1; MWU uses the tie-corrected normal approximation and
continuity correction.

References and sources:

- LJ38: -173.928427; paper/jctc/manuscript.tex, Cambridge references.
- Au38 Gupta: -130.54001708; examples/h2h_metal.rs, Au38 system specification.
- M55 rho=6 Morse: -250.286609; paper/jctc/manuscript.tex, Cambridge references.
- LJ38 budget: 400000; PAPER_BUDGETS in scripts/paper_0120/run_panel.py.

Au38 p=1 gate: 8/12 hits. It did not meet the 10/12 gate,
so the complete second matrix uses M55 rho=6 Morse.

## LJ38

| p | hits (Wilson CI) | calls median (CI) | calls geometric mean (CI) | wall median s (CI) | efficiency (CI) | Fisher p | MWU p | ledger total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10/12 ([0.552, 0.953]) | 75968.500 ([52994, 302434]) | 87337.762 ([41023.701, 172297.066]) | 3.792 ([2.132, 10.499]) | 1 ([1, 1]) |  |  | 1877317 |
| 2 | 9/12 ([0.468, 0.911]) | 109652.500 ([55843.500, 392113.500]) | 111672.383 ([55698.788, 204798.919]) | 2.407 ([1.174, 4.241]) | 0.7876 ([0.4751, 1.8057]) | 1 | 0.5426 | 2201112 |
| 4 | 8/12 ([0.391, 0.862]) | 269599 ([90320.500, 400001]) | 162741.864 ([82908.100, 288235.599]) | 2.262 ([0.256, 5.936]) | 0.4191 ([0.2083, 2.7952]) | 0.6404 | 0.2110 | 2924206 |
| 8 | 9/12 ([0.468, 0.911]) | 177722 ([129071, 365347]) | 169433.899 ([105219.784, 256934.239]) | 1.699 ([0.281, 2.423]) | 0.2790 ([0.1763, 1.1572]) | 1 | 0.2125 | 2608891 |

For every row, oracle_calls = ledger_charged = ledger_sum =
objective_calls. The table reports their per-cell sum.

## Au38 Gupta

| p | hits (Wilson CI) | calls median (CI) | calls geometric mean (CI) | wall median s (CI) | efficiency (CI) | Fisher p | MWU p | ledger total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8/12 ([0.391, 0.862]) | 295647.500 ([141093.500, 400001]) | 227507.662 ([156678.304, 318829.635]) | 6.264 ([4.670, 12.839]) | 1 ([1, 1]) |  |  | 3215224 |

For every row, oracle_calls = ledger_charged = ledger_sum =
objective_calls. The table reports their per-cell sum.

## M55 rho=6 Morse

| p | hits (Wilson CI) | calls median (CI) | calls geometric mean (CI) | wall median s (CI) | efficiency (CI) | Fisher p | MWU p | ledger total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 12/12 ([0.758, 1]) | 49982 ([27639, 70493.500]) | 41502.764 ([25103.857, 65885.230]) | 5.657 ([3.071, 8.447]) | 1 ([1, 1]) |  |  | 675094 |
| 2 | 12/12 ([0.758, 1]) | 40001 ([17791, 72724]) | 36369.181 ([24333.236, 53234.435]) | 1.966 ([0.825, 3.890]) | 1.4389 ([0.7154, 3.0005]) | 1 | 0.7075 | 544870 |
| 4 | 12/12 ([0.758, 1]) | 50297 ([33603, 124367]) | 56666.308 ([36644.907, 88409.687]) | 1.319 ([0.814, 3.730]) | 1.0721 ([0.3824, 2.0665]) | 1 | 0.6650 | 913917 |
| 8 | 12/12 ([0.758, 1]) | 66512.500 ([46872.500, 73039]) | 66901.421 ([49572.061, 94762.646]) | 0.869 ([0.567, 1.040]) | 0.8140 ([0.4134, 1.6308]) | 1 | 0.2366 | 986884 |

For every row, oracle_calls = ledger_charged = ledger_sum =
objective_calls. The table reports their per-cell sum.

Measurement revision: 8b44d65d797c167b41c67da78df5f35910319b07
