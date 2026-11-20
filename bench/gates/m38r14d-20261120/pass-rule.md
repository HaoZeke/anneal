# Gate pass rule for m38r14d

This gate compares arm A (`61fa67cf`) to arm B (A plus the one fix commit).

Per cell, B **FAILS** if any of the following is true:

1. **Hits test**: B has fewer hits than A and the two-sided exact Fisher p-value is `< 0.05`.
2. **Charged-call test**: B has worse ranked charged calls than A and the two-sided Mann-Whitney p-value is `< 0.05`, with misses ranked worst at the budget.
3. **Outside-time test**: Compute each row's `outside_s / charged`; if the ratio of medians `median(B) / median(A)` is `> 1.1`, B fails.

Hit rule for both arms: a run is a hit when `best <= reference + 0.001`.

Protocol:

- serial execution with `taskset -c 0`;
- thread controls forced to `1` (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS`);
- ABBA arm order per seed;
- same seed list and start policy for both arms;
- paper budgets only;
- no quick mode and no warm-up runs.
