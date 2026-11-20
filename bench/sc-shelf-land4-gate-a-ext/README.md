# sc-shelf-land4 gate A extension

Paired extension of gate A on the four cells that failed that gate.
The rules in this file are the analysis. They are fixed before the
new seeds run and are not revised afterwards.

## Arms and cells

- Arm A: `61fa67cf380e46876b569da45e7cc091647a775b`, tree `cff796c507a8fa1d0bfb28888c3570b61988055e`.
- Arm B: `0f0f41b7541f0aaab3e21261e58380e1468f40b8`, tree `3381b56419aeeece37ff4b13d370967b0b2ab6d0`.
- Arm A': a second `cargo build --release --locked` of arm A, fresh target directory, same commit and tree.
- Cells, drivers, budgets, and references, unchanged from gate A:

| system | driver | budget | reference |
|---|---|---:|---:|
| lj75 | `h2h_gmin_proof` | 4000000 | -397.492331 |
| m30r14 | `h2h_morse` | 400000 | -106.835790 |
| sc96-55 | `h2h_sc` | 1000000 | -13217.8963 |
| sc96-80 | `h2h_sc` | 1000000 | -19560.8966 |

- Build: `rustc 1.88.0`, `cargo +1.88.0 build --release --locked --features rgpot-ex --example h2h_gmin_proof --example h2h_morse --example h2h_sc`. The feature is required to compile `h2h_sc`. Path prefixes in the compiler flags are remapped so a host path is not embedded.
- rgpot revision `672aef430855036125c902ace024756d682e180a`. Sutton-Chen loads that build through `RGPOT_SC_LIB`.
- Hit rule: a finite `best` is a hit when `best <= reference + 0.001`. The stored hit column is that rule, re-derived. A stored hit that disagrees stops the analysis.
- New seeds: `2112..2123` only (12 seeds). Seeds `2100..2111` are read from the gate A `rows.csv` and are not rerun. That file is checked by SHA-256 and is not rewritten.
- No warm-up search and no quick flag. A `budget` query checks the table in the binary before the grid. It does not search.

## Execution

Serial, one search at a time. The process tree is pinned with `taskset -c 0`. Each search is also started under `taskset -c 0`. Every thread variable below is `1`:

`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `MKL_DOMAIN_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `GOTO_NUM_THREADS`, `OMP_THREAD_LIMIT`, `RAYON_NUM_THREADS`, `RUST_TEST_THREADS`, `TOKIO_WORKER_THREADS`, `POLARS_MAX_THREADS`.

Arm order is ABBA on the index in `2112..2123`. Index mod 4 in {0, 3} runs the first arm and then the second. The other indices run the second arm and then the first. The A/B grid uses A then B. The A/A' grid uses A then A', after the A/B grid has finished, in the same cell order. Six seeds in each grid have each arm in slot 0 and six have it in slot 1.

`rows.csv` holds the new runs only, both grids, with the gate A row columns and no seat column. A/B rows have order `AB` or `BA`. A/A' rows have order `A/A_prime` or `A_prime/A`. `sessions.csv` records the binaries and the pin, with no seat column. The machine is recorded as the SHA-256 of the operating system machine id.

## Analysis

The pooled sample is seeds `2100..2123` (n=24) for each cell. The new sample is seeds `2112..2123` (n=12). Both are reported for every test below.

Discordance counts, per cell, of the paired hits: both, A only, B only, neither.

Exact two-sided McNemar p, binomial on the discordant pairs. Let b be A only, c be B only, and n = b + c. If n is 0 the p-value is 1. Otherwise it is the sum of Binomial(n, 1/2) point masses whose binomial coefficient is at most C(n, b), divided by 2^n.

Wilcoxon signed-rank on charged calls, paired by seed, misses ranked worst. The paired value is the `charged` column. A miss adds 1000000000000, which is larger than any budget here, so every miss ranks strictly worse than every hit. Within a hit class, a larger charged count is worse. The signed difference is B minus A. Zero differences are dropped. Tied absolute differences take average ranks. The exact two-sided p-value is the fraction of the 2^k sign assignments whose absolute deviation from the null mean is at least the observed deviation. W+ is the sum of the positive ranks.

Outside time per charged call is `outside_s / charged` on that row. The arm summary is the type-7 median (the same quantile gate A uses), and the reported ratio is the median of B divided by the median of A. It is not the median of pairwise ratios and not a ratio of sums.

The bootstrap 95% interval resamples seeds with replacement, keeping each pair together, 10000 times. A new `random.Random(20261120)` is created for each system and cohort. Each draw recomputes the two type-7 medians and their ratio. The endpoints are type 7 at 0.025 and 0.975.

The new A/A band uses the A and A' rows of the new seeds only. For each cell, r is the type-7 median of A' divided by the type-7 median of A, and the cell contribution is max(r, 1/r). The new band is the maximum of those four contributions. The gate A band is the published value 1.033093. The envelope is the larger of the new band and 1.033093. A point exceedance is a B/A median ratio strictly greater than the band. A CI exceedance is a lower endpoint strictly greater than the band. Both comparisons are reported for the new band, the gate band, and the envelope.

`pooled.csv` is one row per cell and seed from 2100 through 2123. `summary.csv` is one row per cell and cohort (`pooled`, then `new`). `aa_band.csv` is one row per cell with the A/A' medians, the symmetric ratio, the new band, and the gate band. Hit strings are ordered by increasing seed, `1` for a hit and `0` for a miss.

## Results

The measurement has not been run. The tables below are filled
from rows.csv after the grid finishes. The rules above are the
analysis. They are not revised after the runs.
