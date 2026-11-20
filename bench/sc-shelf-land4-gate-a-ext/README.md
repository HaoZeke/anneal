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

- Hashed machine id: `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`
- Toolchain: `rustc 1.88.0 (6b00bc388 2025-06-23)`; `cargo 1.88.0 (873a06493 2025-05-10)`
- rgpot: `672aef430855036125c902ace024756d682e180a`
- librgpot SHA-256: `0328d60e84423e1a5823ab3bc72032b4888d7a1602125dea00f3c6cd6fb8ddd8`
- CPython: `3.12.3`

### Binary SHA-256

| phase | arm | driver | code_revision | tree | binary_sha256 |
|---|---|---|---|---|---|
| ab | A | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `2d78400ca48c1d5aa1f44fcc7ea89c22be1aaa1d94dcf9689092eeac64faf161` |
| ab | A | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `8f57eef60c46949c30d0afa823addbdcb528b1b6a11b6aeba7e47ef50158636b` |
| ab | A | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `5e6f5f891e4fa9335f8534e6749dad486eab979d1a3ef4f4f4664db8bd39b05d` |
| ab | B | `h2h_gmin_proof` | `0f0f41b7541f0aaab3e21261e58380e1468f40b8` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `05b1a78d6583bbdb6e11b9554d99bb63ecfac4c9607a720e8d2ddf6c555b01d5` |
| ab | B | `h2h_morse` | `0f0f41b7541f0aaab3e21261e58380e1468f40b8` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `9242ceefba2f04f791df5bf4ba2229f57ddd6f8d02f86fc6b90c7389a3ae0cee` |
| ab | B | `h2h_sc` | `0f0f41b7541f0aaab3e21261e58380e1468f40b8` | `3381b56419aeeece37ff4b13d370967b0b2ab6d0` | `99dbd8b2870d0e245e6d2cfddab15a5214ecb34d78393b4a5bca0891036888d7` |
| aa | A | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `2d78400ca48c1d5aa1f44fcc7ea89c22be1aaa1d94dcf9689092eeac64faf161` |
| aa | A | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `8f57eef60c46949c30d0afa823addbdcb528b1b6a11b6aeba7e47ef50158636b` |
| aa | A | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `5e6f5f891e4fa9335f8534e6749dad486eab979d1a3ef4f4f4664db8bd39b05d` |
| aa | A_prime | `h2h_gmin_proof` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `2d78400ca48c1d5aa1f44fcc7ea89c22be1aaa1d94dcf9689092eeac64faf161` |
| aa | A_prime | `h2h_morse` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `8f57eef60c46949c30d0afa823addbdcb528b1b6a11b6aeba7e47ef50158636b` |
| aa | A_prime | `h2h_sc` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | `5e6f5f891e4fa9335f8534e6749dad486eab979d1a3ef4f4f4664db8bd39b05d` |

### A/A band on the new seeds

| system | med A | med A' | A'/A | max(r, 1/r) |
|---|---:|---:|---:|---:|
| lj75 | 6.275287e-06 | 6.347883e-06 | 1.011569 | 1.011569 |
| m30r14 | 1.942091e-06 | 1.908366e-06 | 0.982635 | 1.017672 |
| sc96-55 | 4.345664e-06 | 4.447089e-06 | 1.023339 | 1.023339 |
| sc96-80 | 9.667558e-06 | 9.686998e-06 | 1.002011 | 1.002011 |

New band `1.023339`. Gate A band `1.033093`. Envelope `1.033093`.

### Paired tests

| system | cohort | n | both | A only | B only | neither | McNemar p | W+ | Wilcoxon n | Wilcoxon p | med opc B/A | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj75 | pooled | 24 | 4 | 8 | 4 | 8 | 0.387695 | 86 | 15 | 0.151428 | 0.979644 | [0.949643, 1.022578] |
| lj75 | new | 12 | 2 | 2 | 2 | 6 | 1 | 13 | 6 | 0.6875 | 0.994010 | [0.938302, 1.046222] |
| m30r14 | pooled | 24 | 17 | 2 | 2 | 3 | 1 | 17 | 7 | 0.6875 | 1.019904 | [0.986765, 1.039300] |
| m30r14 | new | 12 | 9 | 0 | 1 | 2 | 1 | 1 | 2 | 1 | 1.019746 | [1.012799, 1.031635] |
| sc96-55 | pooled | 24 | 24 | 0 | 0 | 0 | 1 | 3 | 2 | 0.5 | 1.010015 | [0.960099, 1.059559] |
| sc96-55 | new | 12 | 12 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0.976289 | [0.940267, 1.028192] |
| sc96-80 | pooled | 24 | 9 | 4 | 0 | 11 | 0.125 | 10 | 4 | 0.125 | 1.060161 | [1.004697, 1.106769] |
| sc96-80 | new | 12 | 4 | 2 | 0 | 6 | 0.5 | 3 | 2 | 0.5 | 1.031588 | [0.999687, 1.056834] |

### Band comparisons

A 1 means the B/A median ratio, or its lower CI endpoint, is strictly above that band.

| system | cohort | ratio | CI low | point new | point gate | point envelope | CI new | CI gate | CI envelope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lj75 | pooled | 0.979644 | 0.949643 | 0 | 0 | 0 | 0 | 0 | 0 |
| lj75 | new | 0.994010 | 0.938302 | 0 | 0 | 0 | 0 | 0 | 0 |
| m30r14 | pooled | 1.019904 | 0.986765 | 0 | 0 | 0 | 0 | 0 | 0 |
| m30r14 | new | 1.019746 | 1.012799 | 0 | 0 | 0 | 0 | 0 | 0 |
| sc96-55 | pooled | 1.010015 | 0.960099 | 0 | 0 | 0 | 0 | 0 | 0 |
| sc96-55 | new | 0.976289 | 0.940267 | 0 | 0 | 0 | 0 | 0 | 0 |
| sc96-80 | pooled | 1.060161 | 1.004697 | 1 | 1 | 1 | 0 | 0 | 0 |
| sc96-80 | new | 1.031588 | 0.999687 | 1 | 0 | 0 | 0 | 0 | 0 |

### Per-seed hit strings

| system | cohort | A | B |
|---|---|---|---|
| lj75 | pooled | `001110111110100011001000` | `110010010000010010101000` |
| lj75 | new | `100011001000` | `010010101000` |
| m30r14 | pooled | `110111101111011111011110` | `100110111111011111011111` |
| m30r14 | new | `011111011110` | `011111011111` |
| sc96-55 | pooled | `111111111111111111111111` | `111111111111111111111111` |
| sc96-55 | new | `111111111111` | `111111111111` |
| sc96-80 | pooled | `000011110111010011110010` | `000011100110000011110000` |
| sc96-80 | new | `010011110010` | `000011110000` |
