# anneal #204 r-next2 No-Hit-Loss Default-Panel Gate Protocol

Pre-registered protocol committed on BASE before any benchmark rows are
recorded.

## Revisions and Arms

- **BASE / Arm A**: `develop` (`cb0cfbfa0cfae21b62cc32037defc5327dc52a51`).
- **Arm B**: `anneal-204-rnext2-B` mbox (sha256
  `01c7af8cbba8258dd3eb8665ba55af38c96af7ca87d70460c686b0620ac68f15`)
  applied via `git am -3` onto `cb0cfbfa0cfae21b62cc32037defc5327dc52a51`.
  If `git am -3` applies cleanly, Arm B is that resulting tree and `A0`
  is omitted; if `git am -3` conflicts, Arm B is built on `91732899`
  (expected tree `e4b32263`) and `A0` (`91732899`) is added.
- **Arm AA**: second independent release build of Arm A in a separate
  worktree and target directory (A/A control).
- **Arm AP**: release build of Arm A with only placement/object layout
  changed via a 64 KiB `.rodata` pad (`placement-pad` feature).

All four arms compile the identical driver source in `driver/` in
`--release` mode with `rustc 1.88.0` and locked dependencies.

## Panel and Budgets

Default atomic panel at `PAPER_BUDGETS`:

| Cell | Family | Atoms | Budget | Reference | Driver |
| --- | --- | --- | --- | --- | --- |
| `lj38` | lj | 38 | 400000 | -173.928427 | `h2h_lj` |
| `lj75` | lj | 75 | 4000000 | -397.492331 | `h2h_lj` |
| `lj98` | lj | 98 | 4000000 | -543.665361 | `h2h_lj` |
| `lj102` | lj | 102 | 4000000 | -569.363652 | `h2h_lj` |
| `lj104` | lj | 104 | 4000000 | -582.086642 | `h2h_lj` |
| `m80r14` | morse | 80 | 4000000 | -340.811371 | `h2h_morse` |
| `sc108-80` | sc | 80 | 1000000 | -21819.9952 | `h2h_sc` |
| `blj130n38` | blj | 38 | 400000 | -191.1840192602 | `h2h_blj` |

- **Default panel (`panel=default`)**: `n = 12` seeds `2000..2011` per
  cell and arm (`8 * 12 * 4 = 384` runs).
- **Explicit `lj104` check (`panel=explicit`)**: `lj104` seeds `2001`
  and `2021` executed explicitly as separate fresh runs for every arm
  (`2 * 4 = 8` runs) to test the builder claims (`s2021` HIT
  `-582.08664207` at `2,593,588` calls; `s2001` MISS `-581.01667626` at
  `4,000,000`).

Hit threshold is `best` finite and `best <= reference + 0.001`. Runs
with non-zero exit status, timeout, or non-finite `best` are misses.

## Execution Protocol

- Serial execution, one process at a time, pinned to core 0 with
  `taskset -c 0`.
- All thread-pool environment variables (`OMP_NUM_THREADS`,
  `OMP_THREAD_LIMIT`, `MKL_NUM_THREADS`, `MKL_DOMAIN_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS`, `NUMBA_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, `BLIS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `GOTO_NUM_THREADS`, `TOKIO_WORKER_THREADS`, `POLARS_MAX_THREADS`,
  `TBB_NUM_THREADS`) set to `1`. No `--quick` mode; no `H2H_*` feature
  flags set.
- Rotated ABBA-style 4-arm order per seed using a balanced 4x4 Williams
  schedule on `(A, B, AA, AP)` for `k = (seed - 2000) % 4`:
  - `k = 0`: `A, B, AP, AA`
  - `k = 1`: `B, AA, A, AP`
  - `k = 2`: `AA, AP, B, A`
  - `k = 3`: `AP, A, AA, B`
  For `panel=explicit`, seed `2001` uses `A, B, AA, AP` and seed `2021`
  uses `AP, AA, B, A`.
- Rows append to `rows.csv` (`lineterminator='\n'`) recording `hit`,
  `best`, `best_bits`, `evals_to_hit`, `charged`, `wall_s`, `inside_s`,
  `outside_s`, `rss_anon_kb` (maximum of in-process `/proc/self/status`
  `h2h_mem` reading and `/proc/<pid>/status` polling), `end_rss_anon_kb`,
  `vm_hwm_kb`, and `ru_maxrss_kb`.
- If the campaign must stop early, a `PARTIAL` seal is written before
  stopping; otherwise the full panel completes before verdict generation.

## Gate Rules

Evaluated on `panel=default` (`n = 12` per cell and arm):

1. **No hit loss**: `hits_B >= hits_A` in every cell. Two-sided Fisher
   exact p-value (`scipy.stats.fisher_exact`) is reported for `B` vs `A`
   per cell.
2. **Outside-potential overhead (`outside_us`)**: per row,
   `outside_us = outside_s / max(1, charged) * 1e6`. Cell summary uses
   the Hyndman-Fan type-7 median. Require `med_B / med_A <= 1.10` in
   every cell. Report `B/A`, `AA/A`, and `AP/A` median ratios and
   two-sided Mann-Whitney U p-values (`scipy.stats.mannwhitneyu`, default
   parameters) for `B` vs `A`, `AA` vs `A`, and `AP` vs `A`.
3. **Bit-identity check**: for every atomic default cell row (and the
   explicit `lj104` runs), report whether `B` equals `A` in `hit`,
   `evals_to_hit`, `charged`, and `best` (exact 64-bit IEEE-754 bit
   pattern `best_bits`).
4. **Memory (`RssAnon`, one-sided)**: only `B` using more `RssAnon` than
   the `A/A` plus placement control band counts as a regression. For each
   cell, let `med_A`, `med_AA`, `med_AP`, `med_B` be the type-7 medians
   of peak `rss_anon_kb`. The control band upper bound is:
   `band_upper_kb = max(med_A, med_AA, med_AP, med_A + max(|med_AA - med_A|, |med_AP - med_A|, 4.0))`
   (the symmetric reflection of the observed `AA`/`AP` median shift
   around `A`, floored at one 4 KiB OS page for exact-tie page
   quantization). A cell fails memory iff `med_B > band_upper_kb`.
   `B/A`, `AA/A`, `AP/A`, and one-sided Mann-Whitney U (`alternative='greater'`)
   are also reported.

Overall verdict is **PASS** if and only if all 8 default cells complete
with `n = 12` seeds per arm, explicit `lj104` `2001`/`2021` checks pass
bit-identity, and rules (1), (2), and (4) hold in every cell; otherwise
**FIX**.
