# Anneal #179 r8 Default-On Panel Report

## Verdict: FIX

Arm B (`develop` `cb0cfbfa0cfae21b62cc32037defc5327dc52a51` plus the 19
commits in `anneal-179-r8-B.mbox`, reachable as
`c1d56fb54df96f53ce7b082e680e9c40dabe74b7`, tree
`dd97e61fbe57f633fa131794918d647a3a20b6c3`; recorded head
`66ff08090348296009a1b1c78b76034dddc6f4b0`) **fails** the no-hit-loss
default-on bar against Arm A (`cb0cfbfa0cfae21b62cc32037defc5327dc52a51`,
tree `65972b37b798ebacd9d99d789df761099eefea1f`):

1. **Hit loss in 3 of 8 cells**:
   - `lj38`: B 11/12 vs A 12/12 (AA 12/12, AP 12/12; Fisher p = 1)
   - `lj104`: B 2/12 vs A 5/12 (AA 5/12, AP 5/12; Fisher p = 0.3707)
   - `blj130n38`: B 7/12 vs A 11/12 (AA 11/12, AP 11/12; Fisher p = 0.1550)
   B improved hits on `lj98` (11/12 vs 7/12), `m80r14` (8/12 vs 6/12),
   and `sc108-80` (1/12 vs 0/12), and matched A on `lj75` (5/12) and
   `lj102` (6/12).
2. **Outside-potential overhead exceeds 1.10x in all 8 cells**:
   Type-7 median `outside_us` (`outside_s / charged * 1e6`) B/A ratios
   range from `1.1117` (`lj38`) to `1.3038` (`lj102`), with Mann-Whitney
   U `p < 0.005` on 6 of 8 cells (`lj38`, `lj75`, `lj98`, `lj102`,
   `lj104`, `m80r14`). Meanwhile the `AA/A` noise band remains centered
   around `0.9816` to `1.0698` (all `MWU p >= 0.544`) and the placement
   band `AP/A` spans `0.9055` to `1.0645` (all `MWU p >= 0.436`).

## Panel Execution
- Total rows: `384` (`8` cells x `12` seeds `2000..2011` x `4` arms
  `A`, `B`, `AA`, `AP`).
- Execution mode: serial (`1` process at a time), pinned to core 0
  (`taskset -c 0`), single-threaded BLAS/OpenMP/Rayon env, `PAPER_BUDGETS`,
  interleaved rotated ABBA order per seed.
  Both arms stop when `best <= reference + 0.001`, and otherwise run to
  the cell budget.
- Memory metric: peak in-process `RssAnon` (KiB) read from
  `/proc/self/status` by the benchmark process after exec.

## Per-Cell Summary (n = 12 per arm)

| Cell | Hits A / B / AA / AP | Fisher p (B/A) | Calls Med A / B / AA | Out us Med A / B / AA / AP | Out B/A (MWU p) | Out AA/A (MWU p) | Out AP/A (MWU p) | RssAnon KiB Med A / B / AA / AP | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lj38` | 12 / 11 / 12 / 12 | 1 | 127839.5 / 182857.5 / 127839.5 | 4.7024 / 5.2279 / 4.6160 / 4.3824 | 1.1117 (0.00426) | 0.9816 (0.707) | 0.9320 (0.583) | 3908 / 4900 / 3908 / 3908 | FIX |
| `lj75` | 5 / 5 / 5 / 5 | 1 | 4000000 / 4000000 / 4000000 | 7.3880 / 8.7545 / 7.5543 / 7.8647 | 1.1850 (0.00426) | 1.0225 (0.840) | 1.0645 (0.583) | 120446 / 125020 / 120446 / 120446 | FIX |
| `lj98` | 7 / 11 / 7 / 7 | 0.155 | 3166811.5 / 1504879.5 / 3166811.5 | 10.0700 / 12.1621 / 10.0561 / 10.0636 | 1.2078 (0.00355) | 0.9986 (0.977) | 0.9994 (0.977) | 120256 / 80626 / 120256 / 120258 | FIX |
| `lj102` | 6 / 6 / 6 / 6 | 1 | 3256052.5 / 3786338 / 3256052.5 | 11.0163 / 14.3627 / 11.1702 / 11.1987 | 1.3038 (7.66e-5) | 1.0140 (0.665) | 1.0166 (0.977) | 141346 / 199352 / 141346 / 141346 | FIX |
| `lj104` | 5 / 2 / 5 / 5 | 0.371 | 4000000 / 4000000 / 4000000 | 11.0983 / 13.7378 / 11.3168 / 11.0417 | 1.2378 (0.00244) | 1.0197 (0.665) | 0.9949 (0.977) | 151614 / 207976 / 151612 / 151612 | FIX |
| `m80r14` | 6 / 8 / 6 / 6 | 0.680 | 3901429 / 3069083.5 / 3901429 | 9.3244 / 11.4922 / 9.2870 / 9.3781 | 1.2325 (1.23e-4) | 0.9960 (0.544) | 1.0058 (0.507) | 192018 / 185268 / 192014 / 192018 | FIX |
| `sc108-80` | 0 / 1 / 0 / 0 | 1 | 1000000 / 1000000 / 1000000 | 15.3586 / 17.3734 / 15.3307 / 16.2579 | 1.1312 (0.285) | 0.9982 (0.665) | 1.0586 (0.436) | 40910 / 39824 / 40908 / 40908 | FIX |
| `blj130n38` | 11 / 7 / 11 / 11 | 0.155 | 134072.5 / 329587.5 / 134072.5 | 12.9464 / 16.4632 / 13.8503 / 11.7236 | 1.2716 (0.751) | 1.0698 (0.977) | 0.9055 (0.885) | 26236 / 28264 / 26238 / 26238 | FIX |

## B-Path Execution Evidence
- **B-only counter status**: There is **no** B-only counter or log line in
  the final B tree (`dd97e61fbe57f633fa131794918d647a3a20b6c3`, commit
  `c1d56fb54df96f53ce7b082e680e9c40dabe74b7`). Commit 11
  of the 19-commit series introduced `safe6_nearest_index` debug prints,
  and commit 17 (`fix: keep a finite lane and the reference order`)
  explicitly removed that counter and its prints ("The nearest-index
  counter and its prints are gone.").
- **Pre-flight probe verification (`probe.log`)**:
  - `A`: `csm_group=D_5h csm_measure=0.020856 detect_low_orders=26 oct_csm_group=C_i`
  - `B`: `csm_group=none csm_measure=NaN detect_low_orders=24 oct_csm_group=detected`
  - `AP`: `csm_group=D_5h csm_measure=0.020856 detect_low_orders=26 oct_csm_group=C_i`
  The `B` binary returns `csm_group=none` on the identity-rotation test
  (from commit 16's identity-skip guard) and `oct_csm_group=detected`
  from the r8 `csm::propose` path.
- **Symmetrisation column**: `rows.csv` records `symmetrised=0` and
  `sym_gain=0.000000` on every row. The driver prints proposal counts
  on stderr and does not copy them into `rows.csv`.

## Controls and Determinism
- **A/A control (`AA`)**: 12/12 runs in every cell (96/96 total) are
  bit-identical to `A` in `hit`, `evals_to_hit`, `charged`, and `best`.
  Outside-potential `AA/A` median ratios span `0.9816` to `1.0698`
  (all `MWU p >= 0.544`).
- **Placement control (`AP`)**: Rebuilt `A` with `--features placement-pad`
  (`PLACEMENT_PAD` is a 65536-byte array in `driver/src/main.rs`, with
  `r8_placement_pad_stub`). `rows.csv` records one `bin_sha256` for `AP`
  and a different value for `A` and `AA`.
  Outside-potential `AP/A` median ratios span `0.9055` to `1.0645`
  (all `MWU p >= 0.436`), confirming binary layout jitter does not
  explain the `1.11x–1.30x` outside-potential overhead in `B`.

## Additional Notes on Total Wall Time and Memory
- While per-call outside-potential time (`outside_us`) increases
  `1.11x–1.30x` across every cell in `B`, total median `wall_s` is
  lower in `B` on `lj98` (`43.39s` vs `81.33s`) and `m80r14` (`121.25s`
  vs `147.68s`) due to fewer calls to hit, and higher on `lj38`
  (`1.62s` vs `1.08s`), `lj102` (`118.90s` vs `91.11s`), `lj104`
  (`120.18s` vs `109.72s`), and `blj130n38` (`7.56s` vs `2.31s`).
- Peak `RssAnon` scales with evaluated steps / archive size; `B/A`
  `RssAnon` median ratios range from `0.6705` (`lj98`) to `1.4104`
  (`lj102`).
