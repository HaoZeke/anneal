# Anneal #179 r8 Default-On Panel Protocol

## Objective
Evaluate whether anneal PR #179 r8 (arm B) meets the default-on bar
against `develop` (arm A) across eight benchmark cells with an
interleaved A/A noise control and a placement-shifted A control (`AP`).

## Arms
- `A`: `develop` at commit `cb0cfbfa0cfae21b62cc32037defc5327dc52a51`.
- `AA`: Second interleaved execution of the `A` release binary (noise band).
- `AP`: `A` rebuilt with `placement-pad` (64 KiB `.rodata` padding + cold stub,
  shifting binary section layout/object sizes with identical algorithm).
- `B`: `A` + 19 commits from `anneal-179-r8-B.mbox`, tree
  `dd97e61fbe57f633fa131794918d647a3a20b6c3`.

## Cells and PAPER_BUDGETS
Hit rule: `best <= reference + 0.001` (`DEFAULT_HIT_TOL`).

| Cell | Atoms | Potential | Budget | Reference |
| --- | --- | --- | --- | --- |
| `lj38` | 38 | Lennard-Jones | 400000 | -173.928427 |
| `blj130n38` | 38 (13 A, 25 B, sigma_BB=1.30) | Binary LJ | 400000 | -191.1840192602 |
| `sc108-80` | 80 | Sutton-Chen (10-8, c=34.408) | 1000000 | -21819.9952 |
| `lj75` | 75 | Lennard-Jones | 4000000 | -397.492331 |
| `lj98` | 98 | Lennard-Jones | 4000000 | -543.665361 |
| `lj102` | 102 | Lennard-Jones | 4000000 | -569.363652 |
| `lj104` | 104 | Lennard-Jones | 4000000 | -582.086642 |
| `m80r14` | 80 | Morse (rho=14.0) | 4000000 | -340.811371 |

## Execution and Controls
- Seeds: `2000..2011` (`n = 12` per cell per arm), identical starts per seed.
- Execution: serial (one process at a time), pinned via `taskset -c 0`,
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `RAYON_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`.
- Interleaved ABBA schedule per seed `k = seed - 2000`:
  - Base 4-arm cycle `[A, B, AA, AP]` rotated by `(k // 2) % 4` and reversed
    when `k % 2 == 1` (each arm occupies every slot 3 times over 12 seeds).
- Memory: `RssAnon` (KiB) read in-process by the benchmark binary from
  `/proc/self/status` at start, periodically every 4096 charged calls, and
  at completion, reporting peak `RssAnon`.
- Persistence: rows written append-only with `fsync` to `rows.csv` and
  `journal.jsonl` after each run.

## B-Path Evidence
Commit 11 of r8 added `safe6_nearest_index` prints, and commit 17
(`fix: keep a finite lane and the reference order`) explicitly removed
that counter and its prints. Pre-flight `driver probe` verifies that `B`
executes the new `csm::propose` (`group = "detected"`) path, and per-run
logs record `outcome.symmetrised` and `outcome.proposal_counts`.

## Decision Bar
- Completeness: all 8 cells have `n = 12` rows for `A`, `B`, `AA`.
- Hit gate: `hits_B >= hits_A` in every cell (report two-sided Fisher exact p).
- Outside-potential time gate: Hyndman-Fan type-7 median of
  `outside_us = outside_s / charged * 1e6` satisfies `B/A <= 1.10` in every
  cell (report two-sided tie/continuity-corrected Mann-Whitney U p and
  compare against `AA/A` noise band).
- Verdict:
  - `INCONCLUSIVE`: missing rows (`n < 12` in any cell for `A`, `B`, `AA`) or
    `B/A > 1.10` inside an `AA/A > 1.10` noise band with `MWU p >= 0.05`.
  - `FIX`: `hits_B < hits_A` in any cell, or `outside_us B/A > 1.10`
    not explained by the `AA/A` noise band (`MWU p < 0.05` or `B/A > AA/A`).
  - `PASS`: complete panel with `hits_B >= hits_A` and `outside_us B/A <= 1.10`
    in every cell.
