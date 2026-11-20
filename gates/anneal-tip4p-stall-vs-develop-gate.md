# TIP4P stall-vs-develop and PR #160 gate

Status: pre-registered. The rules and bars below are fixed before any
gate row is run or read. Results are appended under "Results" after the
campaigns complete.

## Goal

Test whether `fix/tip4p-generic-stall` (`87280ef9`, carrying the
reworked `f911c2ec` archive-walk patch) and PR #160's head
`fix/tip4p-hits-b9` (`ea777eaf`) still lose TIP4P hits and outside time
per call against `develop` (`61fa67cf`), and whether PR #160 changes
trajectories or outside time relative to `87280ef9`.

## Arms

| Arm | Source | Commit | Tree | Role |
| --- | --- | --- | --- | --- |
| A | `develop` | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | baseline (`develop`) |
| A2 | `develop`, second build in its own worktree and target directory | `61fa67cf380e46876b569da45e7cc091647a775b` | `cff796c507a8fa1d0bfb28888c3570b61988055e` | A/A noise band (`develop-2`) |
| B | `origin/fix/tip4p-generic-stall` | `87280ef9745f8367d9857d757e21af35410d7b25` | `b778132b34901c95570b129acb44ab6bb0e06402` | candidate (`stall`) |
| C | `origin/fix/tip4p-hits-b9` (PR #160 head) | `ea777eafb9b80d8b93d94e68f499d66740e8a05b` | `da15bc0a999531019e6e53fcc6b6b104afc47187` | candidate (`pr160`) |

Stop conditions verify both commit and tree before any run. Each arm is
built in release mode (`cargo +1.88.0 build --release --locked`) in its
own worktree and target directory by
`scripts/tip4p_h2h/gate_stall_vs_develop.py build`. The harness refuses
to start if any arm's recorded commit, recorded tree, or live worktree
differs from the table above or if any worktree is dirty.

## Panel and protocol

Every run is one serial process pinned to CPU 0 (`taskset -c 0`). Every
`*_NUM_THREADS` and `*_MAX_THREADS` variable, including `OMP`,
`OPENBLAS`, `MKL`, `RAYON`, `BLIS`, `NUMEXPR`, `NUMBA`, `VECLIB`,
`GOTO`, `TBB`, `POLARS` and `TOKIO`, is set to `1`, with `OMP_DYNAMIC`
and `MKL_DYNAMIC` set to `FALSE`. Pinning evidence (`taskset -p` and
`Cpus_allowed_list` from `/proc/<pid>/status`) and the thread
environment variables are recorded into
`gates/anneal-tip4p-stall-vs-develop/session.json`. No `--quick` mode
or shortened budget is used, and no opening run is discarded.

### Rigid panel

Cells are `TIP4P9` through `TIP4P21` (`n = 9..21`, 13 cells). Each cell
runs seeds `2000` through `2023` (`n = 24` seeds per cell per arm)
through:

```text
h2h_seed tip4p <n> 0 2000000 <seed> recommended <stop> <prefix>
```

where `<stop>` is the Cambridge reference energy in
`data/tip4p_ccd.csv` plus `0.001`. All four arms (`A`, `A2`, `B`, `C`)
run on every `(cell, seed)` block in a balanced 4x4 Williams design
rotated by `seed - 2000 mod 4`:

1. `A, B, C, A2`
2. `B, A2, A, C`
3. `A2, C, B, A`
4. `C, A, A2, B`

Over the 24 seeds of each cell, every arm occupies each of the four
slots six times and immediately follows every other arm six times. All
arms of a seed use the same seed and therefore the same initial
geometry; the harness hashes the written `.start.gmin` file and stops
immediately if any two arms disagree.

### Non-rigid identity panel

The non-rigid panel is the thirteen systems of
`scripts/gmin_panel/run_panel.py` (`lj38`, `lj55`, `lj75`, `lj98`,
`lj102`, `lj103`, `lj104`, `lj150`, `m55r6`, `m55r14`, `m80r6`,
`m80r14`, `au55`) at their `PAPER_BUDGETS`, seeds `2000` to `2002`, run
on arms `A`, `B` and `C` in a 3-arm Williams order rotated by seed
(`A, B, C`, `B, C, A`, `C, A, B`). Each system runs its panel driver
(`h2h_gmin_proof`, `h2h_morse`, or `h2h_metal`). For Morse systems the
start file is written with `h2h_morse coords <system> <seed> <start>`
and the run is invoked with the panel's own arguments. Each
Lennard-Jones and Morse system also runs `h2h_seed` at the panel budget
with `reference + 0.001` as stop, writing the lowest geometry and the
trace of every new lowest energy. `au55` writes its best structure
through `H2H_BEST`.

## Definitions

- A hit is `energy_hits(best, reference, 0.001)`, where `best` is the
  lowest energy returned by the objective (the value printed in the
  driver's result row).
- Outside time per call on a row is `(wall_s - pot_s) / charged`. Every
  TIP4P row must have `calls == charged`; the scorer checks this and
  fails if any row differs.
- Cell medians are Hyndman-Fan type 7 over the 24 seeds of a cell.
- Peak `RssAnon` (`rss_anon_kb`, sampled from `/proc/<pid>/status`) and
  `ru_maxrss` (`ru_maxrss_kb`, from `wait4` resource usage) are
  recorded per row and summarised by peak and median per arm. Both RSS
  metrics are diagnostic only and never gate.
- Host pauses between consecutive serial runs are measured by
  `pause_gap_s = mtime(current_log) - mtime(prev_log) - wall_s`; any
  row with `pause_gap_s > 5.0` is kept and flagged.

## Pre-registered bars

The three bars below are evaluated separately for `B` vs `A` and for
`C` vs `A`:

1. **No hit loss.** For candidate arm `X` in `{B, C}` against `A`,
   per-cell exact two-sided paired McNemar tests (`n = 24` seeds per
   cell) and a pooled exact two-sided paired McNemar test over
   `TIP4P15` to `TIP4P21` (`7 * 24 = 168` seed-cell pairs) are
   computed. With `b` counting pairs where `A` hits and `X` misses and
   `c` counting pairs where `A` misses and `X` hits, the exact
   two-sided p-value is `min(1, 2 * P(Binomial(b + c, 1/2) <= min(b, c)))`
   (and `1` when `b + c = 0`). Any cell where `X` has fewer hits than
   `A` (`b > c`) is flagged, and Rule 1 fails if pooled `X`-only misses
   exceed `A`-only misses (`b > c`) on `TIP4P15` to `TIP4P21` with
   McNemar `p < 0.05` (or if per-cell hit loss is gated). Both the
   pooled McNemar verdict and the per-cell flag status are reported.
2. **Outside time per call.** In every rigid cell (`TIP4P9` through
   `TIP4P21`), the ratio of `X`'s type-7 median outside time per
   charged call to `A`'s type-7 median must be at most
   `max(1.10, med(A2) / med(A))`.
3. **Non-rigid identity.** Every non-rigid system and seed (`25`
   system-driver cells across `3` seeds = `75` pairs) must be identical
   between `A` and `X` in panel best energy, search best energy, hit,
   calls to hit, calls, charged calls, start file digest, `au55`
   best-structure digest, and `h2h_seed` best, lowest and geometry
   energies, hops, basins, trace digest, and geometry digest.

A candidate arm passes the gate only if all three rules pass.

## Additional reports

 alongside the `B` vs `A` and `C` vs `A` verdicts, the report gives:

- `C` vs `B` comparison: per-cell and pooled hit counts, exact McNemar
  test, row-by-row trajectory identity count across the 312 rigid
  `(cell, seed)` pairs, and per-cell and pooled `C/B` outside-time
  ratios.
- The complete per-cell hit table (counts and 24-seed bitstrings) for
  all four arms (`A`, `A2`, `B`, `C`).
- `A2/A` agreement: binary digests, trajectory identity across all 312
  rigid pairs, and per-cell and pooled `A2/A` outside-time ratios.
- Peak `RssAnon` (`rss_anon_kb`) and peak `ru_maxrss` (`ru_maxrss_kb`)
  per arm (reported only, never gated).

## Results

Pending.
