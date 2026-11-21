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

Verdict for `B` (`87280ef9`, `fix/tip4p-generic-stall`) vs `A`
(`61fa67cf`, `develop`): **FAIL**. Rule 1 fails, Rule 2 fails, Rule 3
passes.

Verdict for `C` (`ea777eaf`, `fix/tip4p-hits-b9`, PR #160 head) vs `A`
(`61fa67cf`, `develop`): **FAIL**. Rule 1 fails, Rule 2 fails, Rule 3
passes.

Rows are in `gates/anneal-tip4p-stall-vs-develop/rows.csv` (`1,473`
rows: `1,248` rigid across `13` cells, `24` seeds and `4` arms, plus
`225` non-rigid across `25` system-driver cells, `3` seeds and `3`
arms). Session and pinning evidence are in
`gates/anneal-tip4p-stall-vs-develop/session.json`, and per-cell and
pooled metrics are in `gates/anneal-tip4p-stall-vs-develop/summary.json`.
Every row carries hashed machine id
`3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873` and
core `0` (`taskset -p` mask `1`, `Cpus_allowed_list: 0`). Every TIP4P
row has `calls == charged`. Written start digests agree across all arms
on every seed of every cell.

Three rigid rows straddled a host pause (`pause_gap_s > 5.0 s`):
`tip4p15` seed `2011` arm `A` (`888.883 s`), `tip4p16` seed `2007` arm
`A2` (`2378.307 s`), and `tip4p17` seed `2000` arm `A` (`2346.312 s`).
No non-rigid row straddled a pause. All three rows are kept and flagged
in `rows.csv` and `summary.json`. Leaving any one out changes its cell's
median outside time by at most `0.0499 us/call` (`tip4p15` `A` moves
from `1.9765` to `1.9902`, `tip4p16` `A2` from `2.0921` to `2.0579`,
and `tip4p17` `A` from `2.2716` to `2.2217`) and alters no cell or gate
verdict.

### Rule 1 and Rule 2: per-cell hits and outside time (`n = 24`, seeds 2000-2023)

In every cell `A2` has the same hits as `A`, and `C` has the same hits,
discordant pairs (`b`, `c`) and McNemar `p` against `A` as `B` does.
Hits are shown as `total/24 (primary 2000-2011 + second 2012-2023)`.
Outside times are Hyndman-Fan type-7 medians in microseconds per charged
call.

| Cell | A (A2) hits | B (C) hits | b (A hit, B/C miss) | c (A miss, B/C hit) | McNemar p | Seeds: A hit, B/C miss | Seeds: A miss, B/C hit | A us/call | A2 us/call | B us/call | C us/call | A2/A | B/A | C/A | C/B | Rule 1 cell flag | Rule 2 (B/A, C/A) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TIP4P9 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 1.3228 | 1.3277 | 1.3344 | 1.3108 | 1.0037 | 1.0087 | 0.9909 | 0.9823 | ok | pass, pass |
| TIP4P10 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 1.4465 | 1.4442 | 1.4548 | 1.4454 | 0.9984 | 1.0058 | 0.9993 | 0.9935 | ok | pass, pass |
| TIP4P11 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 1.5676 | 1.5724 | 1.5853 | 1.5515 | 1.0031 | 1.0113 | 0.9897 | 0.9787 | ok | pass, pass |
| TIP4P12 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 1.7688 | 1.7793 | 1.7914 | 1.7425 | 1.0059 | 1.0128 | 0.9852 | 0.9727 | ok | pass, pass |
| TIP4P13 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 1.8807 | 1.8845 | 1.9135 | 1.8694 | 1.0020 | 1.0175 | 0.9940 | 0.9769 | ok | pass, pass |
| TIP4P14 | 24/24 (12+12) | 24/24 (12+12) | 0 | 0 | 1 | - | - | 2.0622 | 2.0559 | 2.0772 | 2.0439 | 0.9969 | 1.0072 | 0.9911 | 0.9840 | ok | pass, pass |
| TIP4P15 | 15/24 (7+8) | 12/24 (6+6) | 5 | 2 | 0.4531 | 2007 2010 2015 2017 2018 | 2011 2014 | 1.9765 | 1.9620 | 2.3229 | 2.2746 | 0.9927 | 1.1753 | 1.1508 | 0.9792 | flag | fail, fail |
| TIP4P16 | 7/24 (2+5) | 5/24 (1+4) | 3 | 1 | 0.6250 | 2000 2012 2018 | 2019 | 2.0924 | 2.0921 | 2.6159 | 2.5562 | 0.9999 | 1.2502 | 1.2217 | 0.9772 | flag | fail, fail |
| TIP4P17 | 9/24 (5+4) | 7/24 (3+4) | 3 | 1 | 0.6250 | 2003 2006 2023 | 2017 | 2.2716 | 2.2477 | 2.8755 | 2.8327 | 0.9895 | 1.2658 | 1.2470 | 0.9851 | flag | fail, fail |
| TIP4P18 | 5/24 (2+3) | 4/24 (3+1) | 2 | 1 | 1 | 2016 2017 | 2003 | 2.3338 | 2.3475 | 3.0574 | 2.9912 | 1.0059 | 1.3100 | 1.2817 | 0.9783 | flag | fail, fail |
| TIP4P19 | 12/24 (7+5) | 9/24 (5+4) | 4 | 1 | 0.3750 | 2001 2005 2011 2021 | 2010 | 2.5144 | 2.5001 | 3.1669 | 3.0820 | 0.9943 | 1.2595 | 1.2257 | 0.9732 | flag | fail, fail |
| TIP4P20 | 6/24 (6+0) | 5/24 (4+1) | 2 | 1 | 1 | 2006 2010 | 2013 | 2.6925 | 2.6781 | 3.3785 | 3.3319 | 0.9947 | 1.2548 | 1.2375 | 0.9862 | flag | fail, fail |
| TIP4P21 | 1/24 (0+1) | 0/24 (0+0) | 1 | 0 | 1 | 2012 | - | 2.9733 | 2.9436 | 3.7332 | 3.6713 | 0.9900 | 1.2556 | 1.2347 | 0.9834 | flag | fail, fail |

### Pooled exact McNemar tests

| Window | Comparison | Pairs | Ref hits | Cand hits | b (Ref hit, Cand miss) | c (Ref miss, Cand hit) | Exact McNemar p | Rule 1 (`b > c` and `p < 0.05`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TIP4P15-21, all seeds 2000-2023 | B vs A | 168 | 55 | 42 | 20 | 7 | 0.019157 | FAIL |
| TIP4P15-21, all seeds 2000-2023 | C vs A | 168 | 55 | 42 | 20 | 7 | 0.019157 | FAIL |
| TIP4P15-21, all seeds 2000-2023 | C vs B | 168 | 42 | 42 | 0 | 0 | 1.000000 | pass |
| TIP4P15-21, all seeds 2000-2023 | A2 vs A | 168 | 55 | 55 | 0 | 0 | 1.000000 | pass |
| TIP4P15-21, primary seeds 2000-2011 | B vs A (and C vs A) | 84 | 29 | 22 | 10 | 3 | 0.092285 | sub-window |
| TIP4P15-21, second seeds 2012-2023 | B vs A (and C vs A) | 84 | 26 | 20 | 10 | 4 | 0.179565 | sub-window |
| TIP4P9-21, all seeds 2000-2023 | B vs A (and C vs A) | 312 | 199 | 186 | 20 | 7 | 0.019157 | FAIL |

On the primary `2000-2011` subset of `TIP4P15-21`, `A` and `A2` hit
`29/84` while `B` and `C` hit `22/84`, reproducing both `develop`'s
earlier `29/84` and the PR #160 gate's `22/84` on the same machine. On
the second `2012-2023` subset (with target stop enabled), `A` and `A2`
hit `26/84` while `B` and `C` hit `20/84`. Pooled across all `24` seeds
(`168` pairs), `A` and `A2` hit `55/168` against `42/168` for `B` and
`C`, with `b = 20` vs `c = 7` (`p = 0.019157 < 0.05`), and all seven
cells `TIP4P15` through `TIP4P21` are flagged for fewer hits. Both `B`
and `C` therefore fail Rule 1.

In outside time per charged call, `A2/A` lies in `0.9895` to `1.0059`
across all `13` cells, so `max(1.10, med(A2)/med(A)) = 1.10` in every
cell. From `TIP4P15` to `TIP4P21`, `B/A` is `1.1753` to `1.3100`
(pooled `1.2902` on `TIP4P15-21`, `1.2854` on `TIP4P9-21`) and `C/A` is
`1.1508` to `1.2817` (pooled `1.2654` on `TIP4P15-21`, `1.2609` on
`TIP4P9-21`). Both `B` and `C` exceed `1.10` in all seven cells
`TIP4P15` through `TIP4P21` and fail Rule 2.

### Per-cell hit table for all four arms (seeds 2000-2023 left to right)

In every cell `A2` equals `A` and `C` equals `B` on every seed. The last
three columns count how many of the `24` seeds are bit-for-bit identical
 across `hit`, `calls`, `charged`, `best`, `search_best`, `basins`,
`trace_sha`, and `geometry_sha`.

| Cell | A (and A2) | B (and C) | A == A2 | B == C | A == B (and A == C) |
| --- | --- | --- | ---: | ---: | ---: |
| TIP4P9 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 23/24 |
| TIP4P10 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 24/24 |
| TIP4P11 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 16/24 |
| TIP4P12 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 23/24 |
| TIP4P13 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 14/24 |
| TIP4P14 | `111111111111111111111111` | `111111111111111111111111` | 24/24 | 24/24 | 22/24 |
| TIP4P15 | `110101011010010111101101` | `110101001001011010001101` | 24/24 | 24/24 | 4/24 |
| TIP4P16 | `100000100000100101100001` | `000000100000000101010001` | 24/24 | 24/24 | 0/24 |
| TIP4P17 | `001100101001010000110001` | `001000001001010001110000` | 24/24 | 24/24 | 6/24 |
| TIP4P18 | `100000000010001011000000` | `100100000010001000000000` | 24/24 | 24/24 | 1/24 |
| TIP4P19 | `110001110101100101000110` | `100000110110100101000010` | 24/24 | 24/24 | 3/24 |
| TIP4P20 | `011010100011000000000000` | `011010000001010000000000` | 24/24 | 24/24 | 2/24 |
| TIP4P21 | `000000000000100000000000` | `000000000000000000000000` | 24/24 | 24/24 | 0/24 |
| Total | 199/312 (55/168 on 15-21) | 186/312 (42/168 on 15-21) | 312/312 | 312/312 | 138/312 |

### C vs B (does PR #160 add anything on top of B?)

The production diff `87280ef9..ea777eaf` touches only
`src/methods/cluster_search.rs` to reuse the coordinate and gradient
buffers of `PaidCheck`. Across all `312` rigid `(cell, seed)` pairs and
all `75` non-rigid `(cell, driver, seed)` pairs, `C` (`ea777eaf`) and
`B` (`87280ef9`) are `100%` trajectory-identical: `312/312` rigid pairs
and `75/75` non-rigid pairs match in `hit`, `calls`, `charged`, `best`,
`search_best`, `basins`, `trace_sha`, and `geometry_sha` (`b = 0`,
`c = 0`, McNemar `p = 1.0`). In outside time per charged call, `C/B`
ranges from `0.9727` (`TIP4P12`) to `0.9935` (`TIP4P10`), with a pooled
ratio of `0.9809` over `TIP4P9-21` and `0.9808` over `TIP4P15-21` (a
`1.9%` reduction in outside time per call). That buffer reuse does not
change any trajectory or hit count on top of `B` and removes only a
small fraction of `B`'s `17.5%` to `31.0%` outside-time regression
against `develop`.

### A2/A agreement

The `A` and `A2` release builds from `61fa67cf` produce byte-identical
binaries (`h2h_seed` SHA-256
`bb6e3d0d4c7a27fab919e3048a03636a99208b6b0f86e2e1bdbfa68951ac6247`).
All `312/312` rigid pairs match between `A` and `A2` on every trajectory
and digest field. Per-cell median outside-time ratios `A2/A` lie in
`0.9895` (`TIP4P17`) to `1.0059` (`TIP4P12` and `TIP4P18`), with pooled
`A2/A` at `0.9995` over `TIP4P9-21` and `0.9994` over `TIP4P15-21`.

### Rule 3: non-rigid identity panel

All `13` systems of `scripts/gmin_panel/run_panel.py` across seeds
`2000` to `2002` (`39` panel-driver pairs and `36` `h2h_seed` pairs on
the Lennard-Jones and Morse systems = `75` pairs per comparison) are
identical between `A` and `B`, between `A` and `C`, and between `B` and
`C` on every compared field (`hit`, `evals_to_hit`, `calls`, `charged`,
`best`, `search_best`, `lowest`, `geometry_energy`, `hops`, `basins`,
`start_sha`, `trace_sha`, `geometry_sha`, including the `au55`
best-structure digests). Rule 3 passes for both `B` vs `A` and `C` vs
`A`.

### RSS (`RssAnon` and `ru_maxrss`, reported only, not gated)

Peak and median `RssAnon` (`rss_anon_kb` from `/proc/<pid>/status`) and
`ru_maxrss` (`ru_maxrss_kb` from `wait4`) in KiB by arm:

| Arm | Rigid median `RssAnon` (KiB) | Rigid peak `RssAnon` (KiB) | Rigid median `ru_maxrss` (KiB) | Rigid peak `ru_maxrss` (KiB) | Non-rigid peak `RssAnon` (KiB) | Non-rigid peak `ru_maxrss` (KiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A (`develop`) | 25,336 | 163,068 | 30,086 | 167,916 | 735,212 | 740,624 |
| A2 (`develop-2`) | 25,218 | 162,660 | 30,082 | 167,492 | - | - |
| B (`87280ef9`) | 34,436 | 204,300 | 39,106 | 208,880 | 735,472 | 740,564 |
| C (`ea777eaf`) | 34,524 | 205,552 | 39,336 | 210,312 | 735,568 | 740,500 |

Per-cell peak `RssAnon` and `ru_maxrss` (KiB) on the rigid panel:

| Cell | Peak `RssAnon` A | Peak `RssAnon` A2 | Peak `RssAnon` B | Peak `RssAnon` C | Peak `ru_maxrss` A | Peak `ru_maxrss` A2 | Peak `ru_maxrss` B | Peak `ru_maxrss` C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TIP4P9 | 2,012 | 2,060 | 2,036 | 2,060 | 25,368 | 25,368 | 25,368 | 25,368 |
| TIP4P10 | 3,172 | 3,152 | 3,136 | 3,196 | 25,624 | 25,624 | 25,624 | 25,624 |
| TIP4P11 | 20,580 | 20,536 | 20,976 | 20,980 | 25,880 | 25,880 | 25,880 | 25,880 |
| TIP4P12 | 18,020 | 17,984 | 27,296 | 27,388 | 26,008 | 26,008 | 31,956 | 32,052 |
| TIP4P13 | 33,320 | 33,360 | 25,876 | 25,904 | 38,080 | 38,256 | 30,588 | 30,480 |
| TIP4P14 | 22,392 | 22,372 | 80,452 | 80,652 | 27,208 | 27,192 | 84,992 | 85,452 |
| TIP4P15 | 101,980 | 101,704 | 122,096 | 122,368 | 106,684 | 106,344 | 126,580 | 127,160 |
| TIP4P16 | 115,612 | 115,368 | 138,500 | 138,704 | 120,404 | 120,116 | 143,184 | 143,324 |
| TIP4P17 | 131,716 | 131,532 | 161,828 | 162,016 | 136,432 | 136,224 | 166,448 | 166,748 |
| TIP4P18 | 132,528 | 132,604 | 160,752 | 160,452 | 137,368 | 137,296 | 165,252 | 165,092 |
| TIP4P19 | 137,120 | 136,960 | 177,872 | 178,040 | 141,980 | 141,680 | 182,384 | 182,880 |
| TIP4P20 | 152,820 | 152,376 | 186,536 | 186,616 | 157,628 | 157,028 | 191,176 | 191,356 |
| TIP4P21 | 163,068 | 162,660 | 204,300 | 205,552 | 167,916 | 167,492 | 208,880 | 210,312 |

### Verdict summary

| Comparison | Rule 1 (no hit loss) | Rule 2 (outside time <= max(1.10, A2/A)) | Rule 3 (non-rigid identity) | Overall Verdict | Failing / flagged cells |
| --- | --- | --- | --- | --- | --- |
| B (`87280ef9`) vs A (`61fa67cf`) | FAIL (`42/168` vs `55/168` on TIP4P15-21, `b=20, c=7, p=0.019157`) | FAIL (`1.1753x` to `1.3100x` on TIP4P15-21) | PASS (`75/75` identical) | **FAIL** | `TIP4P15` through `TIP4P21` (`7/13` cells fail Rule 1 and Rule 2) |
| C (`ea777eaf`) vs A (`61fa67cf`) | FAIL (`42/168` vs `55/168` on TIP4P15-21, `b=20, c=7, p=0.019157`) | FAIL (`1.1508x` to `1.2817x` on TIP4P15-21) | PASS (`75/75` identical) | **FAIL** | `TIP4P15` through `TIP4P21` (`7/13` cells fail Rule 1 and Rule 2) |
| C (`ea777eaf`) vs B (`87280ef9`) | `42/168` vs `42/168` (`b=0, c=0, p=1.0`, `312/312` rows identical) | `0.9727x` to `0.9935x` (`0.9808x` pooled on TIP4P15-21) | `75/75` identical | diagnostic | none (trajectory-neutral, ~1.9% faster outside per call than B) |

