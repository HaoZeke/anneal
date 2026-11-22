# 153 r-next2 performance and parity gate

Arm A is develop `61fa67cf380e46876b569da45e7cc091647a775b`. Arm B is the
land9 r-next2 series, 27 patches on A, commit `72c13523` with tree
`94568f3903d1cd1fbb387a8b4423421a93f7039b`. Here B is built from the
mbox applied with `git am -3` on A. The applied tree must equal that
tree. Arm S is the same B tree built with
`RUSTFLAGS='--cfg anneal_force_scalar'`, so every `*_available()` check in
the library returns false on the same hardware. The gate record is this
directory on A. It does not include arm B's commits and changes no
library source.

The harness is `bench/nm3-land9-full-gate/run_gate.py` from
`record/nm3-land9-full-gate` (`1a94dc5e`), extended. The cells,
budgets, references, and seeds are that gate's.

## Builds

Each arm is a release build from its own worktree, with its own target
directory, on stable Rust 1.100.0:

```
cargo build --release --locked --features rgpot-ex \
  --example h2h_gmin_proof --example h2h_seed --example h2h_metal \
  --example h2h_morse --example h2h_sc
```

A and B set no `RUSTFLAGS`. S sets `RUSTFLAGS='--cfg anneal_force_scalar'`.
The compiler is 1.89 or newer, so `build.rs` also sets `anneal_avx512`
and B carries the AVX-512 scans. A2 is a second build of A from a fourth
worktree. `binary-sha256.txt` lists every binary. A row stores the
sha256 of the binary that produced it.

The Sutton-Chen cells load `rgpot_sutton_chen_energy_forces` from one
shared library that all arms use. It is built from rgpot
`8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e`, the revision `Cargo.toml`
pins for `rgpot-pair`, with
`g++ -std=c++20 -O2 -fPIC -shared -ffp-contract=off -ICppCore` over
`SuttonChen/SuttonChenPot.cc`, `SuttonChen/SuttonChenScales.cc`,
`nlist/HalfShell.cc`, `nlist/HalfShellWalk.cc`, `nlist/NeighbourList.cc`,
`nlist/Neighbours.cc`, `nlist/MinimumImage.cpp`, `simd/exact_fma_hw.cc`,
and `simd/exp.cc` under `CppCore/rgpot`, without OpenMP. Its sha256 is in
`binary-sha256.txt`. The run passes it as `RGPOT_SC_LIB`.

## Run

One process at a time, pinned with `taskset -c 0`. The process is fresh
for every seed. Nothing is discarded as a warm-up, and no quick flag is
passed.

Every thread-count variable is `1`: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, `RAYON_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`GOTO_NUM_THREADS`, `OMP_THREAD_LIMIT`, `MKL_DOMAIN_NUM_THREADS`,
`JULIA_NUM_THREADS`, `TBB_NUM_THREADS`, `POLARS_MAX_THREADS`,
`OPENMP_NUM_THREADS`, and any inherited variable whose name ends in
`_NUM_THREADS`, `_MAX_THREADS`, or `_MAXIMUM_THREADS`. `OMP_DYNAMIC` and
`MKL_DYNAMIC` are `FALSE`. Variables named `H2H_*` are unset, so each
binary keeps the harness preset.

Seeds are 2000 through 2011 (n = 12). In an `abs` phase each seed runs
three processes. An even seed runs A, B, S. An odd seed runs S, B, A.
A against B is then ABBA, and so is B against S. Cells are not
interleaved. The order block below is the whole schedule.

The A/A band runs A against A2 on LJ75, LJ38, LJ98, and LJ104, labelled
A1 and A2. A1 is the arm A binary. A2 is the second A build. Even seeds
run A1 then A2. Odd seeds run A2 then A1. The band is the noise floor
for outside time and memory.

The run refuses to start unless each `run` binary prints the budget and
reference of the cells block, on all four builds.

LJ cells run `h2h_gmin_proof run`, which runs anneal only. No GMIN
executable, GMIN input, or GMIN arm is part of this gate. The seed
driver `h2h_seed` writes its own lowest geometry under a per-run
temporary prefix. The harness hashes that prefix's trace and xyz files
for parity and the directory goes away with the run.

## Probe

`probe.c` is compiled by the run with
`cc -O2 -fPIC -shared -Wall -Wextra -Werror` and loaded into each search
with `LD_PRELOAD`. It is inert in every process except the one whose
`/proc/self/exe` is the arm binary, so `taskset` and any `git` child are
untouched. Inside the search it arms `ITIMER_PROF` with a requested
period of 1000 microseconds of process CPU time. The kernel delivers it
at its own tick, so the tick count is recorded rather than assumed.

Memory. On every tick, once at start, and once at exit, the search reads
`RssAnon` from its own `/proc/self/status` and keeps the maximum. That
maximum is `rss_anon_peak_kb`. `vm_hwm_kb` is `VmHWM` at exit.
`rss_reads` counts the reads.

Vector path. Before the run the harness lists the functions each arm's
`src/` marks `#[target_feature(enable = ...)]`, and the `anneal_core`
text symbols of each binary with those names, from `nm --demangle
--print-size`. Those symbols are anneal's vector kernels, the bodies the
`*_available()` checks dispatch to. On each tick the probe compares the
interrupted instruction pointer with those address ranges.
`vector_ticks` is the number of ticks that landed inside one,
`probe_ticks` all ticks, and `vector_ranges` the number of ranges. A
row with `vector_ticks` above 0 ran a vector kernel. The S build drops
the kernels it can never call, so its `vector_ranges` is expected to be
0. The rgpot potential kernels are not anneal symbols and are not
counted. The full symbol list goes to the log directory.

## Record

Rows are appended under `<cell>/rows.csv` and are not rewritten. A
finished `(cell, arm, seed)` is left in place. Columns:

```
cell,core,affinity,machine_id,binary_sha256,probe_sha256,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,rss_anon_peak_kb,vm_hwm_kb,rss_reads,probe_ticks,vector_ticks,vector_ranges,parity_digest
```

`core` is 0. `affinity` is the process CPU list, which must be `0`.
`machine_id` is the sha256 of the machine id, not a hostname.
`inside_s` is time inside the potential. `outside_s` is the rest of the
wall. `charged` is the number of charged evaluations. `best` is the
lowest energy as the binary prints it.

`parity_digest` is the sha256 of the run's stdout then stderr, line by
line, with the timing and checkout fields removed (`wall_s`,
`wall_seconds`, `pot_s`, `outside_s`, `pot_per_eval_s`,
`outside_per_eval_s`, `head`, `tree`, `cpu`, `pins`, as CSV columns,
`key=value` tokens, or `key:` lines), followed for the seed driver by the bytes of its trace
and xyz files. It covers the result row, the per-arm draw and accept
counts, the move counts, and the driver diagnostics.

`<cell>/summary.csv` is computed from those rows. The directory
`summary.csv` repeats every cell. Columns:

```
cell,n,budget,reference,hits_a,hits_b,hits_s,evals_median_a,evals_median_b,fisher_p,mannwhitney_p,outside_per_call_median_a,outside_per_call_median_b,outside_ratio_unpaired,outside_ratio,outside_ratio_lo,outside_ratio_hi,rss_peak_median_a_kb,rss_peak_median_b_kb,rss_ratio,rss_ratio_lo,rss_limit,aa_n,aa_outside_ratio,aa_outside_ratio_lo,aa_outside_ratio_hi,aa_rss_ratio,aa_rss_ratio_hi,s_outside_ratio,vector_ticks_a,vector_ticks_b,vector_ticks_s,vector_rows_b,vector_rows_s,vector_ranges_a,vector_ranges_b,vector_ranges_s,parity,parity_mismatches,verdict,reasons
```

Medians are Hyndman-Fan type 7. For n = 12 that is the mean of the 6th
and 7th ordered values. `outside_ratio_unpaired` is the previous gate's
statistic, the median of B over the median of A.

A paired ratio is taken per seed, B over A (A2 over A1 in the band, S
over B for `s_outside_ratio`). The interval for its median is the
distribution-free order-statistic interval: rank r is the largest with
P(Binomial(12, 1/2) <= r - 1) at most the tail. For n = 12 and a tail
of 0.025 that is r = 3, so the two-sided 95% interval is the 3rd and
10th ordered ratios, with coverage 0.9614. For a one-sided 95% bound
the tail is 0.05, again r = 3: the lower bound is the 3rd ordered
ratio and the upper bound the 10th, each with coverage 0.9807.

## Verdict

A cell with fewer than 12 A, B, or S rows, with any other seed set, or
with an unfinished A/A band, is `INCOMPLETE`.

Otherwise the cell is `FAIL` when any of these holds, and `PASS` when
none do. `reasons` names the failed rules.

- `fisher`. Exact two-sided Fisher on the A and B hit counts, and only
  when B has fewer hits than A. `FAIL` when that p is below 0.05. The
  p-value sums the tables with the observed margins whose probability
  is at most the observed table's probability.
- `mannwhitney`. One-sided Mann-Whitney on calls, alternative that B is
  worse. A hit scores `evals_to_hit`. A miss scores `budget + 1`, so a
  miss ranks worse than a hit that spent the whole budget. The statistic
  is the tie-corrected normal approximation with the continuity
  correction, on the upper tail. `FAIL` when p is below 0.05.
- `outside`. Outside time per call, `outside_s / charged`, paired by
  seed. `outside_ratio` is the median of the twelve B over A ratios and
  `outside_ratio_lo` and `outside_ratio_hi` its two-sided 95% interval.
  `FAIL` when the upper bound is above 1.10.
- `rss`. Peak `RssAnon`, paired by seed. `rss_ratio` is the median B
  over A ratio and `rss_ratio_lo` its one-sided 95% lower bound.
  `rss_limit` is the larger of 1.10 and the A/A upper bound, which is
  the largest one-sided 95% upper bound of the A2 over A1 median ratio
  across the four band cells. `FAIL` when the lower bound is above the
  limit.
- `parity`. S against B, seed by seed: `best`, `charged`,
  `evals_to_hit`, `hit`, and `parity_digest` must all be equal.
  `parity_mismatches` lists each failing seed with the fields that
  differ. Any mismatch is a `FAIL`. `best` is compared as printed, which
  is 8 decimals for the LJ, Gupta, Morse, and Sutton-Chen drivers and 10
  for the seed driver. The digest adds every count those drivers print.

The A/A band, the S over B outside ratio, and the vector counts are
reported and are not fail rules by themselves.

## Move diagnostics

The Sutton-Chen run binary prints per-move-class proposal, acceptance,
and improvement counts. The `sc96-80` records are stored by arm and seed
in `diag/sc96-80-moves.csv`.

The summary refuses a row whose core is not 0 or whose affinity is not
0, a row whose machine id is not one 64-digit hex digest shared by the
cell, more than one probe build, or an arm whose binary sha256 is not
constant. A1 must be the arm A binary. B must differ from A and S must
differ from B.

```
order
abs lj104 lj75 lj98 lj102 lj103
aa lj75 lj38 lj98 lj104
abs lj38 lj55 lj150 lj75-seed au55 m55r6 m55r14 m80r6 m80r14 sc126-38 sc126-55 sc126-80 sc108-38 sc108-55 sc108-80 sc96-38 sc96-55 sc96-80 tip4p8 tip4p15 tip4p20 tip4p21
```

```
cells
cell,family,binary,budget,reference
lj38,lj,h2h_gmin_proof,400000,-173.928427
lj55,lj,h2h_gmin_proof,1000000,-279.248470
lj75,lj,h2h_gmin_proof,4000000,-397.492331
lj98,lj,h2h_gmin_proof,4000000,-543.665361
lj102,lj,h2h_gmin_proof,4000000,-569.363652
lj103,lj,h2h_gmin_proof,4000000,-575.766131
lj104,lj,h2h_gmin_proof,4000000,-582.086642
lj150,lj,h2h_gmin_proof,4000000,-893.310258
lj75-seed,lj-seed,h2h_seed,4000000,-397.492331
au55,gupta,h2h_metal,1000000,-191.03085280
m55r6,morse,h2h_morse,1000000,-250.286609
m55r14,morse,h2h_morse,1000000,-220.646208
m80r6,morse,h2h_morse,4000000,-378.333471
m80r14,morse,h2h_morse,4000000,-340.811371
sc126-38,sc,h2h_sc,400000,-35419.9804
sc126-55,sc,h2h_sc,1000000,-52884.6806
sc126-80,sc,h2h_sc,1000000,-78414.6271
sc108-38,sc,h2h_sc,400000,-10117.2454
sc108-55,sc,h2h_sc,1000000,-14814.5225
sc108-80,sc,h2h_sc,1000000,-21819.9952
sc96-38,sc,h2h_sc,400000,-8917.7056
sc96-55,sc,h2h_sc,1000000,-13217.8963
sc96-80,sc,h2h_sc,1000000,-19560.8966
tip4p8,tip4p,h2h_seed,2000000,-305.5183221676
tip4p15,tip4p,h2h_seed,2000000,-628.3722525077
tip4p20,tip4p,h2h_seed,2000000,-872.9887544952
tip4p21,tip4p,h2h_seed,2000000,-916.7074302940
```
