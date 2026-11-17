# anneal against GMIN on the paper panel

Both arms of each cluster in one session on one host: anneal develop
`c2de652dca0bc75e2b0cb4afffac91fd656404e4`, built with Rust 1.88.0,
`--release --locked`, and GMIN at Wales revision
`aac1548a8aec13b53ad12a8d6904625081685d2e`.
`snakemake -s workflow/Snakefile gmin_panel`: seeds 2000-2011, the paper
call budgets (`PAPER_BUDGETS` in `scripts/gmin_panel/run_panel.py`), one
process on core 0 (`taskset -c 0`, one thread, every other process moved
to cores 1-7), `sudo unshare --uts` with hostname `bench`. For each
system the seeds run in order, anneal then GMIN on the same seed.

A hit is a quenched energy at or below the Cambridge Cluster Database
energy plus 0.001: the Lennard-Jones table, pure gold at 55 in the Cu-Au
Gupta archive, and the Doye-Wales Morse table. A miss is entered at the
budget. Both arms start from the geometry the anneal example writes with
`coords` for that seed, and both count one call per energy and gradient.
anneal runs `Config::recommended` through `h2h_gmin_proof` (Lennard-Jones),
`h2h_metal` (Au55, with the gold species label and `basin_stride = 8`) and
`h2h_morse` (Morse, which quenches one Mackay cap before the walk when the
fcc second neighbour lies below -0.05, as at ρ=6, and leaves ρ=14
unchanged). At `c2de652d` (before `503470a0`), `h2h_gmin_proof` has no
LJ103 entry, so LJ103 runs on a second build of the same revision with
`bench/paper-0120/lj103-table.patch`, which adds the energy -575.766131
and the 4e6 budget and nothing else; those rows have driver
`h2h_gmin_proof+lj103`.

GMIN is one build (`scripts/gmin_panel/build_gmin.sh`) of the public
tarball with the evaluator cap and Lennard-Jones timer of
`bench/lj104-lattice/v2/h2h-gmin.patch`, the Morse timer of
`scripts/morse_h2h/morse-timer.patch` and the binary Gupta timer of
`bench/metal-cluster-h2h/gmin-bgupta.patch`, without ALIGN, MPI or OpenMP,
linked against the reference BLAS and LAPACK. Its inputs are
`examples/h2h_gmin_lj.data` (from the Wales group LJ38 introduction),
`examples/h2h_gmin_morse.data` plus `MORSE <rho>`, and for Au55 the same
quench and taboo block on `BGUPTAT` with the Darby gold row (A 0.2061,
p 10.229, q 4.036, ξ 1.790, r0 2.884), GMIN's `io1.f90` nearest-neighbour
container radius (`RADIUS 11.829521`), `STEP 1.027741 0.51` and
`AVOID 10.277408 20 F` converted from `σ` by `r0 / 2^(1/6)`, and
`TEMPERATURE 0.1` eV from Wales's `examples/GMIN/MGupta.tar.bz2`, each
with `RANSEED <seed>`.

The earlier Au55 rows of session `20261117T035457Z` ran the alloy block of
`scripts/metal_h2h/enter_gold.sh` (`RADIUS 50.0`, `TEMPERATURE 1.0`,
`STEP 0.40 0.51` and `NEWRESTART 40` without `AVOID ... F`), which left
`AVOIDRESEEDT` true, reseeded into a 57.7 Å cube after 40 stagnant steps,
and hit 0/12; those rows stay in `rows.csv`, and session `20261117T181140Z`
reran both arms of Au55 under the Gupta input above (4/12 against 1/12).

Wall is the search wall each code reports. Inside is `pot_s`, the time
inside the potential; outside is the rest of the wall. Medians and
interquartile ranges are Hyndman-Fan type 7. Fisher is the two-sided exact
test on hits. Mann-Whitney (MW) is two-sided, the normal approximation with
the tie and continuity corrections, on calls to the minimum (misses ranked
above every hit) and on wall to the minimum where call-cap censoring leaves
every cross-arm pair order observed (`-` where misses censor wall time at
unequal horizons).

During session `20261117T035457Z` the virtual machine was suspended three
times: the `utc` stamps of three rows lie 476 s, 2057 s and 2068 s after
the row before plus their own wall. On LJ75 GMIN seed 2007 (wall 120.8 s,
outside 17.5 s) and LJ98 anneal seed 2005 (wall 39.7 s, outside 17.8 s),
both inside and outside rates stayed within the other seeds of the same
system and arm. On LJ98 GMIN seed 2000 (wall 194.9 s), `pot_s` stayed at
161.9 s while `outside_s` rose to 33.0 s against 19.4-24.8 s on the other
eleven seeds, and LJ98 GMIN seed 2005 right after the third resume took
225.3 s (inside 200.5 s); because both LJ98 arms carry misses, `MW p wall`
is already omitted as censored, and `hit` and `evals_to_hit` are unaffected.

- `rows.csv`: one row per run, appended and never rewritten; `--summarize`
  scores the latest stored row of each seed. `code_revision` is the anneal
  commit on anneal rows and the GMIN SVNREV on GMIN rows;
  `harness_revision` is the anneal commit whose examples wrote the starts;
  `host` is the UTS hostname and `session` the start of the runner process.
- `gmin-source.txt`: the GMIN tarball, SVNREV, patches, CMake flags,
  compiler, linked BLAS and LAPACK and binary hash.
  `bin-sha256.txt`: the five binaries (`4f772314...` for GMIN in session
  `20261117T035457Z`, `4665badc...` in `20261117T152941Z`; the four Rust
  binaries are bit-identical across both sessions).
- `summary.csv`, the table below and table `tab:gmin` in
  `paper/jctc/manuscript.tex` are written from `rows.csv` by
  `bash scripts/gmin_panel/enter_bench.sh --summarize --tex`.

<!-- gmin-panel table: start -->
| System | Budget | Arm | Hits | Median calls | IQR | Wall (s) | Inside (s) | Outside (s) | Fisher p | MW p calls | MW p wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ38 | 400000 | anneal | 12/12 | 10615.5 | 3591.25 | 0.075 | 0.029 | 0.047 | 0.037 | 3.4e-5 | 3.4e-5 |
|  |  | GMIN | 7/12 | 389399 | 173127.75 | 3.658 | 2.673 | 0.979 |  |  |  |
| LJ55 | 1000000 | anneal | 12/12 | 12242 | 22988.5 | 0.142 | 0.062 | 0.082 | 1 | 0.75 | 0.31 |
|  |  | GMIN | 12/12 | 12764.5 | 12435.75 | 0.236 | 0.191 | 0.045 |  |  |  |
| LJ75 | 4000000 | anneal | 12/12 | 1409229.5 | 983622 | 27.019 | 12.543 | 13.818 | 7.4e-7 | 1.0e-5 | 1.0e-5 |
|  |  | GMIN | 0/12 | 4000000 | 0 | 119.9 | 102.9 | 16.952 |  |  |  |
| LJ98 | 4000000 | anneal | 10/12 | 1511030.5 | 1985194.5 | 41.586 | 23.825 | 17.949 | 6.7e-5 | 1.3e-4 | - |
|  |  | GMIN | 0/12 | 4000000 | 0 | 182.5 | 161.9 | 20.311 |  |  |  |
| LJ102 | 4000000 | anneal | 9/12 | 894801 | 3401143.75 | 27.445 | 15.463 | 11.982 | 0.21 | 0.093 | - |
|  |  | GMIN | 5/12 | 4000000 | 2463305.5 | 194.4 | 173.8 | 20.614 |  |  |  |
| LJ103 | 4000000 | anneal | 8/12 | 2293133 | 2608443.75 | 66.188 | 39.673 | 26.763 | 0.10 | 0.047 | - |
|  |  | GMIN | 3/12 | 4000000 | 239453.25 | 204.9 | 183.1 | 21.660 |  |  |  |
| LJ104 | 4000000 | anneal | 9/12 | 2328723 | 2947098 | 69.306 | 39.241 | 30.064 | 0.40 | 0.16 | - |
|  |  | GMIN | 6/12 | 3696118 | 1285122.25 | 193.6 | 173.6 | 20.007 |  |  |  |
| LJ150 | 4000000 | anneal | 5/12 | 4000000 | 1477851 | 237.2 | 139.7 | 96.075 | 0.21 | 0.17 | - |
|  |  | GMIN | 9/12 | 3129518.5 | 3240050.5 | 326.1 | 302.1 | 23.955 |  |  |  |
| Au55 | 1000000 | anneal | 4/12 | 1000000 | 736743.75 | 38.043 | 28.083 | 9.966 | 0.32 | 0.21 | - |
|  |  | GMIN | 1/12 | 1000000 | 0 | 59.573 | 56.427 | 3.152 |  |  |  |
| M55 ρ=6 | 1000000 | anneal | 12/12 | 92049.5 | 103998.75 | 1.742 | 1.196 | 0.547 | 1 | 0.069 | 0.078 |
|  |  | GMIN | 12/12 | 37047.5 | 37358.75 | 0.751 | 0.637 | 0.115 |  |  |  |
| M55 ρ=14 | 1000000 | anneal | 7/12 | 730604 | 623707.5 | 12.814 | 9.196 | 3.618 | 4.6e-3 | 2.9e-3 | - |
|  |  | GMIN | 0/12 | 1000000 | 0 | 18.898 | 16.322 | 2.577 |  |  |  |
| M80 ρ=6 | 4000000 | anneal | 2/12 | 4000000 | 0 | 161.4 | 107.8 | 53.778 | 6.7e-5 | 3.1e-5 | 3.1e-5 |
|  |  | GMIN | 12/12 | 268934 | 268091.5 | 10.691 | 9.582 | 1.109 |  |  |  |
| M80 ρ=14 | 4000000 | anneal | 9/12 | 1338932.5 | 3064673 | 49.325 | 37.783 | 11.542 | 3.4e-4 | 3.8e-4 | - |
|  |  | GMIN | 0/12 | 4000000 | 0 | 152.7 | 138.3 | 14.430 |  |  |  |
<!-- gmin-panel table: end -->
