# anneal against GMIN on the paper panel

Both arms of each cluster on one host. The anneal rows are the stored
campaign; GMIN was rerun. anneal develop
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
`scripts/morse_h2h/morse-timer.patch`, the binary Gupta timer of
`bench/metal-cluster-h2h/gmin-bgupta.patch`, and
`scripts/gmin_panel/symmetrise-jp.patch`. ALIGN and MPI stay off. OpenMP
is on, because the GMIN build supports it; each run still sets one
thread. The link is OpenBLAS. `symmetrise-jp.patch` passes the run index
into `TESTSYMOP`. On this SVNREV the three live calls in `symmetry.f90`
omit it, and `TESTSYMOP` then indexes `ATMASS` with an undefined run,
which segfaults as soon as `SYMMETRISE` is used. The patch changes no
tolerance and no step.

The earlier Au55 rows of session `20261117T035457Z` ran the alloy block of
`scripts/metal_h2h/enter_gold.sh` (`RADIUS 50.0`, `TEMPERATURE 1.0`,
`STEP 0.40 0.51` and `NEWRESTART 40` without `AVOID ... F`), which left
`AVOIDRESEEDT` true, reseeded into a 57.7 Å cube after 40 stagnant steps,
and hit 0/12; those rows stay in `rows.csv`, and session `20261117T173638Z`
reran GMIN under the Gupta input above (6/12 against 0/12).

The fair rerun, driver `GMIN fair`, takes one published block per size
from `scripts/gmin_panel/inputs.py`, plus `RANSEED <seed>`. The stored
rows with driver `GMIN` are the earlier tutorial-file runs and are not
scored. Lennard-Jones uses the OptBench or Oakley block for that size.
Morse adds `MORSE <rho>` to the Lennard-Jones block of the matching
range. Au55 keeps the reviewed Gupta block and is not retuned: the Darby
gold row (A 0.2061, p 10.229, q 4.036, ξ 1.790, r0 2.884), GMIN's
`io1.f90` nearest-neighbour container (`RADIUS 11.829521`),
`STEP 1.027741 0.51` and `AVOID 10.277408 20 F` converted from `σ` by
`r0 / 2^(1/6)`, and `TEMPERATURE 0.1` eV from Wales's
`examples/GMIN/MGupta.tar.bz2`. The `F` keeps the taboo list and does
not reseed. Sources for each block are in the docstring of `inputs.py`.

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
| LJ38 | 400000 | anneal | 12/12 | 10615.5 | 3591.25 | 0.075 | 0.029 | 0.047 | 1 | 0.11 | 0.51 |
|  |  | GMIN | 12/12 | 3388.5 | 9862.25 | 0.037 | 0.020 | 0.018 |  |  |  |
| LJ55 | 1000000 | anneal | 12/12 | 12242 | 22988.5 | 0.142 | 0.062 | 0.082 | 1 | 0.40 | 0.93 |
|  |  | GMIN | 12/12 | 8070.5 | 11203.75 | 0.121 | 0.097 | 0.027 |  |  |  |
| LJ75 | 4000000 | anneal | 12/12 | 1409229.5 | 983622 | 27.019 | 12.543 | 13.818 | 1 | 1.2e-4 | 2.9e-3 |
|  |  | GMIN | 12/12 | 70388 | 177511.5 | 3.672 | 1.577 | 2.059 |  |  |  |
| LJ98 | 4000000 | anneal | 10/12 | 1511030.5 | 1985194.5 | 41.586 | 23.825 | 17.949 | 0.48 | 2.9e-3 | 0.026 |
|  |  | GMIN | 12/12 | 370350 | 529297.75 | 16.378 | 14.740 | 1.638 |  |  |  |
| LJ102 | 4000000 | anneal | 9/12 | 894801 | 3401143.75 | 27.445 | 15.463 | 11.982 | 1 | 0.93 | - |
|  |  | GMIN | 8/12 | 2764598 | 3641355.25 | 127.0 | 115.0 | 12.022 |  |  |  |
| LJ103 | 4000000 | anneal | 8/12 | 2293133 | 2608443.75 | 66.188 | 39.673 | 26.763 | 1 | 0.66 | - |
|  |  | GMIN | 7/12 | 1929800.5 | 3208932.25 | 91.998 | 83.636 | 8.396 |  |  |  |
| LJ104 | 4000000 | anneal | 9/12 | 2328723 | 2947098 | 69.306 | 39.241 | 30.064 | 0.21 | 0.063 | - |
|  |  | GMIN | 5/12 | 4000000 | 1326273.25 | 192.0 | 174.4 | 17.507 |  |  |  |
| LJ150 | 4000000 | anneal | 5/12 | 4000000 | 1477851 | 237.2 | 139.7 | 96.075 | 4.6e-3 | 5.1e-4 | 5.1e-4 |
|  |  | GMIN | 12/12 | 123671.5 | 103274.5 | 11.353 | 10.761 | 0.592 |  |  |  |
| Au55 | 1000000 | anneal | 6/12 | 852736 | 572700.75 | 31.522 | 23.474 | 7.920 | 0.014 | 7.1e-3 | - |
|  |  | GMIN | 0/12 | 1000000 | 0 | 60.540 | 58.077 | 2.463 |  |  |  |
| M55 ρ=6 | 1000000 | anneal | 12/12 | 563 | 128.75 | 0.010 | 0.008 | 0.001 | 1 | 3.7e-5 | 3.7e-5 |
|  |  | GMIN | 12/12 | 24190 | 25035.25 | 0.469 | 0.391 | 0.079 |  |  |  |
| M55 ρ=14 | 1000000 | anneal | 7/12 | 730604 | 623707.5 | 13.593 | 9.672 | 3.921 | 4.6e-3 | 2.9e-3 | - |
|  |  | GMIN | 0/12 | 1000000 | 0 | 18.875 | 15.784 | 3.075 |  |  |  |
| M80 ρ=6 | 4000000 | anneal | 2/12 | 4000000 | 0 | 168.9 | 111.3 | 57.363 | 0.012 | 3.4e-3 | - |
|  |  | GMIN | 9/12 | 1378958 | 2895956.25 | 68.281 | 45.984 | 22.297 |  |  |  |
| M80 ρ=14 | 4000000 | anneal | 9/12 | 1338932.5 | 3064673 | 49.869 | 38.033 | 12.262 | 3.4e-4 | 3.8e-4 | - |
|  |  | GMIN | 0/12 | 4000000 | 0 | 166.8 | 134.1 | 32.494 |  |  |  |
<!-- gmin-panel table: end -->
