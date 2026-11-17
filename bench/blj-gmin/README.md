# anneal against GMIN on binary Lennard-Jones clusters

Both arms in one session on one host. anneal is `Config::recommended` with
two species labels and no diameters, through `h2h_blj`. GMIN is Wales
revision `aac1548a8aec13b53ad12a8d6904625081685d2e`.
`snakemake -s workflow/Snakefile blj_gmin`: seeds 2000-2011, the call
budgets of the matching sizes on the GMIN panel (`PAPER_BUDGETS` in
`scripts/blj_gmin/run_panel.py`), one process on core 0 (`taskset -c 0`,
one thread), `sudo unshare --uts` with hostname `bench`. For each system
the seeds run in order, anneal then GMIN on the same seed.

The clusters are the Doye-Meyer binary Lennard-Jones set: ε_AA = ε_AB =
ε_BB = 1, σ_AA = 1, σ_AB = (σ_AA + σ_BB) / 2, and σ_BB/σ_AA of 1.05, 1.15
and 1.30. N is 38 and 55 at each ratio, and 98 at 1.30. The potential is
`rgpot_pair::binary_lj` at the pinned rgpot revision, with no cutoff.
Species A is the first N_A atoms. The labels passed to the preset are 200
and 201, which are outside the covalent-radius table, so the container,
the neighbour cutoff and the move scales stay those of the one-component
preset. The unlike-pair exchange is the generic species swap.

A hit is a quenched energy at or below the header energy of the stored
Cambridge Cluster Database structure plus 0.001. A miss is entered at the
budget. Both arms start from the geometry `h2h_blj coords` writes for that
seed, and both count one call per energy and gradient. The reference
energies, compositions, source URLs and sha256 checksums are
`bench/blj-gmin/ccd/sources.csv`. The table at
`https://www-wales.ch.cam.ac.uk/~wales/CCD/jon/structures/BLJ/table.html`
links `1.15/55new` and `1.3/98new`; both return HTTP 404. For N=55 at 1.15
the reference is the downloadable points file, header -289.2931853058,
rather than the table's unpublished -289.358203. For N=98 at 1.30 the
reference is
`https://www-wales.ch.cam.ac.uk/~wales/CCD/NEW_BINARY_LJ_MINIMA/1.3/98`,
header -590.7874132760 with N_A = 41. The original points file, header
-588.7548700592 with N_A = 42, is stored beside it and is not the target.
The CCD writes the directory `1.3` for the ratio 1.30.

GMIN is one build (`scripts/blj_gmin/build_gmin.sh`) of the public tarball
with the evaluator cap of `bench/lj104-lattice/v2/h2h-gmin.patch` and a
timer around `BLJ_CLUST` only (`scripts/blj_gmin/blj-timer.patch`), without
ALIGN, MPI or OpenMP, linked against the reference BLAS and LAPACK. Its
input is `examples/h2h_gmin_lj.data` plus
`BLJCLUSTER_NOCUT <N_A> 1.0 1.0 <σ_AB> <σ_BB>` and `RANSEED <seed>`.

Wall is the search wall each code reports. Inside is `pot_s`, the time
inside the potential; outside is the rest of the wall. Medians and
interquartile ranges are Hyndman-Fan type 7. Fisher is the two-sided exact
test on hits. Mann-Whitney (MW) is two-sided, the normal approximation with
the tie and continuity corrections, on calls to the minimum and on wall.

<!-- blj-gmin prose: start -->
BLJ38 σ=1.05 hits on 1 of 12 against GMIN on 0 of 12.
BLJ55 σ=1.05 hits on 12 of 12 against GMIN on 0 of 12.
BLJ38 σ=1.15 hits on 12 of 12 against GMIN on 10 of 12.
BLJ55 σ=1.15 hits on 0 of 12 against GMIN on 1 of 12.
BLJ38 σ=1.30 hits on 12 of 12 against GMIN on 4 of 12.
BLJ55 σ=1.30 hits on 1 of 12 against GMIN on 7 of 12.
BLJ98 σ=1.30 hits on 0 of 12 against GMIN on 0 of 12.
This setting hits more seeds than GMIN on 4 of the 7 clusters.
GMIN hits more on BLJ55 σ=1.15, 1 of 12 against 0 of 12, and on BLJ55 σ=1.30, 7 of 12 against 1 of 12.
The two tie on BLJ98 σ=1.30.
Fisher's test on hits is below 0.05 on 3 of the 7 clusters: 2 for this setting and BLJ55 σ=1.30 for GMIN.
<!-- blj-gmin prose: end -->

- `rows.csv`: one row per run, appended and never rewritten.
  `code_revision` is the anneal commit on anneal rows and the GMIN SVNREV
  on GMIN rows; `harness_revision` is the anneal commit whose example
  wrote the starts; `host` is the UTS hostname and `session` the start of
  the runner process.
- `ccd/`: the Cambridge Cluster Database structures, and `ccd/sources.csv`
  with the URL and sha256 of each file.
- `gmin-source.txt`: the GMIN tarball, SVNREV, patches, CMake flags,
  compiler, linked BLAS and LAPACK and binary hash.
  `bin-sha256.txt`: the binaries the session ran.
- `summary.csv`, the table below and table `tab:blj` in
  `paper/jctc/manuscript.tex` are written from `rows.csv` by
  `bash scripts/blj_gmin/enter_bench.sh --summarize --tex`.

<!-- blj-gmin table: start -->
| System | Budget | Arm | Hits | Median calls | IQR | Wall (s) | Inside (s) | Outside (s) | Fisher p | MW p calls | MW p wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLJ38 σ=1.05 | 400000 | anneal | 1/12 | 400000 | 0 | 53.683 | 1.667 | 52.014 | 1 | 0.36 | 3.7e-5 |
|  |  | GMIN | 0/12 | 400000 | 0 | 2.337 | 1.442 | 0.893 |  |  |  |
| BLJ55 σ=1.05 | 1000000 | anneal | 12/12 | 5413 | 51.75 | 0.733 | 0.676 | 0.057 | 7.4e-7 | 1.0e-5 | 3.7e-5 |
|  |  | GMIN | 0/12 | 1000000 | 0 | 10.223 | 7.372 | 2.850 |  |  |  |
| BLJ38 σ=1.15 | 400000 | anneal | 12/12 | 47287.5 | 108.75 | 0.536 | 0.449 | 0.087 | 0.48 | 3.6e-5 | 3.5e-3 |
|  |  | GMIN | 10/12 | 219767.5 | 170116 | 1.266 | 0.780 | 0.486 |  |  |  |
| BLJ55 σ=1.15 | 1000000 | anneal | 0/12 | 1000000 | 0 | 121.9 | 8.829 | 113.1 | 1 | 0.36 | 3.7e-5 |
|  |  | GMIN | 1/12 | 1000000 | 0 | 10.460 | 7.543 | 2.916 |  |  |  |
| BLJ38 σ=1.30 | 400000 | anneal | 12/12 | 3202 | 122.25 | 0.408 | 0.336 | 0.072 | 1.3e-3 | 2.6e-5 | 3.7e-5 |
|  |  | GMIN | 4/12 | 400000 | 26012.5 | 2.556 | 1.551 | 1.005 |  |  |  |
| BLJ55 σ=1.30 | 1000000 | anneal | 1/12 | 1000000 | 0 | 85.630 | 9.250 | 76.212 | 0.027 | 0.028 | 4.8e-4 |
|  |  | GMIN | 7/12 | 712287 | 418274 | 7.660 | 5.500 | 2.160 |  |  |  |
| BLJ98 σ=1.30 | 4000000 | anneal | 0/12 | 4000000 | 0 | 772.4 | 114.4 | 655.9 | 1 | 1 | 3.7e-5 |
|  |  | GMIN | 0/12 | 4000000 | 0 | 114.8 | 95.806 | 19.079 |  |  |  |
<!-- blj-gmin table: end -->
