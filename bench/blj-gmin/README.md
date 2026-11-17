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
<!-- blj-gmin table: end -->
