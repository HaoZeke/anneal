# Stock recommended GMIN panel completion

This directory completes the paper's Lennard-Jones and Morse GMIN
head-to-head panel. The anneal arm is stock `Config::recommended` at
revision `61fa67cf380e46876b569da45e7cc091647a775b`, with the structure
templates already removed.

The first thirteen cells and their 156 anneal rows are reused from
`bench/g0-lj-stock/rows.csv`. They are not rerun. The ten added Morse
cells use seeds 2000 through 2011:

| Systems | Budget per seed |
| --- | ---: |
| m30r6, m30r14 | 400000 |
| m38r6, m38r14 | 400000 |
| m70r3, m70r6, m70r10, m70r14 | 1000000 |
| m80r3, m80r10 | 4000000 |

The references and hit tolerance are those in
`scripts/morse_gmin/run_panel.py` and
`scripts/morse_panel/run_panel.py`. The comparison arm is not rerun.
Its rows come first from `bench/morse-gmin/seeds.csv`, then from
`bench/morse-panel/seeds.csv` when the first file has no matching row.
The source of every comparison row is recorded in `rows.csv`.

## Commands

Run these commands from the repository root:

```sh
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 MKL_DOMAIN_NUM_THREADS=1 RAYON_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export POLARS_MAX_THREADS=1 CARGO_BUILD_JOBS=1
rustup run 1.88.0 cargo build --release --locked -j 1 \
  --example h2h_morse
nohup python3 bench/g1-develop-stock/run_panel.py \
  > campaign.log 2>&1 &
run_pid=$!
while kill -0 "$run_pid" 2>/dev/null; do sleep 60; done
wait "$run_pid"
python3 bench/g1-develop-stock/summarize.py
```

`run_panel.py` launches one process at a time through `taskset -c 0`.
It passes only a minimal environment, fixes all listed thread controls
at one, and does not use a quick mode. Each completed row is synced
before the next run starts.

`summarize.py` reads the two source scripts directly and refuses a
changed system size or budget. It also refuses any cell without exactly
twelve anneal rows. A miss remains a miss and ranks at positive infinity
for the median and Mann-Whitney calculation.

`rows.csv` contains the ten added cells paired seed by seed with their
stored comparison rows. `panel-rows.csv` contains the thirteen reused
cells followed by the ten added cells. `summary.csv` reports the
two-sided Fisher exact test, the tie- and continuity-corrected
Mann-Whitney test, type-7 call medians, and the median anneal outside
time per charged call.

`gmin_hits` counts only rows with code `gmin` and code revision
`aac1548a8aec13b53ad12a8d6904625081685d2e` in
`bench/gmin-panel/rows.csv`, using the latest stored row of each seed.
The hit is re-derived from that row's best energy against the reference
in `bench/g0-lj-stock/manifest.csv` plus the shared hit tolerance, the
same rule used for the anneal arm. `gmin_revision` records that
revision. A system with no such row has an empty `gmin_hits`, and the
Fisher and Mann-Whitney columns are empty with it.
