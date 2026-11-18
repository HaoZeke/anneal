# Reproduce

We drew the nineteen figures with the committed generators.
Run these from paper/jctc.
The scripts check chemparseplot 1.11.0.
They also check xyzrender 0.3.8 and cairosvg 2.9.1.

## Benchmark panels

Run these commands from the repository root.  The first three campaigns are
serial and their runners pin the process and children to CPU 0.  The
water-hexamer campaign uses eight cores, as recorded by its pinned workflow.

Sutton--Chen, all nine systems, both anneal and GMIN, seeds 2000--2011:

    snakemake -s workflow/Snakefile sc_h2h_n12 --cores 1 --force

Rigid TIP4P against GMIN, (H2O)8, 15, 20, 21 and 25, both arms, seeds
2000--2011:

    snakemake -s workflow/Snakefile tip4p_h2h --cores 1 --force

AGOX GOFEE against the paper arm on LJ38, LJ55 and LJ75, seeds 2000--2005:

    snakemake -s workflow/Snakefile agox_gofee_h2h --cores 1 --force

The full water-hexamer campaign (default anneal and plain CREST, seeds
0--191, 4000 calls) is owned by the separate `HaoZeke/anneal_repro`
repository.  Check out revision
`56160f2d100fc2c9f7365f4ee6d1d86653c8b31f` (tree
`149899211dbec73aa88877cb7af0ebdf9b99f39e`) at
`/work/anneal_repro`, then run:

    snakemake -s workflow/Snakefile water_crest_fair --cores 8 --force --config anneal_repro=/work/anneal_repro

The wrapper verifies that revision and tree, then runs this exact command in
the pinned checkout:

    pixi run --locked --frozen snakemake water_crest_fair --cores 8 --force

Each runner appends one JSON object per invocation to its `sessions.jsonl`
beside the result rows.  The record includes `/proc/sys/kernel/random/boot_id`,
the effective `sched_getaffinity` CPU list, UTC session start, host, kernel,
CPU model, code revision and the exact Snakemake command above.  The recorder
runs under the campaign's effective affinity: CPU 0 for the three serial
campaigns and the scheduler-provided mask for the water campaign.  Its
`--cores 8` setting caps Snakemake concurrency but does not invent a CPU
affinity mask.  These fields were not captured by the historical sessions
and cannot be recovered after a boot.

The TIP4P/GMIN per-seed logs are not in this repository or in any pushed
branch fetched on 2026-11-18.  The committed files contain summaries and some
later per-seed row tables, but not the logs of the twelve-seed head-to-head;
they are not reconstructed here.

The driver draws the nineteen figures.
It also draws the summary-rates figure and the contents graphic.

    python3 scripts/render_all.py

Rows 1 and 2, the two schematics:

    python3 scripts/fig_algorithm_stack.py figs/algorithm_stack.pdf

    python3 scripts/fig_rec_arms.py figs/rec_arms.pdf

Rows 3 to 6:

    python3 scripts/fig_lj75_ecdf.py figs/lj75_ecdf.pdf

    python3 scripts/fig_water_crest.py figs/water_crest.pdf

    python3 scripts/fig_disconnectivity.py figs/disconnectivity.pdf

    python3 scripts/fig_landfold_maps.py figs/landfold_maps.pdf

Rows 7 to 10:

    python3 scripts/fig_landscapes.py figs/landscapes.pdf

    python3 scripts/fig_systems.py figs/systems.pdf

    python3 scripts/fig_si_structures.py figs/si_structures.pdf

    python3 scripts/fig_si_water.py figs/si_water.pdf

Rows 11 to 18:

    python3 scripts/fig_si_panels.py figs

Source hashes of the vendored files are in FIGURE-SOURCE.tsv.
The kit is finish-anneal-figures-v1.
Its branch is paper/finish-2026-11-04.
Its tip follows.

094e8a997c3b6d040a83a24c1827ce71937beff5

Seventeen nested checksum files have no line in the kit checksum list.
They still match the sealed files and the rows in FIGURE-SOURCE.tsv.
The paths follow.

```
data/lj75_gmin/audit/SHA256SUMS
data/lj75_gmin/audit/gmin-lj75/SHA256SUMS
data/lj75_gmin/audit/lj75-derived/SHA256SUMS
data/lj75_gmin/audit/lj75-recommended/SHA256SUMS
data/lj_csm_sizes/audit/SHA256SUMS
data/lj_csm_sizes/audit/lj38-derived/SHA256SUMS
data/lj_csm_sizes/audit/lj38-recommended/SHA256SUMS
data/lj_csm_sizes/audit/lj55-derived/SHA256SUMS
data/lj_csm_sizes/audit/lj55-recommended/SHA256SUMS
data/lj_csm_sizes/audit/lj98-derived/SHA256SUMS
data/lj_csm_sizes/audit/lj98-recommended/SHA256SUMS
data/results_jcc_c47d301/gmin_lj75/SHA256SUMS
data/results_jcc_c47d301/lj_cell/SHA256SUMS
data/results_jcc_c47d301/water_36086e98/budget-fair/SHA256SUMS
data/results_jcc_c47d301/water_36086e98/crest/SHA256SUMS
data/results_jcc_c47d301/water_36086e98/crest-polished/SHA256SUMS
data/results_jcc_c47d301/water_36086e98/default/SHA256SUMS
```

The copper slab command is `scripts/run_cuh2_fcc_rec.sh`.
The budget is 2500.
The seeds are 0 through 7.
The host line is the output of `hostname`.
