# Reproduce

We drew the nineteen figures with the committed generators.
Run these from paper/jctc.
The scripts check chemparseplot 1.11.0.
They also check xyzrender 0.3.8 and cairosvg 2.9.1.

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
