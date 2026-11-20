# Reproduce

We drew the 17 figures with the committed generators.
Run every command of this file from the repository root.
The scripts check chemparseplot 1.11.0.
They also check xyzrender 0.3.8, cairosvg 2.9.1 and pymupdf 1.28.2.
The dumps they read come from `HaoZeke/anneal-paper-data` at
`71185263eb6a8130d1262fd76d90109f5fabdcf5`; a script that needs one runs
`scripts/data/fetch_data.py paper/jctc/`, which downloads each file named
in `data/hf-files.csv` from that commit and checks each sha256.

## Building the papers

The manuscript and the Supporting Information are built apart, so each reads
the other's label numbers from `paper/jctc/labels-manuscript.tex` and
`paper/jctc/labels-si.tex`.  After a label moves or a float is added, rewrite
both files, build the manuscript, and measure the rendered text:

    python3 paper/jctc/scripts/cross_labels.py --output /tmp/jctc-labels
    (cd paper/jctc && latexmk -pdf -outdir=/tmp/jctc manuscript.tex)
    python3 paper/jctc/scripts/pdf_length.py --pdf /tmp/jctc/manuscript.pdf --check

`scripts/paper_audit.py` fails on a stale label file or a stale
`paper/jctc/logs/pdf-length.txt`; that receipt is tracked although `logs/` is
ignored, so it is staged with `git add -f`.

## Benchmark panels

The first three campaigns are
serial and their runners pin the process and children to CPU 0.

Sutton--Chen, all nine systems, both anneal and GMIN, seeds 2000--2011:

    SC_H2H_N12_OUTPUT=/work/metal-sc/output \
      snakemake -s workflow/Snakefile sc_h2h_n12 --cores 1 --force

The fresh results, seed rows, logs, and session receipt are written below
`/work/metal-sc/output`.  The committed tables are references only.

Rigid TIP4P against GMIN, (H2O)8, 15, 20, 21 and 25, both arms, seeds
2000--2011:

    TIP4P_H2H_OUTPUT=/work/tip4p-h2h/output \
      snakemake -s workflow/Snakefile tip4p_h2h --cores 1 --force

The fresh rows, summary, verification row, and session receipt are written
below `/work/tip4p-h2h/output`.

AGOX GOFEE against the paper arm on LJ38, LJ55 and LJ75, seeds 2000--2005:

    AGOX_GOFEE_OUTPUT=/work/ase-gofee/output \
      snakemake -s workflow/Snakefile agox_gofee_h2h --cores 1 --force

The fresh result slices and session receipt are written below
`/work/ase-gofee/output`.

The recommended CuH2 slab on the FCC fixture, budget 2500, seeds 0 through 7,
one process at a time.  The output logs and hardware receipt go below
`/work/cuh2-fcc-rec`; the receipt's host line is the output of `hostname`:

    CUH2_FCC_REC_OUTPUT=/work/cuh2-fcc-rec \
      snakemake -s workflow/Snakefile cuh2_fcc_rec --cores 1 --force

The rule runs `scripts/run_cuh2_fcc_rec.sh`, which changes to the repository
root itself.

Each runner appends one JSON object per invocation to its `sessions.jsonl`
beside the result rows.  The record includes `/proc/sys/kernel/random/boot_id`,
the effective `sched_getaffinity` CPU list, UTC session start, host, kernel,
CPU model, code revision and the exact Snakemake command above, the versions
of rustc, cargo, gfortran and cc (null when one is not installed), the Python
version, and the OpenMP libraries the dynamic loader resolves with
`OMP_NUM_THREADS` and every other `OMP_*`, `GOMP_*` and `KMP_*` setting.
The recorder runs under the campaign's effective affinity (CPU 0 for the serial
campaigns). Each `sessions.jsonl` object contains these fields for the
invocation and boot that it describes.

The TIP4P/GMIN table reads `bench/tip4p-h2h/rows-h2h-r2.csv` and
`bench/tip4p-h2h/wall-r2.csv`.  The adjacent host and session records capture
the one-machine, core-0 ABBA protocol.  The harness writes the 120 runs, 5
sizes of 12 seeds with both arms, with this command.  The anneal binary, the
GMIN executable and the start files are the ones the TIP4P driver
`scripts/tip4p_h2h/enter_h2h.sh` builds under `/work/tip4p-h2h`, and the hit
tolerance is 0.001 kJ/mol.  The output paths below are fresh; the command
refuses to start when any of them exists:

    python3 bench/tip4p-h2h/run_h2h_r2.py run \
      --binary /work/tip4p-h2h/bin/h2h_seed \
      --gmin /work/tip4p-h2h/bin/GMIN \
      --gmin-data examples/h2h_gmin_tip4p.data \
      --work-dir /work/tip4p-h2h-rerun \
      --start-dir /work/tip4p-h2h/runs \
      --reference-rows bench/tip4p-h2h/rows-h2h.csv \
      --rows-out /work/tip4p-h2h-rerun/rows-r2.csv \
      --sessions-out /work/tip4p-h2h-rerun/sessions-r2.jsonl \
      --hosts-out /work/tip4p-h2h-rerun/hosts-r2.csv \
      --anneal-sha e1a69a57a8dda1b44fdde9413ae31e3fc3af1889 \
      --gmin-revision aac1548a8aec13b53ad12a8d6904625081685d2e \
      --sizes 8,15,20,21,25 \
      --seeds 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011 \
      --budget 2000000 \
      --core 0 \
      --hit-tol 0.001

Compare the fresh calls and scored-size hit flags with the committed rows
while writing a fresh wall summary and exact rank-enumeration test:

    python3 bench/tip4p-h2h/run_h2h_r2.py report \
      --rows /work/tip4p-h2h-rerun/rows-r2.csv \
      --reference-rows bench/tip4p-h2h/rows-h2h-r2.csv \
      --check-out /work/tip4p-h2h-rerun/check-r2.md \
      --wall-out /work/tip4p-h2h-rerun/wall-r2.csv

`python3 scripts/data/fetch_data.py bench/tip4p-h2h/logs/` retrieves the
pinned paper-data logs.

The driver draws the 17 figures.
It also draws the summary-rates figure and the contents graphic.

    python3 paper/jctc/scripts/render_all.py /tmp/jctc-figs

Rows 1 and 2, the two schematics:

    python3 paper/jctc/scripts/fig_algorithm_stack.py /tmp/jctc-figs/algorithm_stack.pdf

    python3 paper/jctc/scripts/fig_rec_arms.py /tmp/jctc-figs/rec_arms.pdf

Rows 3 to 5:

    python3 paper/jctc/scripts/fig_lj75_ecdf.py /tmp/jctc-figs/lj75_ecdf.pdf

    python3 paper/jctc/scripts/fig_disconnectivity.py /tmp/jctc-figs/disconnectivity.pdf

    python3 paper/jctc/scripts/fig_landfold_maps.py /tmp/jctc-figs/landfold_maps.pdf

Rows 6 to 9:

    python3 paper/jctc/scripts/fig_landscapes.py /tmp/jctc-figs/landscapes.pdf

    python3 paper/jctc/scripts/fig_systems.py /tmp/jctc-figs/systems.pdf

    python3 paper/jctc/scripts/fig_si_structures.py /tmp/jctc-figs/si_structures.pdf

    python3 paper/jctc/scripts/fig_si_water.py /tmp/jctc-figs/si_water.pdf

Rows 10 to 17:

    python3 paper/jctc/scripts/fig_si_panels.py /tmp/jctc-figs

Source hashes of the vendored files are in the immutable
paper/jctc/FIGURE-SOURCE.tsv ledger and its append-only updates file.
The paper audit checks every effective row against the current file.
The nested checksum files match the sealed files and the effective
figure-source rows.
The paths follow.

```
paper/jctc/data/lj75_gmin/audit/SHA256SUMS
paper/jctc/data/lj75_gmin/audit/gmin-lj75/SHA256SUMS
paper/jctc/data/lj75_gmin/audit/lj75-derived/SHA256SUMS
paper/jctc/data/lj75_gmin/audit/lj75-recommended/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj38-derived/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj38-recommended/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj55-derived/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj55-recommended/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj98-derived/SHA256SUMS
paper/jctc/data/lj_csm_sizes/audit/lj98-recommended/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/gmin_lj75/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/lj_cell/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/water_36086e98/budget-fair/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/water_36086e98/crest/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/water_36086e98/crest-polished/SHA256SUMS
paper/jctc/data/results_jcc_c47d301/water_36086e98/default/SHA256SUMS
```
