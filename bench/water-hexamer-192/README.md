# Water hexamer, 192 seeds, paper budget rebuild

This campaign rebuilds the paper water panel from committed inputs at the
develop base `bc16d35c`.

## Inputs committed here

- `inputs/water-v10.xyz`: regenerated from the committed isolated-water
  template in `src/catalog/molecular.rs` (`reference_coordinates` for six
  molecules). The private file used in the earlier run was not available
  byte-for-byte.
- `inputs/isomers/{prism,cage,book}.xyz`: reordered O-H-H references derived
  from committed paper files in `paper/jctc/data/hq_sci_h2o6/`.
- `inputs/config.yaml`: panel seed, arm, and threading settings.
- `inputs/benchmarks.yaml`: paper budget and comparison file references.

## Source rows and comparison baseline

- Old 192-seed rows: `paper/jctc/data/water_v45b/water-v45b-perseed.csv`
- Old 36086 summaries: `paper/jctc/data/results_jcc_c47d301/water_36086e98/`

The inputs differ from the 36086 run: `water-v10.xyz` was regenerated and the
isomer references were reordered, so the hit-rate shifts in
`compare-vs-36086.csv` mix input changes with code changes and are not a
code-only comparison.

`bench/unreachable-revisions.csv` currently has no water row for `36086e98`,
so there is no superseded water entry to remove there.

## Runner layout

- Snakemake rule: `water_hexamer_192`
- Driver: `scripts/water_hexamer_192/enter_bench.sh`
- Campaign script: `scripts/water_hexamer_192/run_campaign.py`
- asv suite: `benchmarks/water_hexamer_192.py`

Rows are append-only and include `code_revision` and `host` on every row.
`summary.csv` and `compare-vs-36086.csv` are rewritten from the current
revision rows.
