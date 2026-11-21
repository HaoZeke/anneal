Copyright (c) anneal developers

LJ98 and LJ104 against GMIN, seeds 2000-2005, written by
`scripts/gmin_h2h/run_campaign.py`. The budget unit is one combined energy
and gradient pass. A miss counts as the full 4e6-pass ceiling in the median
and the interquartile range. Each file holds one summary row per system and
arm; the per-seed rows of these sessions were not committed.

- `results.csv`: a8406b41, with the pentagonal retry. `anneal_head`
  422628e8 is not in this repository.
- `results-05e9ef4d.csv`: 0bff90a6, before the pentagonal retry,
  `anneal_head` 05e9ef4d. a8406b41 overwrote these rows in `results.csv`;
  this file is that blob unchanged. Table `tab:gmin` of
  `paper/jctc/manuscript.tex` reports these rows.

The GMIN arm's per-seed rows (same GMIN revision, seeds and ceiling) are in
`../gmin-wall-v4/results.csv`, and its LJ104 median and interquartile range
follow from them. Neither file has a `host` column.

## Rows written by the next campaign

Every new per-seed and summary row keeps `host`, whose value remains the
compatibility UTS name `bench`. Each row adds:

- `machine_id`: lowercase hexadecimal SHA-256 of the public salt
  `anneal-bench-machine-id-v1` followed by a NUL byte and the stripped
  contents of `/etc/machine-id`. If that file is absent or empty, the
  stripped DMI `/sys/class/dmi/id/product_uuid` is used instead.
- `cpu_model`: the first CPU model string from `/proc/cpuinfo`, normalized
  to single spaces.

The patched GMIN per-seed row records two non-overlapping counters:

- `gradient_passes`: entries to `POTENTIAL` with `GRADT=true`. `charged`,
  `evals_to_hit`, and `budget` use this unit, matching one anneal combined
  energy and gradient pass.
- `energy_only_entries`: entries to `POTENTIAL` with `GRADT=false`. These
  are reported but do not consume the gradient-pass budget.

This correction keeps the pinned numeric budget at 4,000,000 but increases
GMIN's effective cap on total `POTENTIAL` entries: energy-only entries no
longer consume it. It does not change any existing pinned result or row.
The files above are provisional pre-correction results and are intentionally
not rewritten; the corrected panels require a new campaign.
