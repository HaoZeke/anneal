Copyright (c) anneal developers

LJ98 and LJ104 against GMIN, seeds 2000-2005, a ceiling of 4e6 force
calls, written by `scripts/gmin_h2h/run_campaign.py`. A miss counts as the
full ceiling in the median and the interquartile range. Each file holds
one summary row per system and arm; the per-seed rows of these sessions
were not committed.

- `results.csv`: a8406b41, with the pentagonal retry. `anneal_head`
  422628e8 is not in this repository.
- `results-05e9ef4d.csv`: 0bff90a6, before the pentagonal retry,
  `anneal_head` 05e9ef4d. a8406b41 overwrote these rows in `results.csv`;
  this file is that blob unchanged. Table `tab:gmin` of
  `paper/jctc/manuscript.tex` reports these rows.

The GMIN arm's per-seed rows (same GMIN revision, seeds and ceiling) are in
`../gmin-wall-v4/results.csv`, and its LJ104 median and interquartile range
follow from them. Neither file has a `host` column.
