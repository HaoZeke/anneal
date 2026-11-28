# (H2O)6 at GFN2, anneal against CREST on served calls

`rows.csv` and `summary.csv` are the (H2O)6 panel of anneal and CREST 3.0.2
at `-mquick -len 0.90` and `-len 0.11`, seeds 2000 to 2011, one core, host
`bench`. On these rows the driver was started with a ceiling of 50000 calls.
`scripts/crest_equal/run_panel.py` is a later runner and is not how these
rows stopped.

The seed picks the start geometry, which the three arms share. CREST is not
given a seed of its own, so its dynamics are not paired with anneal's.

## What the rows at 723b7c23 show

- Revision `723b7c23` contains `acf5f0aa`, which flipped the GFN2 gradient sign
  before `94a37da7`, and the CREST wrapper negated the served gradient.
  The paper does not cite the committed hit counts of these rows.
- The CREST rows are charged the calls the driver served. CREST's own
  `Total number of energy+grad calls` line was not read and no CREST log was
  kept, so these rows cannot show that CREST spent no more than it was
  charged. The driver was started with a ceiling of 50000 calls, so the 4000
  in the `budget` column of the CREST rows is a label, not a cap.
- The most any CREST row was charged is 3531, so a cap of 4000 would not have
  changed a row.
- The 0.90 ps arm was charged 2510 to 3531 calls. The 0.11 ps arm was
  charged 812 to 1168 calls. It is the shortest length that finished.
  Its length is not the 0.90 ps length.
- Median wall is 11.41 s for anneal, 69.54 s at 0.90 ps and 20.31 s at
  0.11 ps. The 0.11 ps arm is not matched to anneal's wall either.
- Anneal seed 2001 polished without converging, so it counts as a miss with
  no gap.

## Recorded call accounting

On these rows the 4000 in the budget column is a label, and the driver
ceiling was 50000 calls. No CREST row is charged 4000 calls. A later run of
`scripts/crest_equal/run_panel.py` can refuse a request after 4000 served
calls. That refusal is not how these rows stopped.
