# (H2O)6 at GFN2, anneal against CREST on served calls

`scripts/crest_equal/run_panel.py` runs anneal at the 4000-call ledger and
CREST 3.0.2 at `-mquick -len 0.90` and `-len 0.11` on seeds 2000 to 2011, one
core, host `bench`. `rows.csv` and `summary.csv` hold the result;
`call-counts.csv` holds the per-row counts of later runs.

The seed picks the start geometry, which the three arms share. CREST is not
given a seed of its own, so its dynamics are not paired with anneal's.

## Claim

Anneal charges 3021 calls and hits 1 of 12. CREST at 0.90 ps has median 3055 calls and hits 4 of 12. CREST at 0.11 ps has median 916.5 calls and hits 0 of 12.

## What the rows at 723b7c23 show

- The CREST rows are charged the calls the driver served. CREST's own
  `Total number of energy+grad calls` line was not read and no CREST log was
  kept, so these rows cannot show that CREST spent no more than it was
  charged. The driver was started with a ceiling of 50000 calls, so the 4000
  in the `budget` column of the CREST rows is a label, not a cap.
- The most any CREST row was charged is 3531, so a cap of 4000 would not have
  changed a row.
- The 0.90 ps arm was charged 2510 to 3531 calls, 0.83 to 1.17 of anneal's
  3021; 5 of 12 seeds lie within 5 percent of it. The 0.11 ps arm was charged
  812 to 1168 calls; 0 of 12 lie within 5 percent. It is the shortest length
  that finished, not an equal-call arm.
- Median wall is 11.41 s for anneal, 69.54 s at 0.90 ps and 20.31 s at
  0.11 ps. The 0.11 ps arm is not matched to anneal's wall either.
- Anneal seed 2001 polished without converging, so it counts as a miss with
  no gap.

## From the next run

The driver is served with the 4000-call budget, so CREST stops being served
after its 4000th call. Each CREST row is charged the larger of the served
count and CREST's own energy+grad total. `call-counts.csv` records both, the
stop reason and the sha256 of `crest.out`, and `logs/` keeps the gzipped log.
`python3 scripts/crest_equal/run_panel.py --claim` prints the claim above from
`summary.csv`.
