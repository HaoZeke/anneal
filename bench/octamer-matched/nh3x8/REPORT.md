# NH3 octamer matched-budget result

Twelve anneal seeds, 2000-2011, are in `results.csv`. CREST was not rerun.
The hit comparison uses seeds 17000-17005 from `bench/nh3-octamer/results.csv`.

Anneal hits 10/12. Wilson 95% CI [0.551969, 0.953035].
Median calls_to_hit, with a miss stored as 332676: 332675.0.
Bootstrap 95% CI of that median, 10000 resamples, seed 20261123, linear quantiles: [332675.0, 332675.0].

CREST hits 6/6. The two-sided Fisher exact test on the hit table
[[10, 2], [6, 0]]
gives odds ratio 0 and p = 0.529412.

Every anneal hit has calls_to_hit 332675 and stop_reason target.
That number is `energy_calls` from the run: the ledger total after the hop
and the final polish. Saved run.txt files for seeds 2002-2011 record
search_calls 331075 and polish_calls 1600.
Seeds 2000 and 2001 were recorded by the same harness and have no saved
run.txt in this tree. `README.md` names the lines that define the count.

The CREST calls column holds full-run totals. This summary does not compare
those totals with anneal calls_to_hit. A calls comparison needs CREST
first-hit counts, which are not recorded for NH3.

## Anneal rows

| seed | hit | calls to hit | stop reason | best energy / Eh | wall / s |
| ---: | :---: | ---: | :--- | ---: | ---: |
| 2000 | yes | 332675 | target | -35.470770945094 | 5632.148167 |
| 2001 | yes | 332675 | target | -35.470770945156 | 4914.563635 |
| 2002 | yes | 332675 | target | -35.470770943337 | 4810.354501 |
| 2003 | yes | 332675 | target | -35.470770944723 | 4518.948883 |
| 2004 | yes | 332675 | target | -35.470770943668 | 4618.598030 |
| 2005 | yes | 332675 | target | -35.470770942940 | 5239.392020 |
| 2006 | yes | 332675 | target | -35.470770942951 | 4672.991289 |
| 2007 | yes | 332675 | target | -35.470770945123 | 4436.101296 |
| 2008 | yes | 332675 | target | -35.470770945068 | 4760.432801 |
| 2009 | no | 332676 | budget | -35.470735870808 | 4493.842425 |
| 2010 | no | 332676 | budget | -35.470735862359 | 4768.606529 |
| 2011 | yes | 332675 | target | -35.470770945483 | 4504.249444 |
