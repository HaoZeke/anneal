# NH3 octamer matched-budget result

## Four-hour stopping outcome

The projected campaign duration exceeded the four-hour run cap. Two seeds
finished, and seed 2002 was stopped while in progress and is not included as
a result. No process was left running.

The two completed searches took 5632.148167 and 4914.563635 seconds. Their
mean projects to about 17.6 hours for twelve seeds at the observed rate,
before build time. Both completed seeds reached the target only on their last
charged call.

| seed | hit | calls to hit | stop reason | best energy / Eh | wall / s |
| ---: | :---: | ---: | :--- | ---: | ---: |
| 2000 | yes | 332675 | target | -35.470770945094 | 5632.148167 |
| 2001 | yes | 332675 | target | -35.470770945156 | 4914.563635 |

For the completed subset, 2 / 2 seeds hit and the median calls-to-hit is
332675. Neither hit occurred within 134305 calls, so the observed fraction at
or before the minimum recorded CREST call count is 0 / 2.

These partial rows do not support the requested twelve-seed Wilson interval,
bootstrap interval, Fisher exact test, or Mann--Whitney test. Those statistics
are deliberately not reported as if the interrupted subset were the planned
sample.

## Independent CREST sample

The CREST rows are seeds 17000--17005 from
`bench/nh3-octamer/results.csv`. CREST was not rerun. All six recorded rows
are hits. Their call counts are 243868, 221579, 332675, 241500, 134305, and
196621, with median 231539.5 calls. The common anneal budget, 332675 calls,
is their maximum.

The anneal and CREST seed sets differ and are independent samples. CREST's
values are full recorded search-call totals, while anneal stops at the first
target hit.
