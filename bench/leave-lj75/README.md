Copyright (c) anneal developers

LJ75 trail leave against `Config::for_cluster`.
Seeds 2000-2011, budget 4,000,000, one core, hostname bench, one session.
Reference minimum -397.492331. Hit when the best is below the reference
plus 0.001, the tolerance `h2h_arm3` uses for this system.
This record is based on develop 98dcfb3106f191baf7381fecd5624426831094fe,
on top of the two LJ75 AGOX commits.
`base` is `Config::for_cluster`. `leave` is that configuration with
`trail_on_stall` set and nothing else. The Lennard-Jones, Morse and
reduced pins are the `h2h_arm3` pins.

| arm | hits | median best | gap to reference | median inside s | median outside s |
| --- | --- | --- | --- | --- | --- |
| base | 1/12 | -396.28224883 | 1.21008217 | 36.787541 | 18.839934 |
| leave | 2/12 | -396.28224883 | 1.21008217 | 36.428840 | 18.844169 |

Fisher exact test on the hit table, two-sided: p = 1.
Calls to a hit, with a miss censored at 4,000,000. Two-sided
Mann-Whitney, normal approximation, tie correction and continuity
correction: U = 67, p = 0.651291.
Medians are type 7.

The trail does not help enough to change the default. The extra hit is
not separated from the baseline, and the median best of both arms is the
icosahedral floor, 1.21008217 above the Marks minimum. Ten leave seeds
stay on that floor. The trail restarts from the screened entry of the
occupied basin. It does not cross the barrier between that funnel and the
Marks funnel. That barrier is a saddle of the Lennard-Jones potential.
Saddle owns that fix. LJ38, LJ55, LJ98, LJ104, LJ150, Au55 and M80 were
not run, and `trail_on_stall` stays off in `Config::for_cluster`.

`scripts/leave_lj75/summarize.py` reprints the table.
`scripts/leave_lj75/enter_bench.sh` appends one new seed and leaves an
existing row untouched.
