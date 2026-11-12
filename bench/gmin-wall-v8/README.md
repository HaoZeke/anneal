# GMIN wall comparison, v8

Anneal `ec64b77b` (incumbent projection) against the same serial GMIN binary as v4–v7, in the session that followed the develop-versus-incumbent hit panel. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is within 0.001 of the Cambridge minimum. Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

The harness stops at the first hit, so `tts_s` is the time to solution and a miss is censored at the budget wall. Its type-7 median is the median wall. Outside microseconds are `(wall_s - pot_s) / charged`.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | Outside µs |
| --- | --- | --- | --- | --- | --- | --- |
| LJ98 | anneal | 5/6 | 53.546 | 53.546 | 3.177 | 24.92 |
| LJ98 | gmin | 0/6 | 170.115 | 170.115 | 3.177 |  |
| LJ104 | anneal | 6/6 | 86.016 | 86.016 | 2.228 | 26.52 |
| LJ104 | gmin | 2/6 | 191.649 | 191.649 | 2.228 |  |

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN.
