# Outside the Lennard-Jones potential

Anneal `3da4b358` against a GMIN binary with a timer around the Lennard-Jones call. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ75 runs before LJ98, and LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is within 0.001 of the Cambridge minimum. Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

The harness stops at the first hit, so time to solution is the wall and a miss is censored at the budget wall. Inside µs is `1e6 * pot_s / charged`. Outside µs is `1e6 * (wall_s - pot_s) / charged`. Each table entry is the type-7 median of those per-seed rates. Every anneal evaluation count matches the v9 `4fdc37e8` arm. The GMIN rows of that panel stay in `bench/gmin-wall-v9`; the timed rerun is appended there and repeated here.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | anneal | 6/6 | 25.467 | 25.467 | 3.978 | 10.61 | 15.98 |
| LJ75 | GMIN | 0/6 | 101.314 | 101.314 | 3.978 | 21.59 | 3.74 |
| LJ98 | anneal | 5/6 | 59.080 | 59.080 | 2.810 | 18.12 | 23.19 |
| LJ98 | GMIN | 0/6 | 166.003 | 166.003 | 2.810 | 36.86 | 4.65 |
| LJ104 | anneal | 6/6 | 82.246 | 82.246 | 2.250 | 20.45 | 24.61 |
| LJ104 | GMIN | 2/6 | 185.040 | 185.040 | 2.250 | 41.61 | 4.93 |

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN. The timer is a pair of `SYSTEM_CLOCK` calls around `LJ` in the local build.

## Per evaluation at LJ104, seed 2000, 150_000 calls

`cpu-clock` self time, turned into microseconds per charged call. Before is `1ba7417e`. After is this cut. The uninstrumented inside and outside times are the potential timer, not the sample share.

| | Before | After |
| --- | --- | --- |
| Inside µs | 20.63 | 20.19 |
| Outside µs | 18.49 | 15.74 |
| Nearest-image scan | 2.83 | 0.98 |
| Assignment column scan | 2.29 | 2.26 |
| Orbit completion | 1.43 | 1.43 |
| Sorted-pair sort | 1.34 | 1.16 |

The scan keeps the running minimum in registers. A repeated distance spectrum is reused. A strict unique row-minimum matching skips the shortest-path solve; on this prefix that skip did not move the column-scan time.
