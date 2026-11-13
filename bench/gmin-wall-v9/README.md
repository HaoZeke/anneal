# GMIN wall comparison, v9

Anneal `4fdc37e8` against the same serial GMIN binary as v4–v8. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ75 runs before LJ98, and LJ98 runs before LJ104. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls. A hit is within 0.001 of the Cambridge minimum. Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

The session first recorded GMIN and an anneal arm at `09dd4c2f`. Seed 2002 of LJ75, LJ98 and LJ104 at `1ba7417e` repeats the `4fdc37e8` evaluation count and quenched energy (11228, 612380, 635431), so the ratios use `4fdc37e8`. That arm ran next on the same core. The `09dd4c2f` rows stay in the session, and all eighteen `4fdc37e8` evaluation counts match them.

The harness stops at the first hit, so `tts_s` is the time to solution and a miss is censored at the budget wall. Its type-7 median is the median wall. Inside µs is `1e6 * pot_s / charged`. Outside µs is `1e6 * (wall_s - pot_s) / charged`. Each table entry is the type-7 median of those per-seed rates.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ75 | anneal | 6/6 | 26.067 | 26.067 | 4.037 | 10.63 | 16.50 |
| LJ75 | gmin | 0/6 | 105.232 | 105.232 | 4.037 | 21.59 | 3.74 |
| LJ98 | anneal | 5/6 | 61.440 | 61.440 | 2.789 | 18.10 | 24.94 |
| LJ98 | gmin | 0/6 | 171.330 | 171.330 | 2.789 | 36.86 | 4.65 |
| LJ104 | anneal | 6/6 | 86.623 | 86.623 | 2.197 | 20.52 | 26.47 |
| LJ104 | gmin | 2/6 | 190.279 | 190.279 | 2.197 | 41.61 | 4.93 |

Anneal leads on wall time at all three sizes. The search is the `4fdc37e8` chain.

The GMIN inside and outside columns are a later session. A timer around the Lennard-Jones call in the local GMIN build fills `pot_s`, and those rows are appended after the original 36. The median walls of that rerun are 101.314, 166.002 and 185.040 s.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN.
