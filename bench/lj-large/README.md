# GMIN wall comparison, LJ102, LJ150 and LJ200

These rows were measured on anneal `2d00f044` and are unchanged by the rebase onto `e2865427`. Anneal `2d00f044` against the timed GMIN binary. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ102 runs before LJ150, and LJ150 runs before LJ200. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls, the paper ceiling already used at LJ102 and at the other sizes from 75 atoms up. A hit is within 0.001 of the Cambridge minimum. Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

LJ75, LJ98 and LJ104 already have rows. This session does not repeat them.

The targets are the Cambridge Cluster Database values. LJ102 is −569.363652 and LJ150 is −893.310258, from the N=3–150 table. LJ200 is −1229.184776, from the N=151–309 table the database links. The harness stops at the first hit, so time to solution is the wall and a miss is censored at the budget wall. Inside µs is `1e6 * pot_s / charged`. Outside µs is `1e6 * (wall_s - pot_s) / charged`. Each table entry is the type-7 median of those per-seed rates. GMIN's `pot_s` is a timer around the Lennard-Jones call in the local build.

| Size | Arm | Hits | Median wall (s) | Median time to solution (s) | Ratio | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LJ102 | anneal | 3/6 | 111.822 | 111.822 | 1.147 | 19.52 | 22.15 |
| LJ102 | GMIN | 3/6 | 128.287 | 128.287 | 1.147 | 39.41 | 4.84 |
| LJ150 | anneal | 1/6 | 390.020 | 390.020 | 0.866 | 42.46 | 55.02 |
| LJ150 | GMIN | 4/6 | 337.591 | 337.591 | 0.866 | 85.89 | 6.93 |
| LJ200 | anneal | 0/6 | 609.626 | 609.626 | 1.071 | 74.64 | 77.76 |
| LJ200 | GMIN | 0/6 | 653.070 | 653.070 | 1.071 | 154.20 | 9.08 |

Anneal hits LJ102 seeds 2000, 2004 and 2005. GMIN hits LJ102 seeds 2002, 2004 and 2005. Anneal hits LJ150 seed 2003. GMIN hits LJ150 seeds 2000, 2002, 2003 and 2005. Neither arm hits LJ200.

GMIN is Wales revision `aac1548a8aec13b53ad12a8d6904625081685d2e`, built without OpenMP, MPI, or ALIGN.
