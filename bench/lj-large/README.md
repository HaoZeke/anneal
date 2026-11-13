# GMIN wall comparison, LJ102, LJ150 and LJ200

These rows were measured on anneal `2d00f044` and are unchanged by the rebases onto `e2865427` and `b3d03061`. Anneal `2d00f044` against the timed GMIN binary. The driver is `scripts/gmin_h2h/enter_bench.sh`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, `taskset -c 0`. LJ102 runs before LJ150, and LJ150 runs before LJ200. Inside a size the six seeds and two arms are shuffled with schedule seed 20261111. The budget is 4_000_000 force calls, the paper ceiling already used at LJ102 and at the other sizes from 75 atoms up. A hit is within 0.001 of the Cambridge minimum. Wall medians are Hyndman–Fan type 7. The ratio is the GMIN median wall divided by the anneal median wall.

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

## LJ150 outside time on `b3d03061`

`b3d03061` packs the long-distance suffix of each stored basin and rejects a far centre before loading its descriptor. The same-session prefixes below are LJ150 seed 2001, the miss whose best energy is already the 4_000_000-call best. Outside µs is the mean of the two 80_000-call repetitions, and the single 400_000-call run. The three binaries share one release profile. `e2865427` is the parent before the suffix pack. `b3d03061` is that pack plus the bench commit, without the orbit-merge change. `64887159` adds the orbit-merge root skip.

| Build | Calls | Wall (s) | Potential (s) | Outside µs |
| --- | --- | --- | --- | --- |
| `e2865427` | 80_000 | 6.408, 6.358 | 3.518, 3.476 | 36.08 |
| `b3d03061` | 80_000 | 6.251, 6.183 | 3.381, 3.352 | 35.63 |
| `64887159` | 80_000 | 6.112, 6.094 | 3.346, 3.343 | 34.48 |
| `e2865427` | 400_000 | 35.820 | 17.262 | 46.39 |
| `b3d03061` | 400_000 | 34.980 | 16.677 | 45.76 |
| `64887159` | 400_000 | 34.398 | 16.642 | 44.39 |

The suffix pack moves this prefix by about half a microsecond. The orbit-merge skip takes off about one microsecond more. LJ75 seed 2002 stays at 11_228 evaluations on both `b3d03061` and `64887159`, LJ98 seed 2002 stays at 612_380, and the LJ98 seed 2000 trajectory hash is unchanged. On the pre-suffix orbit commit the eighteen LJ75, LJ98 and LJ104 seeds 2000–2005 kept the recorded evaluation counts, including the LJ98 seed 2004 miss at 4_000_000.

The 80_000-call profile on `e2865427` puts 54% of the samples in the Lennard-Jones potential. The largest outside piece is the dense assignment scan, about 8 µs, then symmetrise, about 6 µs, then the sorted-pair sort, about 3 µs. A nearest-column assignment certified 1_750 of 1_870 LJ150 solves and returned the same permutation, and that run was slower, so it is not on this branch.

## Where the LJ150 misses stop

Anneal is 1/6 and GMIN is 4/6 at the same 4_000_000 calls. The five anneal misses are stored with `charged` equal to the budget. Seed 2001 reaches −882.69459141 at call 12_755 and that is the full-run best. Seed 2002 reaches −883.49689268 at call 642_612. Seed 2004 reaches −883.01890748 at call 2_851_323. The calls after those marks do not lower the energy. Seed 2003 hits the C3v minimum −893.310258 at call 43_488.

The stalled geometries have no icosahedral 555 fraction under a common-neighbour analysis at cutoff 1.35. Seed 2001 is decahedral, with a 422 fraction of 0.41. Seeds 2002 and 2004 are close-packed, with 421 fractions of 0.45 and 0.53. Those basins sit about 10 above the C3v minimum. GMIN reaches that minimum from seeds 2000, 2002, 2003 and 2005 inside the same call budget. The outside cost lengthens the wall. It does not take evaluations away from the search.
