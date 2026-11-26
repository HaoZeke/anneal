# Outside-time trial

Negative record. Three bit-identical edits of the default Lennard-Jones
search were slower on every measured size, so the edits are not in
this tree.

Base is develop `4feb0699d4e61c9531283e82196fcb60baf66932`. The search
is `lj_cluster_search` with feature `bank-rpc` and no fourth argument,
which is `Config::for_cluster`. Release build, one process, one core
(`taskset -c 1`, `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`). Host is
an 8-core Intel Xeon, kernel 6.12.113, perf 6.8.12. Hardware counters
are unavailable here, so the profile uses the software `cpu-clock`
event.

## Profile

`perf record -e cpu-clock -F 1999 -g --call-graph fp` on budget
400000, one seed, seeds 2000, 2001, 2002 and 2003, sizes 38, 55, 75
and 98. Self samples, 16 runs summed:

| bucket | samples | share of remainder |
| --- | ---: | ---: |
| potential `lj` | 117130 | 84.94% of all 137890 |
| rgmin | 9504 | 45.78% |
| pair-spectrum sort | 3889 | 18.73% |
| libc allocation and copy | 2252 | 10.85% |
| finite-gradient check | 1145 | 5.52% |
| ndarray | 1037 | 5.00% |
| bias, other than the sort | 840 | 4.05% |
| shortest pair | 617 | 2.97% |
| other anneal | 431 | 2.08% |
| other | 407 | 1.96% |
| neighbour table | 354 | 1.71% |
| surface move | 284 | 1.37% |

Remainder is 20760 samples (15.06%). By size the remainder is 31.16%
(LJ38), 22.00% (LJ55), 14.10% (LJ75) and 10.07% (LJ98). The sort's
share of that remainder rises from 9.99% on LJ38 to 22.89% on LJ75
and 22.22% on LJ98. The largest single remainder symbols are the
rgmin AVX-512 dot (2453), `Lbfgs::gnorm` (2249), the stable
`sort_by(total_cmp)` quicksort (1750) and its small-sort (1538).

## Edits that were timed

1. `SortedPairs::describe` sorted with `sort_unstable_by` and
   `f64::total_cmp`. Equal `total_cmp` keys are the same bits, so the
   vector matches a stable sort.
2. A one-atom move rewrote the sorted multiset by removing the old
   lengths and merging the new ones, instead of sorting the full list.
3. `min_pair_distance` and `NeighborTable::build_naive` gained an AVX2
   loop. Products were `mul` then `add`, not a fused multiply-add, in
   the scalar association `(dx*dx + dy*dy) + dz*dz`. The shortest pair
   kept the scalar `min` order. Neighbour indices were pushed in the
   scalar loop order.

Pinned tests `recommended_lj98_seed_2000_keeps_its_short_trajectory`
and the four flag-off symmetry pins passed on the edited library.
Default-search stdout matched between the two arms on every timed run.

## Outside time

Both arms wrapped `lj` with the same clock. The wrapper is not in
this tree. Outside time per call is `(wall_ns - pot_ns) / calls`.
Every run reported `calls=400001`. Seed 2000, budget 400000. Six
ABBA cycles on each size give 12 runs per arm. The ratio is the mean
of arm B over the mean of arm A. The interval is a 10000-draw
percentile bootstrap of that ratio (2.5 and 97.5, linear rank),
`random.Random(20261126)`, each arm resampled on its own.

Nanoseconds per call:

| cell | mean A | mean B | B/A | 95% interval |
| --- | ---: | ---: | ---: | --- |
| LJ38 | 1283.734 | 1304.026 | 1.015807 | [0.990851, 1.042139] |
| LJ55 | 1629.047 | 1645.484 | 1.010090 | [0.989653, 1.033237] |
| LJ75 | 1695.862 | 1758.564 | 1.036974 | [1.021665, 1.053422] |
| LJ98 | 2048.178 | 2108.506 | 1.029454 | [1.006056, 1.054986] |

Every upper bound is above 1. LJ75 and LJ98 sit above 1 on the whole
interval. This is not a land candidate.

A probe on the edited library, budget 200000, seed 2000, timed both
sorts on each fresh spectrum and counted one-atom edits. The edit
path ran zero times. Stable sort was faster than unstable sort:
LJ38 26.873 ms versus 28.348 ms over 2287 sorts, LJ75 66.423 ms
versus 71.384 ms over 1357 sorts, LJ98 89.276 ms versus 95.463 ms
over 1052 sorts. The real spectra are ordered enough that the stable
sort wins, which is the opposite of a random-input microbench. The
AVX2 scans also did their arithmetic through small stack buffers, so
the shortest-pair and neighbour loops picked up setup that the scalar
loops do not pay.
