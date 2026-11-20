# Template-free morph escape, r4

Arm A is develop `61fa67cf380e46876b569da45e7cc091647a775b`, stock `Config::recommended`. Arm R4 is `5d0e02d11f922d4e60d54b004187e589978d4fd8`, the gated morph escape of `2ca323e72e4bdbb7f888244789c7f6ffeefe6164` with the observed-order candidate grown in cell lists and its growth reused while the chain stays on one structure. Both arms were built with Rust 1.88 and `--features rgpot-ex` and run by `run_pair.py`: one run at a time, `taskset -c 0`, every thread pool at one, seeds 2000 to 2011, `PAPER_BUDGETS`. Within a seed the arm order alternates with seed parity. A hit is a quenched energy within 0.001 of the reference minimum. Medians are Hyndman–Fan type 7, with a miss counted as infinite calls. Outside time is `wall_s - pot_s`. The outside cost per call for a cell is the type-7 median of `outside_s / charged` over its twelve runs, and the ratio is R4 over A. These are development numbers from one host, not the benchmark gate.

## Identity

The candidate code changes how the trial is built, not which trial is built. The library tests compare the binned growth, the bonded neighbours and the bucketed peel against copies of the all-pairs code, bit for bit, and a reused growth against a fresh one under the same generator state. In the campaign, all 168 rows match the earlier gate rows (R4 against arm B, A against arm A) on `calls_to_hit`, `charged`, `best_bits` and `hit`. The trajectories are therefore those of arm B, and the hit counts below are arm B's.

## Per cell

| Cell | Hits A | Hits R4 | Median calls A | Median calls R4 | Outside/call R4/A | Gate B/A |
| --- | --- | --- | --- | --- | --- | --- |
| m55r6 | 12/12 | 12/12 | 22596 | 22596 | 1.013 | 1.005 |
| lj38 | 12/12 | 12/12 | 127839.5 | 126456 | 0.991 | 0.989 |
| m55r14 | 9/12 | 8/12 | 451561.5 | 506548 | 1.027 | 1.031 |
| lj98 | 7/12 | 11/12 | 3166811.5 | 2558250.5 | 1.131 | 1.203 |
| lj102 | 6/12 | 7/12 | miss | 2551083.5 | 1.018 | 1.097 |
| m80r14 | 6/12 | 11/12 | miss | 2486485 | 1.098 | 1.237 |
| lj75 | 5/12 | 7/12 | miss | 3596765 | 1.067 | 1.071 |

The gate column is the same statistic over the earlier gate rows, which come from a different session.

Losses: lj98 is still above the 1.10 bar at 1.131. m55r14 loses one hit, 9/12 to 8/12, as arm B did, and its median calls rise. m80r14 is now under the bar at 1.098, by a small margin.

## Where the lj98 and m80r14 residual goes

On one observed proposal the old builder took 1113 µs at 80 points and 1628 µs at 98. The cell-list builder takes 550 µs and 703 µs. At seed 2000 on m80r14 the lattice share of non-potential samples fell from 386 to about 158 per million calls.

With `perf record -e cpu-clock` on m80r14 seed 2003, A had 1481 non-potential samples per million calls and R4 had 1852. Of the 371 extra, `BasinIndex::lookup` and its `packed_squared_distance_exceeds` account for 358. Lattice construction accounts for 88 and the twin for 14. The lookup walks every stored basin centre, newest first, so its cost per call grows with the number of basins stored. Arm B finds more distinct basins and runs longer before it hits or runs out of budget, and each call then pays a longer walk. On lj98 seed 2009 the ratio was 1.194 overall and 1.131 without lattice and twin samples. The rest of the excess is the same search outcome, not the morph construction. Gating the activations further would lower it only by proposing fewer escapes, and those escapes are where the lj98 and m80r14 hits come from. This round does not change the basin index.

## Template-free targets

The baseline is arm A above: lj75 5/12, lj38 median 127839.5 calls, m55r6 median 22596. The template-assisted rows at `c2de652d` were lj75 12/12, lj38 10616 and m55r6 563. R4 gives lj75 7/12, lj38 126456 and m55r6 22596. That is two lj75 hits and about 1 percent on lj38, and nothing on m55r6. The targets are not recovered.

`escape_reach.csv` comes from `2ca323e7` with `probe.diff`, which logs every twin and observed proposal and its quenched energy. Mode `forced` latches the escape as soon as it is ready (`PROBE_FORCE`). Mode `natural` leaves the latch alone.

| Cell | Mode | Hits | Median calls | Twin draws | Observed draws | Draws quenched to the minimum |
| --- | --- | --- | --- | --- | --- | --- |
| lj38 | natural | 12/12 | 126456 | 1508 | 405 | 0 |
| lj38 | forced | 11/12 | 108767 | 3929 | 1213 | 0 |
| m55r6 | natural | 12/12 | 22596 | 223 | 278 | 0 |
| m55r6 | forced | 12/12 | 27104.5 | 361 | 295 | 0 |
| lj75 | forced | 5/12 | miss | 44110 | 19159 | 0 |

The mechanism that fails is construction. None of the 71,481 proposals, on any cell or mode, quenched to the global minimum. The twin reflects the incumbent through a plane, and the observed arm regrows the order it measures in the incumbent. Both only reproduce the structure the chain is already on, so neither can build the truncated octahedron at lj38, the Marks decahedron at lj75, or the Mackay icosahedron at m55. The template era reached them by building them: the Mackay cap was relaxed at the start, and the named fcc and icosahedral growths were proposed directly. The latch is a second, smaller cause. On lj38 the first morph proposal arrives at a median of about 120,000 calls, after most hits. Latching earlier does not fix this. It trades m55r6 calls for lj38 calls and loses an lj38 hit.
