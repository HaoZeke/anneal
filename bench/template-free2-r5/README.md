# Template-free morph escape, r5

Three arms, all built with Rust 1.88 and `--features rgpot-ex`:

- **A** is develop `61fa67cf380e46876b569da45e7cc091647a775b`, stock `Config::recommended`.
- **S1** is `795737db3e80a7f00d1b05a47f9532c7ffa38d98`, the r4 tip `8eee13f8` with the block-sum bound on the basin lookup (`a230fb74`) and one sine and cosine per symmetry rotation (`795737db`).
- **S2** is `0e3fe244706ad94fb284b7cd557d9b8e0867f7fd`, S1 with the drawn-site morphology arm (`bdc0b2ee`, `0e3fe244`).

`run_arms.py` runs them one at a time with `taskset -c 0` and every thread pool at one, on seeds 2000 to 2011 and `PAPER_BUDGETS`. Within a seed the arm order rotates with `seed % 3`. A hit is `energy_hits` against the reference minimum. Medians are Hyndman–Fan type 7, with a miss counted as infinite calls. Outside time is `wall_s - pot_s`. The outside cost per call for a cell is the type-7 median of `outside_s / charged` over its twelve runs, and the ratios are over A. Every row of `rows.csv` names the commit its binary was built from, the binary's sha256 and the host label. These are development numbers from one host, not the benchmark gate.

## Pauses

The host paused this machine twice while the campaign ran. In both cases a run was in flight, as the gaps in the progress log show: `lj98,A,2002` and `lj98,S2,2008`. Both runs were repeated on core 0 after the campaign by the same runner, and the repeats are in `reruns.csv`. They reproduce `calls_to_hit`, `charged` and `best_bits` exactly. Their outside times are lower, 17.70 s against 19.74 s and 41.74 s against 51.44 s, so the paused rows are inflated. `rows.csv` keeps the original rows. The lj98 ratios are given both ways below.

## Part 1: the basin lookup

`BasinIndex::lookup` returns the newest stored centre within the merge radius. Before r5 it walked every stored centre past a norm key and a suffix test, and computed a bounded distance for each survivor. Descriptors are 3160 entries long at m80 and 4753 at lj98. At m80r14 seed 2003 a lookup saw about 1800 centres, and later 3000 or more. The key kept about a third of them. Exact repeated queries were 2 to 3 percent, so memoising lookups would not help.

linkcell, the stack's cell-list crate (0.3.11), indexes 3-D points in a periodic box. It offers pair lists and self k-nearest neighbours within that box. A cell grid has no use at a dimension in the thousands, so patching linkcell with cross queries and open boundaries would still not index these descriptors. The route taken instead is a generic lower bound inside `BasinIndex`. It applies to any metric that declares itself Euclidean through the new `BasinMetric::is_euclidean`. Each stored centre keeps its sums over 32-entry blocks. The sum over blocks of the squared block-sum difference, divided by the block length, bounds the squared distance from below. Each block term is shrunk by the most the rounding of the two sums can have moved it. A centre is skipped only when the bound clears the radius by more than the slack the packed full distance already allows. The walk is still linear in the number of centres. What changes is that a skipped centre costs a few dozen block sums instead of the full descriptor.

The lookup is not the whole of the lj98 excess. A profile at lj98 seed 2009 put the lookup under 1.3 percent of samples. The libm sine and cosine taken per rotated point in symmetry detection were about 250 samples per million calls on develop. `795737db` takes them once per axis and order.

Identity: the tests in `src/bias/distance_bound_tests.rs` compare lookups and registrations against a verbatim copy of the old walk, check that every skipped centre lies outside the radius, and cover non-finite entries and a metric that is not Euclidean. The symmetrise test compares the deviation against a copy of the per-point version. In this campaign, all 84 S1 rows match the r4 record's R4 rows, and all 84 A rows match its A rows, on `calls_to_hit`, `charged`, `best_bits` and `hit`. That is 168 rows with 0 differences.

| Cell | Hits A | Hits S1 | Median calls A | Median calls S1 | Outside/call A (µs) | S1 (µs) | S1/A | r4 R4/A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m55r6 | 12/12 | 12/12 | 22596 | 22596 | 4.972 | 4.864 | 0.978 | 1.013 |
| lj38 | 12/12 | 12/12 | 127839.5 | 126456 | 2.470 | 2.425 | 0.981 | 0.991 |
| m55r14 | 9/12 | 8/12 | 451561.5 | 506548 | 4.342 | 4.280 | 0.986 | 1.027 |
| lj98 | 7/12 | 11/12 | 3166811.5 | 2558250.5 | 9.690 | 9.979 | 1.030 | 1.131 |
| lj102 | 6/12 | 7/12 | miss | 2551083.5 | 10.196 | 9.787 | 0.960 | 1.018 |
| m80r14 | 6/12 | 11/12 | miss | 2486485 | 9.361 | 9.444 | 1.009 | 1.098 |
| lj75 | 5/12 | 7/12 | miss | 3596765 | 6.110 | 6.001 | 0.982 | 1.067 |

The µs columns and the lj98 ratio use the repeated `lj98,A,2002` row. With the paused row as recorded in `rows.csv`, lj98 S1/A is 1.003. All seven cells are at or below 1.10 either way. Hits and calls are those of r4, because the trajectories are r4's. That includes r4's m55r14 loss against A, 9/12 to 8/12.

## Part 2: a drawn-site growth

The probe behind this arm was run at `8eee13f8` with `reach.diff`. The diff dumps the incumbent each time the morphology arms latch, as `latch-NNN.xyz`, and adds a scratch `reach` example. That example applies a generator 50 times to each dumped incumbent and quenches each candidate with `WarmLbfgs` for at most 5000 iterations. A hit is within 0.001 of the reference. The incumbents come from seeds 2000 to 2011 on lj38 (7 latches), lj75 (19) and m55r6 (1). `reach.csv` holds one row per incumbent and generator.

| Generator | lj38 hits of 350 | lj75 hits of 950 | m55r6 hits of 50 | Best lj75 |
| --- | --- | --- | --- | --- |
| observed, the best-coordinated shell | 0 | 0 | 0 | -394.413 |
| twin | 0 | 0 | 0 | -396.239 |
| shell of a point drawn from the better-coordinated half | 28 | 0 | 0 | -394.413 |
| the same shell with its inversion added | 28 | 0 | 0 | -394.413 |
| random cluster | 0 | 0 | 0 | -391.769 |

The observed shell of the best-coordinated point rebuilds the incumbent's own order, which is r4's finding. The shell of another well-coordinated point can carry a different local order. From two of the seven lj38 incumbents it grows into the truncated octahedron, in 18 and 10 of 50 draws. No generator reached the lj75 or m55r6 minima.

S2 adds that generator as a fourth arm of the floored Thompson gate: `Source::Site`, `drawn_site`, and a growth cache held per source. It uses no named motif, template, reference structure or size constant.

| Cell | Hits A | Hits S1 | Hits S2 | Median calls S1 | Median calls S2 | S2/A outside |
| --- | --- | --- | --- | --- | --- | --- |
| m55r6 | 12/12 | 12/12 | 12/12 | 22596 | 22596 | 0.969 |
| lj38 | 12/12 | 12/12 | 12/12 | 126456 | 117024.5 | 0.946 |
| m55r14 | 9/12 | 8/12 | 6/12 | 506548 | miss | 1.221 |
| lj98 | 7/12 | 11/12 | 9/12 | 2558250.5 | 3013411.5 | 1.057 |
| lj102 | 6/12 | 7/12 | 9/12 | 2551083.5 | 2653836 | 1.017 |
| m80r14 | 6/12 | 11/12 | 6/12 | 2486485 | miss | 1.076 |
| lj75 | 5/12 | 7/12 | 7/12 | 3596765 | 3533959 | 1.045 |

The S2 lj98 ratio uses the repeated `lj98,S2,2008` row. With the paused row it is 1.029.

This is a negative result. Against S1, the arm lowers the lj38 median by 7.5 percent and gains two lj102 hits. It leaves m55r6 unchanged and lj75 at 7/12, on a different set of seeds. It loses hits in three cells: m55r14 by 2, lj98 by 2 and m80r14 by 5. Against A it still loses three m55r14 hits. It fails the seven-cell no-loss bar. The lj75 target is not met, and the m55r6 median does not move. This record does not separate why. One explanation it does not test is that the arm takes a share of the gate's draws after the latch away from the twin and observed arms, on the cells where those arms were producing the hits. Its own measured reach covers lj38's motif from some incumbents and nothing else that was measured. `248e00c1` reverts both commits, so the tip's code is S1's.

## Files

- `rows.csv`: 252 campaign rows, 7 cells × 12 seeds × 3 arms.
- `reruns.csv`: the two repeated rows.
- `run_arms.py`: the campaign runner.
- `reach.csv`: the generator probe, with `reach.diff` the probe patch against `8eee13f8`.
