# Why the nm3-land9 refactor is slower, and why sc108 misses

Arm A is develop `61fa67cf`. Arm B is `refactor/nm3-land9` at `4dda7c35`. Both are release builds with the same optimisation and with symbols. Twelve seeds, `2000` through `2011`, one core, one thread, no shortened budget. Even seeds run A then B; odd seeds run B then A. `lj104` spends 4,000,000 potential calls. `sc108-80` spends 1,000,000. A hit is the driver's reference test.

The move counter does not change A's trajectory. On `lj104` seeds 2000 and 2001 the full budget gives the same best-energy bits with the counter and without it: `c082304eb3e56fc0` and `c08228222729a380`.

## The extra outside time is the symmetry residual

Outside time is wall time minus time inside the potential, divided by charged potential calls.

On `lj104`, twelve seeds pooled, A spends 10.825 µs outside per call and B spends 14.088 µs. The ratio is 1.301. A charged 34,528,945 calls (373.787 s outside) and B charged 31,381,091 (442.092 s). The hit flags go the other way from the Sutton-Chen loss: A 5/12, B 8/12. The slowdown is a higher cost per call, not a longer search after a miss.

Call stacks cover seeds 2000, 2001 and 2002, sampled on the cpu clock at 399 Hz and unwound with frame pointers. Samples inside the potential are dropped. That is 52,526 outside samples on A and 35,160 on B. On these three seeds the outside cost is 10.937 µs and 13.734 µs, a rise of 2.797 µs per call (ratio 1.256). B hits on all three and so stops early; the per-call figure is what the stacks are scaled to.

`detect_all` is the parent of the rise. Its inclusive time goes from 1.961 µs per call (17.9% of A's outside samples) to 4.914 µs (35.8% of B's). The increase, 2.95 µs, matches the whole rise.

Two functions inside it hold the time. Shares below are self time unless they say inclusive. Inclusive time of the second function includes the first, so the two self shares are not added to the inclusive share.

1. `Cloud::nearest_index_avx2`, which A does not have. It is 23.5% of B's outside samples, 3.225 µs per charged call. A spent 1.354 µs (12.4%) in `Cloud::nearest_squared_avx2`, and that falls to 0.168 µs on B. The scan costs 2.039 µs more per call, 73% of the 2.797 µs rise.
2. `Cloud::symmetry_deviation`. Self time is 8.5% of B, 1.168 µs per call, which is 42% of the rise. Inclusive time is 32.7% (4.486 µs) because it calls the scan. A has `rotation_deviation` at 0.192 µs self and does not have this function.

The two self increases sum to more than the net rise. Other outside work is cheaper by about 0.4 µs per call, mostly `bias::packed_squared_distance_exceeds` and the old rotation path.

The old scan keeps a four-lane minimum in a register and stores it once. The new scan has to return the partner index, so it stores every four-lane chunk and picks the winner in scalar code. The wider scan that stores only when a lane is closer is not the one these runs executed. Detection used to try orders 2, 3, 4, 5 and 6. It now tries every order from 2 through the atom count, 103 orders at 104 atoms, and the residual asks for a nearest image. That is the generality worth keeping. The time is the index scan under that list, not the list itself: a residual that still only needed a distance would be the old scan.

Potential calls per proposal barely move, so the extra time is not more evaluations. Summed over the twelve `lj104` seeds, a symmetrise proposal charges 36.6 calls on A and 34.0 on B. Burst, kernel, single-point, soap and surface stay within a call or two of A's rate. `tables/move-counts.csv` has proposals, accepts and charged calls for every move, seed and arm. On a run that spends the whole budget those charges plus the between-move gap equal the charged total. A run that stops on a hit counts the gap before the unfinished move twice. The excess is 3 or 4 calls on most of those seeds, and 229 to 304 calls on four `lj104` seeds.

## sc108 leaves the chain at the first symmetrise

Twelve seeds. A hits on 2000, 2001 and 2004, three of twelve. B hits on none. Pooled outside time is 11.042 µs on A and 11.418 µs on B, ratio 1.034. The miss is not the `lj104` cost. The chains take different proposals.

On every seed the first difference is a `sym` step, at index 0, 1, 3, 4, 5, 6 or 7. The move name matches. The draw count at the start of that step matches. The proposal-energy bits do not match, on any seed, and the potential calls inside the step differ on every seed. On seeds 2001, 2003 and 2004 the draw count at the end of the step also differs, by one or two. Earlier steps match in move, energy, draw count and acceptance. `tables/sc108-diverge.csv` is the per-seed row.

`Symmetrise` draws one index into its order list. A stores `[2, 3, 4, 5, 6]`. B stores `2..=n`, 107 orders at 108 atoms. The same draw names a different order, so the rotated geometry differs while the stream up to that draw does not. The quench of that geometry is why the potential-call count inside the step changes, and why an acceptance draw can add one sample afterwards. This is the longer order list, not a reordering of random calls.

On the two hitting seeds that diverge at step 0, that difference is the whole miss. Seed 2000 accepts both proposals: A then spends 1,636 potential calls, B spends 37. Seed 2004 accepts A's proposal after 2,149 calls and rejects B's after 35, with one extra draw on the rejection.

## Tables

`tables/runs.csv` is one row per arm, system and seed: charged calls, outside seconds, outside time per call, hit, best energy and its bits.

`tables/identity.csv` is the two `lj104` runs of A without the counter.

`tables/move-counts.csv` is the per-move counts.

`tables/lj104-a.folded` and `tables/lj104-b.folded` are the collapsed outside stacks for seeds 2000 to 2002.

`tables/lj104-outside-a.csv` and `tables/lj104-outside-b.csv` split those stacks into self and inclusive time per symbol, scaled by the outside seconds and divided by the charged calls.

`tables/lj104-perf-diff.csv` is the thirty symbols with the largest rise in self share.

`tables/sc108-diverge.csv` is the first differing step of each seed.
