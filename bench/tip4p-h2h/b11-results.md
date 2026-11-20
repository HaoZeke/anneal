# b11 rigid-chart gate analysis

b11 is based on b6 `53d98cd20fd7614c4cb70571dccc57056589d569`.
The b8 comparison used `75106c35c8466db1d4262acff2b6f79ce8c9ba3b`.
All release runs use the recommended rigid preset and the Cambridge stop
test. Raw rows and counter logs are retained with the delivery evidence.

## Outside-time root cause

The b8 chart was instrumented on TIP4P14 and TIP4P21, seeds 0–11, at
20,000 calls. Its 24 rows contain 448,859 charged calls, 448,907 chart
evaluations, and 420,977 solver iterations.

| Chart operation | Nanoseconds per charged call |
| --- | ---: |
| Retained-gradient borrow and convergence scan | 261.113 |
| Scaled-to-physical coordinate map | 69.535 |
| Rotation-gradient map | 54.339 |
| Zero-vector allocation | 3.704 |
| Chart construction | 1.668 |
| Final expansion | 1.478 |
| Total | 391.837 |

The cost is not an allocation effect. The duplicate physical-gradient scan
is the largest counter, followed by two full coordinate passes required by
the transformed chart. Their measured 392 ns per call accounts for the
10–17 per cent outside-time increase reported by the b8 gate. A fused scan
prototype reduced the overhead but retained b8's changed trajectories.

## Hit regression

The hit drop is a trajectory regression, not a consequence of time spent
outside the potential. The b7 close-gap arm changes the adaptive allocator's
arm count and random stream before that arm need be selected. The b6/b7
first-divergence traces locate that effect on the first or second proposal.

The b8 chart adds a separate divergence. With the same b7 move library,
TIP4P19 seed 2000 differs on objective call 2 of the opening quench:

| Arm | Energy bits | State digest |
| --- | --- | --- |
| b7, unscaled quench | `c04065b048d193b4` | `755062e681713541` |
| b8, scaled chart | `c0417be10d984cb0` | `bbee859adafe1876` |

No proposal has occurred at that point. Coordinate scaling changes the
L-BFGS path itself. Thus the b8-versus-b6 hit changes combine two trajectory
changes: b7's allocator change and b8's quench reparameterisation. Optimising
the chart's copies cannot restore b6 outcomes.

The safe result is to reject both experimental mechanisms. This clean series
starts from b6, retains the b9 stable rigid path, and reapplies b10's
trajectory-neutral paid-check buffer reuse.

## Genericity audit

The b6-to-b8 production diff changes only two files. It has no per-system,
species, or cluster-size dispatch:

- `src/methods/cluster_hopping/moves.rs:327-329` adds close-gap to every
  enabled local rigid-body library.
- `src/methods/cluster_hopping/moves.rs:2535-2538` reads the declared body
  count. The lower bound is the operation's minimum arity; `6 * n` is the
  rigid chart dimension.
- `src/methods/cluster_search.rs:195-204` validates the declared rigid chart.
- `src/methods/cluster_search.rs:821-825` derives scale from declared site
  reach.
- `src/methods/cluster_search.rs:1121-1125` selects the chart from declared
  rigid geometry.

Fixed sizes at `moves.rs:3445-3453` and
`cluster_search.rs:2803-2826` are test fixtures, not production cases.

## Serial paper-budget comparison

The final column compares b6 with the clean b11 result. Each cell has seeds
2000–2011, runs one process at a time on one pinned CPU, and uses two million
calls with the reference plus the public hit tolerance as its stop.

| System | b6 hits | b11 hits | b6 outside us/call | b11 outside us/call | b11/b6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TIP4P8 | 12/12 | 12/12 | 1.243069 | 1.213454 | 0.976176 |
| TIP4P9 | 12/12 | 12/12 | 1.317980 | 1.298992 | 0.985593 |
| TIP4P10 | 12/12 | 12/12 | 1.427345 | 1.406344 | 0.985287 |
| TIP4P11 | 12/12 | 12/12 | 1.532406 | 1.522574 | 0.993584 |
| TIP4P12 | 12/12 | 12/12 | 1.775640 | 1.736325 | 0.977859 |
| TIP4P13 | 12/12 | 12/12 | 1.883270 | 1.871998 | 0.994014 |
| TIP4P14 | 12/12 | 12/12 | 2.046601 | 2.032448 | 0.993085 |
| TIP4P15 | 5/12 | 5/12 | 2.276486 | 2.243857 | 0.985667 |
| TIP4P16 | 1/12 | 1/12 | 2.526512 | 2.503733 | 0.990984 |
| TIP4P17 | 3/12 | 3/12 | 2.860251 | 2.824400 | 0.987466 |
| TIP4P18 | 3/12 | 3/12 | 3.010790 | 3.036923 | 1.008679 |
| TIP4P19 | 7/12 | 7/12 | 3.099627 | 3.076228 | 0.992451 |
| TIP4P20 | 4/12 | 4/12 | 3.341767 | 3.319446 | 0.993320 |
| TIP4P21 | 0/12 | 0/12 | 3.777886 | 3.739072 | 0.989726 |

All 168 paired outcomes have identical hit, call, charged-call, best-energy,
basin-count, and state rows. TIP4P9 has the same median call count, 5,839.5,
in both arms. TIP4P15–21 restores b6's 23/84 hits, rather than b8's 12/84.

Every outside-time ratio is below the 1.10 cap. Thirteen are within the
reported 0.978–1.041 A/A band. TIP4P8 is slightly faster at 0.976. Pooled
outside time is 2.944203 microseconds per call for b6 and 2.923467 for b11,
a ratio of 0.992957.
