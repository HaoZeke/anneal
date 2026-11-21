# b9 rigid trajectory and overhead analysis

b9 is based on b8 `75106c35c8466db1d4262acff2b6f79ce8c9ba3b`.
Release runs use the recommended rigid preset. Raw logs are retained with
the delivery evidence.

## Rigid-chart profile

The b8 chart was instrumented on 14 and 21 bodies, seeds 0–11, at
20,000 calls. The totals cover 448,859 charged calls:

| Component | Nanoseconds per charged call |
| --- | ---: |
| Retained-gradient borrow and convergence scan | 261.1 |
| `set_scaled` coordinate map | 69.5 |
| Rotation-gradient map | 54.3 |
| Zero-vector allocation | 3.7 |
| Chart allocation | 1.7 |
| Final expansion | 1.5 |

There were 448,907 chart evaluations and 420,977 solver iterations.
Allocations and final copies were negligible. The duplicate convergence
scan was the largest chart-local cost. A candidate which fused that scan
with the gradient map preserved every tested b8 outcome and reduced the
matched 14- and 21-body outside time from 3.230123 to 3.041528
microseconds per call.

## b6 to b7 divergence

The paper-budget TIP4P19 rows were reproduced for seeds 2000–2011.
b6 hits 7/12 at seeds 2000, 2005, 2006, 2007, 2009, 2010, and 2011.
b7 hits 2/12 at seeds 2001 and 2007. There are therefore six b6-only
hits, not five.

Objective value and state-bit traces locate the first divergence:

| Seed | First call | b6 proposal | b7 proposal |
| ---: | ---: | --- | --- |
| 2000 | 194 | `rkick` | `rtrans` |
| 2005 | 109 | `rcycle` | `rclose` |
| 2006 | 207 | `rtrans` | `rtrans` |
| 2009 | 104 | `rtrans` | `rtrans` |
| 2010 | 107 | `rreor` | `rreor` |
| 2011 | 213 | `rtrans` | `rtrans` |

The close-gap arm changed the adaptive allocator's arm count. This
changed either the selected arm or the random stream consumed before an
otherwise identically named proposal. All six trajectories diverge on
the first or second evaluated proposal, before close-gap geometry could
demonstrate a general benefit.

b8 does not repair the directional loss: it hits 3/12 on these rows.
Removing only close-gap while retaining the chart hits 4/12. Thus exact
b8 outcome identity conflicts with the newer requirement of no cell loss
against b6. b9 follows the newer requirement and removes both experimental
mechanisms. Its rigid search source is byte-identical to b6, so every b6
rigid trajectory, energy, and hit is restored without a size or species
condition.

## Timing

The repeated b7 20,000-call outside-time values are 2.496509,
2.538849, and 2.559707 microseconds per call. This A/A band is
2.496509–2.559707. b9 measures 2.419128, or 0.9528 times the b7
median and 0.9690 times the fastest b7 replicate.

At the paper budget, repeated b7 values are 3.728791 and 3.734502
microseconds per call. b9 measures 3.727892, a ratio of 0.9990 against
their midpoint.

At 20,000 calls, b9 records 4/132 hits over 11–21 bodies, versus 9/132
for b8. At the measured paper rows, 14 bodies is 12/12 and 21 bodies is
0/12, exactly restoring b6.
