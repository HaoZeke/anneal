# b8 geometry-scaled rigid quench

All release runs use b7 `bf4d1e4af5e8ecd581fb6b42f3d80fec29be029f`
as the baseline. TIP4P rows use seeds 0–11 and the driver's stop test.
The search does not read a reference energy. Raw logs are retained with
the delivery evidence.

## Method

The rigid-body declaration supplies the largest body-frame site radius.
b8 divides local rotation increments by this radius, so translations and
rotations use the same physical length units in L-BFGS. The physical
infinity-norm convergence test is unchanged. There is no fitted scale.

## 20,000-call panel

| Bodies | b7 hits | b8 hits | b7 calls/hop | b8 calls/hop |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 3/12 | 2/12 | 75.047 | 73.422 |
| 12 | 3/12 | 2/12 | 70.291 | 70.672 |
| 13 | 1/12 | 1/12 | 75.472 | 73.937 |
| 14 | 0/12 | 3/12 | 75.908 | 73.937 |
| 15 | 0/12 | 0/12 | 76.191 | 75.047 |
| 16 | 0/12 | 1/12 | 77.822 | 74.488 |
| 17 | 0/12 | 0/12 | 75.047 | 75.189 |
| 18 | 0/12 | 0/12 | 77.973 | 76.482 |
| 19 | 0/12 | 0/12 | 77.673 | 75.330 |
| 20 | 0/12 | 0/12 | 77.822 | 77.074 |
| 21 | 0/12 | 0/12 | 76.337 | 76.783 |

The total improves from 7/132 to 9/132. The plain losses are one hit
each at 11 and 12 bodies. Gains are three hits at 14 bodies and one at
16 bodies.

## Paper budget

At two million calls, 14 bodies remains 12/12. Its median calls per hop
changes from 69.128 to 68.950. Twenty-one bodies improves from 0/12 to
2/12, while median calls per hop changes from 69.704 to 68.895.

## Control walks and outside time

The complete `Outcome` digests for LJ38, LJ75, Morse 38 at range 6 and
Morse 38 at range 14 are bit-identical to b7 for seeds 0–2 at 20,000
calls.

On a paired 20,000-call timing run, weighted outside time is 2.538849
microseconds per call for b7 and 2.843823 for b8, a 1.1201 ratio. At the
paper budget it is 3.728791 and 4.085298, a 1.0956 ratio. Over the full
requested workload, including both budgets, it is 3.619092 and 3.957440,
a 1.0935 ratio.

An exact RMS-equivalent convergence stop reduced calls per hop by about
4–7% but reduced hits, so it was discarded. Two geometry-scaled scalar
initial-curvature variants did not improve the combined quality and cost
and were also discarded.
