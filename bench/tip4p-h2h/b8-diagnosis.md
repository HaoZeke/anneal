# b8 rigid-quench cost diagnosis

This diagnosis uses b7 `bf4d1e4af5e8ecd581fb6b42f3d80fec29be029f`.
Release runs use the recommended rigid preset, seeds 0–11, and the
driver's stop test. Search code does not read a reference energy. Raw
logs are under `kits/tip4p-hits-b8-evidence/`.

## Baseline at 20,000 calls

| Bodies | Hits | Median calls/hop | Total hops |
| ---: | ---: | ---: | ---: |
| 11 | 3/12 | 75.047 | 2,951 |
| 12 | 3/12 | 70.291 | 2,988 |
| 13 | 1/12 | 75.472 | 3,098 |
| 14 | 0/12 | 75.908 | 3,260 |
| 15 | 0/12 | 76.191 | 3,210 |
| 16 | 0/12 | 77.822 | 3,099 |
| 17 | 0/12 | 75.047 | 3,214 |
| 18 | 0/12 | 77.973 | 3,090 |
| 19 | 0/12 | 77.673 | 3,094 |
| 20 | 0/12 | 77.822 | 3,090 |
| 21 | 0/12 | 76.337 | 3,132 |

At the two-million-call paper budget, 14 bodies hits 12/12 with a
median 69.128 calls per hop. Twenty-one bodies hits 0/12 with a median
69.704 calls per hop.

Instrumenting 11, 14, 17, and 21 bodies shows that screening is not the
cost: no screening pass runs. Full relaxations consume 100% of charged
quench calls, and convergence checks consume no additional calls because
the final value and gradient are already retained. Across the 12 seeds,
validated versus capped relaxations are:

| Bodies | Converged | Capped |
| ---: | ---: | ---: |
| 11 | 1,639 | 1,336 |
| 14 | 1,609 | 1,675 |
| 17 | 1,519 | 1,719 |
| 21 | 1,281 | 1,875 |

The fixed iteration ceiling therefore catches an increasing share of
quenches, but calls per hop stay near 70–78 because many other quenches
converge shortly before that ceiling.

## Tolerance

The rigid record contract accepts an RMS generalized force of 1e-3.
The solver stops at an infinity norm of 1e-3, which is stricter. Replacing
that stop with the mathematically equivalent Euclidean threshold
`sqrt(dimensions) * 1e-3` reduced median calls per hop by roughly 4–7%.
It also changed the basin walk and reduced the 20,000-call hit counts:

| Bodies | b7 hits | RMS-stop hits |
| ---: | ---: | ---: |
| 11 | 3/12 | 1/12 |
| 12 | 3/12 | 2/12 |
| 13 | 1/12 | 0/12 |
| 14–21 | 0/12 each | 0/12 each |

The looser stop is therefore rejected. It meets the record contract but
does not preserve search quality.

## Exact rejection bound

An accepted L-BFGS iterate gives an upper bound on the eventual minimum:
continuing descent can only lower its energy. Metropolis rejection needs
a lower bound to prove that no continuation can cross the sampled
acceptance level. The generic objective interface supplies no such lower
bound. Stopping from the current energy would reject some trials whose
continued quench is accepted, so it would not preserve accepted outcomes.

Known-gradient reuse already removes the only exact redundant call at the
quench boundary. There is no additional cheap exact early exit in the
current generic contract.

## Rotation chart

Translation coordinates have length units while exponential-map rotation
coordinates are dimensionless. Applying one isotropic L-BFGS metric to
both leaves the opening direction dimensionally unbalanced. The rigid-body
declaration already supplies a natural conversion: its largest body-frame
site radius. Scaling a rotation increment by that radius expresses both
coordinate blocks as site displacements.

A scratch implementation preserves the original physical infinity-norm
stop while optimizing in these geometry-scaled increments. It reduces
calls per hop at most measured sizes. At 20,000 calls it changes the total
from 7/132 to 9/132, adding three 14-body hits and one 16-body hit while
losing one hit at both 11 and 12 bodies. At the paper budget, 14 bodies
remains 12/12 and 21 bodies improves from 0/12 to 2/12.

This is the supported implementation experiment. Its scale comes only
from declared rigid geometry; it has no species, body-count, motif,
reference-energy, or fitted constant.
