This sweep removed the dead code.

## Factor table

`spme_reciprocal` built a cached table of Gaussian factors.
The table is a particle mesh Ewald (PME) cache.
The bench uses the float charge-only path.
Nothing read the table.

The float kernel builds its own scale.
The stress kernel recomputes the factor from the reciprocal vector while
it accumulates the virial.
`apply_cached` was the only reader of the cached table.
That lambda sat in a branch that cannot run.
The float path runs when stress is off.
The stress path calls `apply_stress`.
The sweep deleted the table and `apply_cached` together.

The deleted table is separate from the singular electronegativity
equilibration (EEQ) checks in the D4 factorization.
Those checks are the failure mode of a singular matrix.

## Unused spread arguments

`deposit` ran with atomic adds off and a null private grid.
`scatter_cached` ran with a null destination.
Neither branch could run.
The spread writes the shared float mesh, or the double mesh when the
float buffer is empty.

## Serial flag that nothing read

Factorizing the EEQ matrix calls `apply_one` directly.
`apply` checked a thread-local flag that only that factorization set, so
the check never saw the flag raised.
The sweep removed the flag and the check.
`apply` is the parallel matvec the conjugate gradient uses.

## Order-5 derivative special case

`bspline_derivative` had a branch for order 5.
The order-5 spread uses
`weights_order5` and never calls `bspline_derivative`.
The sweep removed that branch.
