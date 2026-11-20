The Python potentials gain `hessian(positions, atom_types, box, scheme)`, which
returns the dense `(3N, 3N)` Hessian and the name of the scheme that computed
it; `scheme` has no default, and `"analytic"`, `"ad"` and `"exact"` are refused
on a kernel without them while `"central"` makes `6N` force calls.
`hvp_resolved` returns the product with the scheme that ran, so an `"auto"`
product that resolved to forward differences reports it, and `hvp` accepts
`"ad"` and `"exact"`. `derivative_caps()` reports which analytic, dual-number
and finite-difference schemes a kernel has, with their force-call costs, and
`derivative_counts()` counts the requests each scheme answered.
