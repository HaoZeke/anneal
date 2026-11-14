Hessian-vector products for any potential, for minimum-mode searches
(Lanczos, dimer) that apply the Hessian to many directions at one geometry.
In the C ABI, `rgpot_potential_hvp` computes `H v` on a potential handle.
`RGPOT_HVP_FORWARD`, the default, reuses the forces of the last centre the
handle evaluated, so k products at one geometry cost k + 1 force calls
instead of 2k, at first order in the step.
`RGPOT_HVP_CENTRAL` costs two calls a product at second order.
A potential that knows its Hessian registers an analytic product with
`rgpot_potential_set_hvp_callback` and the products make no force call.
`rgpot_potential_hvp_centre` returns the centre energy and forces from the
same cache, `rgpot_potential_hvp_invalidate` drops it, and
`rgpot_potential_hvp_counts` reports the calls made.
The centre is keyed on the bits of the positions, atomic numbers and cell.
In C++, `PotentialHandle` gains `hvp`, `hvp_centre`, `set_hvp_callback`,
`invalidate_hvp` and `hvp_counts`, and `rgpot/HessianVector.hpp` gives the
same products to potentials used without the Rust core.
`MorsePot::hessianVectorImpl` is the analytic product of the shifted and
switched Morse potential, and `PotentialHandle::from_impl` registers it.
In Python, every potential gains `hvp`, `hvp_centre`, `invalidate_hvp`,
`hvp_counts`, `reset_hvp_counts` and `has_analytic_hvp`.
