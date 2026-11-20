`rgpot::PotentialHandle::hessian` computes the dense Hessian through the Rust
core under an explicit scheme and returns the one that ran, and
`PotentialHandle::hvp` now returns the scheme that computed the product, so an
`RGPOT_HVP_AUTO` request that resolved to forward differences reports it.
`from_impl` registers the analytic and dual-number Hessian and Hessian-vector
hooks of a potential from its `caps()` bits, and `derivative_caps` and
`derivative_counts` report what the handle can do and has done.
