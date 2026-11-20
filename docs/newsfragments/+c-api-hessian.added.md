The C API of `rgpot-core` computes dense Hessians with
`rgpot_potential_hessian`, under an explicit `rgpot_hessian_scheme_t`
(`RGPOT_HESSIAN_ANALYTIC`, `RGPOT_HESSIAN_AD`, `RGPOT_HESSIAN_EXACT` or
`RGPOT_HESSIAN_CENTRAL`, the last making `6 * n_atoms` force calls), and writes
the scheme that ran to `scheme_used`. `rgpot_potential_set_hessian_callback`
and `rgpot_potential_set_ad_hessian_callback` register the analytic and
dual-number Hessians, and `rgpot_potential_set_ad_hvp_callback` a dual-number
Hessian-vector product, which `RGPOT_HVP_AD` calls. `RGPOT_HVP_EXACT` takes the
analytic product, else the dual-number one, and refuses rather than
differencing. `rgpot_potential_hvp_resolved` reports the scheme an
`RGPOT_HVP_AUTO` product resolved to. `rgpot_potential_derivative_caps` reports
which analytic, dual-number and finite-difference schemes a handle has, with
their force-call costs, and `rgpot_potential_derivative_counts` counts the
requests each scheme answered.
