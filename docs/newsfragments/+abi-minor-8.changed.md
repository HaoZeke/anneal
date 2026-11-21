The C ABI minor of `rgpot-core` is now 8, in `RGPOT_ABI_VERSION_MINOR`, for the
additive dense-Hessian, dual-number and derivative-capability functions, the
`rgpot_hessian_scheme_t` enum, the `rgpot_derivative_caps_t` and
`rgpot_derivative_counts_t` structs, and the `RGPOT_HVP_AD` and
`RGPOT_HVP_EXACT` values. A caller built against minor 0 to 7 still matches; a
caller built against minor 8 is refused by an older library.
