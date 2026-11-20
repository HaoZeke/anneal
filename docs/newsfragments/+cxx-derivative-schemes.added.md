`rgpot::HessianVectorCache::apply` returns the scheme that computed the
product, so a `HvpScheme::Auto` request that resolved to `Forward` reports it,
and `derivativeCounts()` counts products by that scheme. `HvpScheme::Ad`
uses a potential's dual-number product and `HvpScheme::Exact` takes the
analytic product, else the dual-number one, and refuses rather than
differencing. `HessianVectorCache::hessian` writes the dense Hessian under an
explicit `HessianScheme` (`Analytic`, `Ad`, `Exact` or `Central`, the last
making `6N` force calls) and returns the one that ran. `rgpot::derivativeCaps`
reports which analytic, dual-number and finite-difference ways a potential
has, with their force-call costs.
