`LJPot` has an analytic Hessian-vector product, the pair second derivatives
of the shifted or switched 12-6 term contracted with the direction, so
`HvpScheme::Auto` and `rgpot.LJPot.hvp` make no force calls. Two atoms on
one point make the product NaN, which `HessianVectorCache` reports as
`std::domain_error`.
