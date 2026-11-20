`LJPot` has a dense analytic Hessian, `hessianImpl`, and a dense Hessian and
Hessian-vector product from dual numbers, `hessianAdImpl` and
`hessianVectorAdImpl`, and advertises them as `analyticHessian`,
`adHessian` and `adHvp`. The dual-number pair term is the templated
`lj::pairEnergy` of `rgpot/LennardJones/LJPairEnergy.hpp`, the energy the
force kernel sums, so its second derivative comes from the energy itself
rather than from the hand-written one. Both Hessians are symmetric to the
last bit and agree to 1e-12 of their largest entry. Coincident atoms throw
`std::domain_error`.
