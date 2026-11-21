`MorsePot` has a dense analytic Hessian, `hessianImpl`, built from the pair
derivatives of its Hessian-vector product, and advertises it as
`analyticHessian`. It is symmetric to the last bit. Morse has no
dual-number path; its `hessianAdImpl` and `hessianVectorAdImpl` throw.
