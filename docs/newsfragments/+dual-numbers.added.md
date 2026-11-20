`rgpot/Dual.hpp` adds forward-mode dual numbers, `rgpot::ad::Dual<T>`, with
`Dual2` for second derivatives and `secondDerivative` to read a value and
its first two derivatives from one evaluation. A kernel written as a
template over its scalar type is differentiated from its own source, with
no step and no truncation error.
