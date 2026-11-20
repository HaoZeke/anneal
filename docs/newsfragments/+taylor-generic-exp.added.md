<!-- Copyright 2023--present rgpot developers -->

`rgpot_core::taylor::Taylor::from_coefficients` builds a truncated series from supplied interval coefficients and refuses an empty list with `JetError::Shape`.
The `TaylorCoefficient` trait names the coefficient operations that the exponential recurrence needs, and `TaylorExponential<C>` evaluates `exp(u)` one coefficient at a time in the Griewank-Walther order.
`Taylor::exp` now runs through that recurrence and returns the same enclosures as before.
