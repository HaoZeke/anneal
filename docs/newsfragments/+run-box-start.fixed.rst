``Reflected`` mirrors a proposal whose offset or sum overflows ``f64``
instead of placing it on a wall. The step comes from
``MoveKernel::displacement``, which ``Gaussian`` and ``Cauchy`` implement.
``run_rs_qmc_variant`` starts its chains on the finite wall of an axis with
an infinite wall. In the Rust sampler a NaN objective value counts as worse
than every number, ``+inf`` included, so a chain no longer stalls at a NaN
start and a NaN never beats a number. A Python objective that returns NaN
is still scored as ``+inf``, the same value an ordinary exception receives.
The unbounded ``boltzmann``, ``fast`` and ``gsa`` constructors stay on
``R^dim``; the boxed presets are ``boltzmann_in_box``, ``fast_in_box`` and
``gsa_in_box``.
