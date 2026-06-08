Fixed-budget benchmark drivers (``anneal_sota.qmc_annealed_hybrid`` and ``sota_cutest``) wire the shared ``eindir`` and ``anneal`` primitives through the comparison path:

- Tensor-train, additive, and Chebyshev surrogates as Move-slot proposal sources.
- GLE colored-noise dynamics with native gradients.
- Native-gradient L-BFGS-B and QMC polish with objective and gradient evaluations charged through ``Counter``.
- QMC seeding through the typed algebra components.

The ``hybrid_de`` / anneal entry in comparisons now calls the same hybrid path used by the benchmark helpers instead of a separate reimplementation.
