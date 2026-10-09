``global_optimize`` and ``global_optimize_objective`` take ``x0``. The
portfolio charges it as its first evaluation and records it as the first
incumbent, so every arm that starts from the incumbent (hop, shift,
trust-region poll, GLE, HMC, population and reduced-space arms) starts from
``x0`` until a lower point is found. A starting point outside the box, of the
wrong length, or not finite raises ``ValueError``.
``portfolio_optimize_from`` is the Rust entry point.
