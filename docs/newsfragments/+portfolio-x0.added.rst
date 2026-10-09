``anneal.global_optimize`` (and the native-handle
``global_optimize_objective``) accept an ``x0`` starting position, evaluated
once up front for one charged budget unit and installed as the incumbent the
Thompson-allocated arms improve on. New Rust entry points
``portfolio_optimize_seeded`` and ``portfolio_optimize_with_policy_seeded``
expose the same warm-start anchor to native consumers.
