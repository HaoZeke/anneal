``global_optimize`` and ``global_optimize_objective`` accept ``x0``, a start
point inside the box that is evaluated first and becomes the first incumbent;
``portfolio_optimize_with_start`` is the Rust entry point. An invalid start
raises ``ValueError``.
