``anneal.run`` and ``anneal.run_qmc`` now drive box-constrained variants
(``BoxConstrained`` neighborhood with mirror-reflected moves) instead of the
unconstrained ``R^dim`` presets, so every evaluation point lies inside
``[low, high]``. Both entry points also accept an ``x0`` starting position,
which is clipped into the box and evaluated once before the chain starts;
``run_qmc`` runs it as one extra chain alongside the QMC starts. Bounds are
now validated strictly (finite, ``low < high`` per dimension) like every
other entry point.
