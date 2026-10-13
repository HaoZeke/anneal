``global_optimize`` counted in ``n_evals``, and ``minimize`` in ``nfev``, the
evaluations a run charged at a point that is not finite, which the objective
never sees: on boxes with coordinates near 1e155 and beyond some arms propose
such points, which are answered with ``+inf``. They still count against the
budget, but now in ``n_answered`` (``diagnostics["n_answered"]`` for
``minimize``), so ``n_evals`` and ``nfev`` count calls. On
``[1e307, 1.7e307] x [0, 1]``, ``minimize`` of
``sum((x / [1.7e307, 1] - 0.3)**2)`` from the upper corner with seed 0 and 120
evaluations reported ``nfev`` 101 for 74 calls and ``n_answered`` 19; it now
reports 74 and 46. The points asked are the same as before.
