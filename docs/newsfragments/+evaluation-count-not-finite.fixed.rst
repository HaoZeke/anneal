``global_optimize`` counted in ``n_evals``, and ``minimize`` with
``replicas=1`` in ``nfev``, the evaluations a run charged at a point that is
not finite, which the objective never sees: on boxes with coordinates near
1e155 and beyond some arms propose such points, which are answered with
``+inf``. They still count against the budget, but now in ``n_answered``
(``diagnostics["n_answered"]`` for ``minimize``), so ``n_evals`` and ``nfev``
count calls. On ``[1e307, 1.7e307] x [0, 1]``, ``global_optimize`` of
``sum((x / [1.7e307, 1] - 0.3)**2)`` from the upper corner with seed 0 and 120
evaluations reported ``n_evals`` 120 for 93 calls, as ``minimize`` with
``replicas=1`` did in ``nfev``. Both now report 74 for 74 calls and
``n_answered`` 46, which also counts the 19 evaluations at points called
before, now answered without a call, and both call the objective at 0.11.1's
points in its order, each once.
