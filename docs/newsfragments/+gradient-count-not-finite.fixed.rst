``global_optimize`` with ``grad_fn`` counted in ``n_grads`` the gradients a
run asked at a point that is not finite, though the gradient is never called
there: those a descent reaches when the gradient returns NaN or infinity, and
the one the probe at the box's centre asks when ``(low + high) / 2``
overflows. They still count against the budget, answered with zeros, but now
in ``n_answered``, so ``n_grads`` counts calls. With a ``grad_fn`` that
returns NaN, ``global_optimize`` of ``(x - 0.3)**2`` from the middle of
``[0, 1]`` with seed 5 and 1000 evaluations reported ``n_grads`` 105 for 77
calls; on ``[1e308, 1.7e308]``, ``global_optimize`` of
``(x / 1e308 - 1.3)**2`` from ``1.35e308`` with its gradient, seed 0 and 200
evaluations reported 19 for 18. They now report 27 for 27 calls and 13 for
13, since a gradient at a point called before is now answered without a call
as well. ``minimize`` with ``jac``, which ran ``ensemble_optimize`` in 0.11.1
and reported ``njev`` 744 for 744 calls and 96 for 96 on these problems, now
takes this route, with ``njev`` 27 for 27 and 13 for 13 and the rest in
``diagnostics["n_answered"]``.
