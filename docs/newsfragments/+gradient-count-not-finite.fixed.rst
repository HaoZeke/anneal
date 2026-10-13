``global_optimize`` counted in ``n_grads``, and ``minimize`` with ``jac``
in ``njev``, the gradients a run asked at a point that is not finite, such
as those a descent reaches when the gradient returns NaN or infinity,
though the gradient is never called there. They still count against the
budget, answered with zeros, but now in ``n_answered``
(``diagnostics["n_answered"]`` for ``minimize``), so ``n_grads`` and
``njev`` count calls. With a ``jac`` that returns NaN, ``minimize`` of
``(x - 0.3)**2`` from the middle of ``[0, 1]`` with seed 5 and 1000
evaluations reported ``njev`` 55 for 27 calls.
