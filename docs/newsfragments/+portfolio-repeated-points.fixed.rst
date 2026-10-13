``global_optimize`` and ``minimize`` called the objective again at points
they had its value at, ``x0`` above all when it is the box's minimiser on a
face or at a corner: a trial step the box projects back onto the iterate, or
a stencil point that rounds onto it, is a point already called. A run
without ``noise_sigma`` now answers an evaluation at a point where it has
called the objective from that call, and one of the gradient likewise. The
evaluation still counts against the budget, so every run takes the steps it
took and ends where it did, and the result counts the calls in ``n_evals``
and ``n_grads`` (``nfev`` and ``njev``) and the evaluations answered in
``n_answered`` (``diagnostics["n_answered"]`` for ``minimize``). From
``x0 = 1`` on ``[1, 2]``, where ``(x - 0.3)**2`` is least, five runs of 200
evaluations called the objective at ``x0`` 131 times through
``global_optimize`` and 88 times through ``minimize`` with four replicas,
and from the corner of ``[1, 2]**6`` 36 and 15 times; each run now calls it
there once, and the five runs make 808 and 839 calls of their 1000
evaluations in one dimension, 789 and 870 in six. On
``[1, nextafter(1, 2)]`` from ``x0 = 1`` with ``jac``, ``minimize`` called
``fun`` at ``x0`` on 17 of its 21 calls with seed 0 and 16 with seed 1, and
now calls ``fun`` once at each end and ``jac`` once. Over ten seeds of 1000
evaluations from a start drawn in the box, ``minimize`` repeated 1139 calls
on a sphere least at a corner of ``[-1, 1]**10`` and 2501 with four
replicas sharing their history, and now repeats none. Batch replicas
evaluate only the rows no replica has evaluated and make no call for a step
without one, so ``nit`` can fall; with private histories, whose descents
from ``x0`` ask for the same points, four replicas on the regime notes'
six-parameter binary Lennard-Jones fit make a median of 251.5 calls in 131
batches at 1000 evaluations, where they made 1000 in 251, and run as they
did. Persistent replicas, which run concurrently, keep a record each, so a
point two of them reach is called by each, and with ``noise_sigma`` every
evaluation is still a call.
