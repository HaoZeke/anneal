``global_optimize`` and ``minimize`` called the objective again at points
they had its value at, ``x0`` above all when it is the box's minimiser on a
face or at a corner: a trial step the box projects back onto the iterate, or
a stencil point that rounds onto it, is a point already called. A run
without ``noise_sigma`` now takes the objective and the gradient to be
functions of the point, and answers an evaluation at a point where it has
called the objective from that call, and one of the gradient likewise;
points match bit for bit, so ``-0.0`` and ``0.0`` are two. The
evaluation still counts against the budget, so where the objective is a
function of the point answering it changes no step of a run, and the result
counts the calls in ``n_evals`` and ``n_grads`` (``nfev`` and ``njev``) and the evaluations answered in
``n_answered`` (``diagnostics["n_answered"]`` for ``minimize``). From
``x0 = 1`` on ``[1, 2]``, where ``(x - 0.3)**2`` is least, five runs of 200
evaluations called the objective at ``x0`` 131 times through
``global_optimize`` and 336 times through ``minimize``, which ran four
replicas by default, and from the corner of ``[1, 2]**6`` 36 and 96 times.
Each run now calls it there once, with one replica or four (``minimize`` now
runs one by default, and four as batch replicas), and the five runs make 808
calls of their 1000 evaluations with one replica and 839 with four in one
dimension, 789 and 870 in six. With a gradient and a budget of 25, from
``x0 = 1`` on ``[1, nextafter(1, 2)]``, ``global_optimize`` called ``fun`` at
``x0`` on 17 of its 21 calls with seed 0 and 16 of 21 with seed 1, and
``minimize`` with ``jac``, which ran ``ensemble_optimize`` in 0.11.1, on 2 of
its 5 calls and 8 of 8; both now call ``fun`` once at each end and ``jac``
once. Over ten seeds of 1000 evaluations from a start drawn in the box,
``global_optimize`` repeated 958 calls on a sphere least at a corner of
``[-1, 1]**10``, and ``minimize`` with its four replicas 2117, and both now
repeat none, with one replica or four. Batch replicas evaluate only the rows
no replica has evaluated and make no call for a step without one; with
private histories, whose descents from ``x0`` ask for the same points, four
replicas on the regime notes' six-parameter binary Lennard-Jones fit make a
median of 251.5 calls in 131 batches at 1000 evaluations. Persistent
replicas, which run concurrently, keep a record each, so a point two of them
reach is called by each, and with ``noise_sigma`` every evaluation is still a
call. An objective that draws a random value or keeps state is not called
again at a repeated point either: with a normal draw of standard deviation
0.05 from ``numpy.random.default_rng(0)`` added at each call,
``global_optimize`` of ``(x - 0.3)**2`` from ``x0 = 1`` on ``[1, 2]`` with
seed 0 and 200 evaluations called it at ``x0`` 17 times, for 17 draws, as
``minimize`` with ``replicas=1`` did, and now calls it there once and answers
32 evaluations from earlier draws. ``minimize`` and
the ChemFit bridges take no ``noise_sigma``; ``global_optimize`` with
``noise_sigma=0.05`` and the same start, box, seed and budget calls it at
all 200 evaluations.
