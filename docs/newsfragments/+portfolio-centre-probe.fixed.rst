``global_optimize`` with a ``grad_fn`` reported in ``n_evals`` more objective
evaluations than it made when the value or the gradient at the centre of the
box was not finite, or the gradient there was zero, as it is at a stationary
point and as a ``grad_fn`` that raises is read: one more for each replica
whose share of the budget, all of it with the default one replica, came to at
least 96 evaluations on a box of one or two coordinates, 224 on three or 288
on four. Each such replica opens with a probe at the centre that reserves two
evaluations and a gradient together, and the probe counted all three even
when it stopped after the first evaluation. It now counts each call once it
is made and gives back the unit it reserved for the evaluation it does not
make, so that ``n_evals + n_grads + n_answered`` is what the run spent. Such
a run has that unit left to spend, and the points it asks after the probe can
differ from those it asked before.
