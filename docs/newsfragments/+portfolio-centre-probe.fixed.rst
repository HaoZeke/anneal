``global_optimize`` with a ``grad_fn`` reported in ``n_evals`` one objective
evaluation more than it made when the value or the gradient at the centre of
the box was not finite, or the gradient there was zero, as it is at a
stationary point and as a ``grad_fn`` that raises is read, given a budget of
at least 96 evaluations on a box of one or two coordinates, 224 on three or
288 on four. Such a run opens with a probe at the centre that reserves two
evaluations and a gradient together, and the probe counted all three even
when it stopped after the first evaluation. It now counts each call once it
is made and still reserves the same work, so every run makes the same calls
as before.
