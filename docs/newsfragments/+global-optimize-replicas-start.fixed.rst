``global_optimize(..., x0=x0, replicas=R)`` with ``R`` above one evaluates
``x0`` first, once, as one replica does; the start had been checked and then
dropped. With no ``grad_fn`` or ``noise_sigma`` and the ``auto`` policy it
now runs the batch replicas of ``minimize``: ``R`` values-only loops start
from ``x0``, evaluated once and shared, ``coverage_shared`` chooses one
archive and incumbent or one each, each step's points go to ``eval_batch``
when the objective has one, ``minimize(..., replicas=R)`` repeats the run
call for call, and the result reports ``n_batches``. Over 40 paired seeds at
equal evaluations on 16 cells, four of which draw the start near the
minimum, four batch replicas ended ahead of the run that dropped the start
on 13 cells and level on three, and ahead of four persistent replicas with
replica 0 at ``x0`` on 10 cells and level on six. Without ``x0``, or with a
gradient, declared noise or the ``legacy`` policy, ``R`` persistent
portfolio controllers still run; ``portfolio_ensemble_optimize`` evaluates
``x0`` on the calling thread before any replica starts, and replica 0 goes
on from it as its first evaluation and incumbent.
