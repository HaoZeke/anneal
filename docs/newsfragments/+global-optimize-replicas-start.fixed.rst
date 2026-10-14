``global_optimize(..., x0=x0, replicas=R)`` with ``R`` above one evaluates
``x0`` first, once, as one replica does; the start had been checked and then
dropped. With no ``grad_fn`` or ``noise_sigma`` and the ``auto`` policy it
now runs the batch replicas of ``minimize``: ``R`` values-only loops start
from ``x0``, evaluated once and shared, ``coverage_shared`` chooses one
archive and incumbent or one each, each step's points that no replica has
evaluated before go to ``eval_batch`` when the objective has one,
``minimize(..., replicas=R)`` repeats the run
call for call, and the result reports ``n_batches``. There
``coverage_radius`` and ``coverage_neighbors``, which now default to
``None``, have nothing to act on and give a ``FutureWarning`` when given, as
they do in ``minimize``. Over 40 paired seeds at equal evaluations on 16
cells, eight of which draw the start near the minimum, four batch replicas
ended ahead of 0.11.1's ``global_optimize(..., x0=x0, replicas=4)``, which
dropped the start, on 13 cells and level on three. The comparison takes in
the batch replicas, where 0.11.1 ran persistent portfolio controllers, and
this release's other changes to the values-only loop as well as the start.
Without ``x0``, or with a gradient, declared noise or the ``legacy``
policy, ``R`` persistent portfolio controllers still run;
``portfolio_ensemble_optimize`` evaluates ``x0`` on the calling thread
before any replica starts, and replica 0 goes on from it as its first
evaluation and incumbent.
