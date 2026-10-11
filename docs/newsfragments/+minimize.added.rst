``anneal.minimize(fun, x0, bounds, jac=None, *, budget=None, seed=0,
replicas=1, history="shared", ...)`` minimizes over a box from ``x0`` with
the portfolio of ``global_optimize``. ``bounds`` is SciPy ``Bounds``, an
``(n, 2)`` array or ``n`` pairs; a coordinate whose ends are equal is fixed
and the search runs in the others, while ``fun`` and ``jac`` still receive
full vectors. ``x0`` is the first evaluation and is made once, and
``budget`` counts every objective and gradient call, by default
``max(1000, 10 n (n + 1))`` for ``n`` free coordinates. With one replica
``minimize`` repeats ``global_optimize(..., x0=x0)`` call for call, ``jac``
taking its gradient route. With ``replicas=R`` and no ``jac``, ``R``
values-only portfolios run from ``x0`` as batch replicas under the one
budget, and each step's points go to ``fun.eval_batch`` when ``fun`` has
one. ``history="shared"`` gives the replicas one archive and incumbent,
``"private"`` or ``"none"`` gives each its own, and ``coverage_shared``
overrides ``history`` when given. A ``store`` that holds a finite
observation inside the box starts the run from the best such one and
receives the result. ``membership``, ``coverage_radius`` and
``coverage_neighbors`` steer the hop chains of ``ensemble_optimize``; here
they are checked and have no effect, with a ``FutureWarning``, as does
``replicas`` above one with ``jac``, which runs one replica. The result
holds ``x``, ``fun``, ``nfev``, ``njev``, ``nit`` (sequential steps: calls
with one replica, batches with more), ``success``, ``message`` and the
engine's ``diagnostics``. One replica is the default: over 40 paired seeds
at equal evaluations, four shared replicas ended worse than one on 13-atom
Lennard-Jones and on ten-dimensional Rastrigin and Rosenbrock at 1000
evaluations, better on a binary Lennard-Jones parameter fit at 1000, and
level with one on all four problems at 5000.
