``anneal.minimize(fun, x0, bounds, jac=None, *, budget=None, seed=0,
replicas=1, history="shared", ...)`` minimizes over a box from ``x0`` with
the portfolio of ``global_optimize`` rather than through
``ensemble_optimize``, and runs one replica by default rather than four
hop chains. ``bounds`` is SciPy ``Bounds``, an ``(n, 2)`` array or ``n``
pairs; a coordinate whose ends are equal is fixed and the search runs in
the others, while ``fun`` and ``jac`` still receive full vectors. ``x0``
is the first evaluation and is made once, and
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
at equal evaluations, four shared batch replicas ended worse than one on
13-atom Lennard-Jones and on ten-dimensional Rastrigin and Rosenbrock at
1000 evaluations, better on a binary Lennard-Jones parameter fit at 1000,
and at 5000 level with one on the other three problems and worse on the
fit. The default budget replaces ``max(32, 80 n)``, in which ``n``
counted fixed coordinates as well as free ones. Over 20 seeds from a
start drawn in the box, every run spending its budget, it ended better
on three-dimensional Rastrigin (median 0 against 2.49, 1000 evaluations
against 240), the six-parameter fit (-4.81 against -1.78, 1000 against
480), ten-dimensional Rastrigin (3.98 against 5.47) and Rosenbrock
(3.5e-20 against 1.3e-3, 1100 against 800 for both) and 13-atom
Lennard-Jones (-40.7 against -39.4, 15600 against 3120, lower on 14 of
the 20 seeds), at a one-sided Wilcoxon ``p`` below 0.05 on each, and was
level on three-dimensional Rosenbrock, which both solve. Both open with the finite-difference descent from ``x0`` up to 15
free coordinates; on the 39 of Lennard-Jones only the new default
reaches the ``5 n (n + 1)`` evaluations the opening needs.
