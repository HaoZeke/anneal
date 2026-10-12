The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, also models the logarithm of
its values while every value it has started from or accepted is positive, and
its line searches step on that model where the values are markedly further
from quadratic than their logarithm. Along a step a quadratic changes by the
step times the mean of its end slopes; each curvature pair counts for the
logarithm when that rule misses the change of the values more than four times
as far as the change of their logarithm, and for the values when it misses the
logarithm's that much further. From two pairs per coordinate on, once more of
the pairs since the descent last started afresh have counted for the logarithm
than for the values, the line searches step on its model while it predicts
each accepted value at least as closely as the model of the values, and a
value at or below zero retires it. A squared error far from its fit falls
through decades and is nearer quadratic in its logarithm. With the mean
squared energy error per atom of a six-parameter binary Lennard-Jones fit as
the loss, the loop ended lower than nevergrad's ``NgIohTuned`` on 17 of 40
seeds at 1000 evaluations: the descent had 10 of the loop's 25 turns per run,
against 35 of 36 with the logarithm of that error as the loss, the other arms
lowered the incumbent in few runs, and before its second curvature pair per
coordinate the model of the values missed the values of the descent's
accepted steps by a median of 39%, against 7.5% for a model of their
logarithm. Over 40 paired seeds from starts in the middle 60% of the box, at
1000 evaluations that fit now ends at a median of 0.0137 against 0.0332 for
``NgIohTuned`` (28/0/12) and 0.0331 for the loop before (25/10/5), a
twelve-parameter fit of three species at 2.63 against 6.88 (29/0/11) and 2.67
(7/33/0), and a two-parameter pair fit at 0, level with both (0/40/0); at 50
evaluations the pair fit ends at 0.789 against 0.993 (27/0/13) and 3.55
(34/5/1), and at 500 the binary fit at 0.0728 against 0.0783 (21/0/19) and
0.129 (22/11/7). On 13-atom Lennard-Jones positions, the six-parameter fit
with the logarithm of its error as the loss, and ten-dimensional Rastrigin and
Rosenbrock, at 1000 and 5000 evaluations, the loop stays level with the loop
before, and ahead of ``NgIohTuned`` except on Rosenbrock at 5000, where the
two are level. ``NgIohTuned`` still ends lower on more seeds of the pair fit
at 200 evaluations, on 12 of the 14 that do not tie, where both medians are
below ``1e-14`` (the one-sided Wilcoxon probability that the loop is not
worse is 0.015), and of the twelve-parameter fit at 200, on 21 of 40 (0.11),
and at 1000 on 12 of the 40 seeds of the binary fit and 11 of the
twelve-parameter one. Four shared batch replicas stay level with those before
on every cell of those four problems, but now trail one replica on the fit of
the logarithm at 5000 evaluations (17/3/20, 0.043), where the two were level
(19/3/18). Runs
in which the descent steps on the logarithm, which it cannot before
``2 d (d + 1)`` evaluations for ``d`` free coordinates, no longer repeat those
of 0.11.1: over 40 seeds at 1000 and at 5000 evaluations, those on the
Lennard-Jones positions and Rastrigin all repeat the loop before, seven change
on each of the log-loss fit and Rosenbrock, and 30 on the plain-loss binary
fit. Multiplying the objective by a power of two still leaves a run unchanged,
while a constant added to a positive objective changes its logarithm and so
can now change the descent's steps.
