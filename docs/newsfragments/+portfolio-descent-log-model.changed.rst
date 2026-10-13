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
than for the values, the line searches step on its model; from then until
the descent next starts afresh, each accepted step puts the next line search
on whichever model predicted its value more closely, the logarithm's on a
tie, and a value at or below zero retires the model of the logarithm. A
squared error far from its fit falls through decades and is nearer quadratic
in its logarithm. With the mean
squared energy error per atom of a six-parameter binary Lennard-Jones fit as
the loss, the loop ended lower than nevergrad's ``NgIohTuned`` on 17 of 40
seeds at 1000 evaluations: the descent had 10 of the loop's 25 turns per run,
against 35 of 36 with the logarithm of that error as the loss, the other arms
lowered the incumbent in few runs, and before its second curvature pair per
coordinate the model of the values missed the values of the descent's
accepted steps by a median of 39%, against 7.5% for a model of their
logarithm. Over 40 paired seeds from starts in the middle 60% of the box, at
1000 evaluations that fit now ends at a median of 9.78e-5 against 0.0332 for
``NgIohTuned`` (36/0/4) and 0.0331 for the loop before (37/3/0), a
twelve-parameter fit of three species at 1.5 against 6.88 (36/0/4) and 2.67
(33/5/2), and a two-parameter pair fit at 0, level with both (0/40/0); at 50
evaluations the pair fit ends at 0.767 against 0.993 (28/0/12) and 3.55
(35/3/2), and at 500 the binary fit at 0.014 against 0.0783 (35/0/5) and
0.129 (37/3/0). These figures and those below are of the loop with the hold
this release also puts on the descent's extrapolated trials, with the log
ratios its opening now reads and with the restart of the descent's
approximation after a direction that does not descend. On 13-atom
Lennard-Jones positions, the six-parameter fit with the logarithm of its error
as the loss, and ten-dimensional Rastrigin and Rosenbrock, at 1000 and 5000
evaluations, the loop stays level with the loop before, but for the
log-loss fit at 5000, where it now ends lower (32/0/8, 1.6e-4 that it is not
better), and ahead of ``NgIohTuned`` except on Rosenbrock at 5000, where the
two are level.
``NgIohTuned`` still ends lower
on more seeds of the pair fit at 200 evaluations, on 6 of the 8 that do not
tie, where both medians are below ``1e-14`` (the one-sided Wilcoxon
probability that the loop is not worse is 0.074, against 0.046 for the loop
before), has the lower median on the twelve-parameter fit at 200 (2.68e4
against 3.12e4, though the loop ends lower on 21 of 40 seeds), and at 1000
ends lower on 4 of the 40 seeds of the binary fit and 4 of the
twelve-parameter one. At about ten evaluations per parameter it leads on
both, ending lower on 32 of 40 seeds of the binary fit at 60 evaluations and
on 37 of the twelve-parameter fit at 100, directly and through ChemFit's
``Fitter.fit_nevergrad`` alike (its medians 98.1 and 7.0e4 asked and told
directly and 91.5 and 7.26e4 through ChemFit, against the loop's 2.82e3 and
2.98e5), and the two are level at 100 evaluations on
the binary fit and at 200 on the twelve-parameter one; the regime notes give
these cells. Four shared batch replicas stay level with those
before on every cell of those four problems but the fit of the logarithm at
5000 evaluations, where they end lower (22/7/11, 0.042 that they are not
better) but trail one replica (11/5/24, 0.0014 that they are not worse),
where the two were level (19/3/18). Runs in
which the descent steps on the logarithm, which it cannot before
``2 d (d + 1)`` evaluations for ``d`` free coordinates, no longer repeat
those of 0.11.1. Multiplying the objective by a power of two still leaves
a run unchanged, while a constant added to a positive objective changes
its logarithm and so can now change the descent's steps.
