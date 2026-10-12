The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, followed a first trial that
gained more than six tenths of the decrease its model predicted with one trial
at the minimiser of the quadratic through the two values and the slope, up to
sixteen times as far along the line, projected onto the box. Far from its fit
the logarithm of a squared error is concave along a line and the quadratic has
no minimiser, so on a log-loss fit the first line searches took the longest
trial, which crossed most of the box and was accepted on a face; the descent
holds a coordinate at its bound while descending would carry it out of the
box, and where the face held a minimum of the boxed fit the descent stopped in
it. A trial whose quadratic has no minimiser now goes at most halfway from the
iterate to the nearest bound along the line, while one at the minimiser of a
quadratic that has one goes there as before. On a six-parameter binary
Lennard-Jones fit to clusters whose atoms are at least 0.75 apart, with the
base-10 logarithm of the mean squared energy error per atom as the loss, 36 of
40 openings had ended on such a face, and one replica had ended lower than
nevergrad's ``NgIohTuned`` on 9 of 40 seeds at 1000 evaluations (medians -1.82
against -7.15). Over the same seeds it now ends lower on 28 at 1000 (median
-10.9; the one-sided Wilcoxon probability that it is not better is 0.075, a
sign test's 0.0083) and on 28 at 500 (-8.67 against -3.77, 1.6e-5), and
through ChemFit's ``Fitter.fit_nevergrad`` on 31 at 1000 (0.012) and 29 at
500. ``NgIohTuned`` still ends lower on 12 of the 40 seeds at both budgets,
all of them seeds on which the loop ends above a log error of -5, and at 5000
the two are level, both reaching the floor of the loss on most seeds. On the
gate's problems, the plain-loss fits and the other problems of the regime
notes, no cell trails the loop without the hold. Runs in which such a trial
would have gone more than halfway to a bound no longer repeat those before;
runs with a gradient, a declared noise scale or the ``legacy`` policy are
unchanged.
