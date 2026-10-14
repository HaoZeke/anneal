The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, kept its quasi-Newton
approximation after a direction from it failed to descend, and stepped down
the steepest-descent direction instead. The updates keep the approximation
positive definite in exact arithmetic, but rounding can leave it indefinite
once the curvature has grown by many decades, and kept, it refused direction
after direction. On a six-parameter binary Lennard-Jones fit to clusters whose
atoms are at least 0.75 apart, with the base-10 logarithm of 1e-16 plus the
mean squared energy error per atom as the loss, the descent then crept toward
the loss's floor of -16: on one seed every direction it refused came from an
approximation with negative diagonal entries, and it took 263
steepest-descent steps, each moving the parameters by 1.4e-11 to 4e-11, over
the last 3700 of its 5000 evaluations, going from -15.74 to -15.84. Such a
direction now also starts the approximation over from the box metric scaled
by the latest curvature pair. Over 40 seeds at 5000 evaluations one replica
now ends at -15.999 or below on 38, against 21 before and 28 for nevergrad's
``NgIohTuned``, and below -15 on the same 38 seeds as before, against 30 for
``NgIohTuned``. It ends lower than ``NgIohTuned`` on 12 seeds and higher on 1
(the one-sided Wilcoxon probability that it is not better is 0.0031), where
it was level (11/10/19), and through ChemFit's ``Fitter.fit_nevergrad``, over
seeds 0 to 9, it ends lower on 2 and ties 8, where it ended higher on 5
(2/3/5). On the same fit to clusters drawn without that separation it now
ends at a median of -13.5 against -13.1 at 5000 evaluations (lower on 30
seeds, higher on 8, 6.6e-5), ahead of ``NgIohTuned`` on 35 seeds where it was
on 34, and four shared batch replicas end at -13.2 against -12.8 (19/15/6,
0.008), still behind one replica (11/5/24, where they were 15/4/21). Runs on
13-atom Lennard-Jones positions and ten-dimensional Rastrigin and Rosenbrock
at 1000 and 5000 evaluations, on both fits at 1000 and fewer, on the
plain-loss fits and on the other benchmark rows of the regime notes repeat
their paths call for call, but for nine that end within 1e-9 of their
earlier values. Runs in which the descent met such a direction part from
their earlier paths at the first.
