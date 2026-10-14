The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, kept its quasi-Newton
approximation after a direction from it failed to descend, and stepped down
the steepest-descent direction instead. The updates keep the approximation
positive definite in exact arithmetic, but rounding can leave it indefinite
once the curvature has grown by many decades, and kept, it refused direction
after direction. Such a direction now also starts the approximation over
from the box metric scaled by the latest curvature pair. With this and the
release's other changes to the loop, on a six-parameter binary Lennard-Jones
fit to clusters whose atoms are at least 0.75 apart, with the base-10
logarithm of 1e-16 plus the mean squared energy error per atom as the loss,
whose floor is -16, one replica now ends at -15.999 or below on 38 of 40
seeds at 5000 evaluations, where 0.11.1 ended there on 32 and nevergrad's
``NgIohTuned`` ends there on 28, and below -15 on 38, the 35 on which 0.11.1
did among them, against 30 for ``NgIohTuned``. It ends lower than
``NgIohTuned`` on 12 seeds and higher on 1 (the one-sided Wilcoxon
probability that it is not better is 0.0031), where 0.11.1 ended lower on 11
and higher on 8, and through ChemFit's ``Fitter.fit_nevergrad``, over seeds
0 to 9, it ends lower on 2 and ties 8, where 0.11.1 ended lower on 2 and
higher on 2. On the same fit to clusters drawn without that separation it
now ends at a median of -13.5 at 5000 evaluations against 0.11.1's -12.9
(lower on 32 seeds and higher on 8, 1.6e-4 that it is not better), ahead of
``NgIohTuned`` on 35 seeds where 0.11.1 was on 33. Runs in which the descent
meets such a direction no longer repeat those of 0.11.1.
