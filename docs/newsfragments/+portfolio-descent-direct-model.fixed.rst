The finite-difference descent of the values-only loop, which
``global_optimize`` runs with no ``grad_fn`` or ``noise_sigma`` under the
``auto`` policy and ``minimize`` runs by default, once its steps have moved
onto the logarithm of the values puts each next line search on the model, of
the values or of their logarithm, that predicted the accepted step's value
more closely, and each model predicts from its direct approximation of the
Hessian. Rounding can leave that approximation indefinite, and nothing then
started it over until the descent restarted or a direction from its model
failed to descend, its update skipped along each step where it gave ``s.B.s``
at or below zero: on a six-parameter binary Lennard-Jones fit with the mean
squared energy error per atom as the loss, at 5000 evaluations, the
logarithm's failed a Cholesky test at 571 of the 962 predictions of seeds 0
to 3, and at 228 of the 962 the choice differed from the one the same pairs
give in 60-digit arithmetic. Along such a step the approximation now becomes
the inverse of the symmetric part of the model's inverse approximation, which
has taken the pair, or, where that part is not positive definite either,
starts over as it does after a direction that does not descend; over the same
seeds it then fails at 37 of 984 predictions and the choice differs at 1. Of
1560 runs of one replica, on Lennard-Jones parameter fits at 20 to 5000
evaluations and on 13-atom Lennard-Jones positions and ten-dimensional
Rastrigin and Rosenbrock at 1000 and 5000, 70 now part from their earlier
paths, all on fits with their mean squared error as the loss and on Rosenbrock
at 5000; of 1440 runs of four replicas from ``x0`` on those three problems and
the six-parameter fit, 12, all on Rosenbrock at 5000; none of the 1720 runs on
the other benchmark rows of the regime notes; and 34 of the 2920 runs of their
strain grid. Each ends within 1e-9 of its earlier value but for four runs of
the grid at 20000 evaluations, on the logarithm of a rotated quadratic in 30
coordinates, which end at most 0.0024 higher.
