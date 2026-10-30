A relaxation that ends at a non-finite energy, as a refused start does,
is counted as capped without a convergence check, so the engine is not
asked a second time about a geometry it has just refused. The check
treated a NaN gradient as converged, because the largest-component fold
skipped NaN; ``gradient_is_converged`` now requires every component to be
below the threshold. A trial that fails the quench sanity test while
budget remains updates the allocator that drew it: the depth allocator
gets the lowest depth reward of the run so far, the Beta allocator a
failure, and the contextual allocator zero. ``Outcome::rejected_trials``
counts those trials; the trial the spent budget cuts off is not one.
