In values-only portfolio runs the success threshold, which decides whether the
opening descent goes on, whether a descent keeps its turn, whether a GSA or
CMA-ES phase has gained and when a CMA-ES run stops, is ``1e-4 max(|f|, s)``,
where ``s`` is the resolution ``eps |f|`` of the start's value, in place of
``1e-4 max(|f|, 1)``, which turned absolute below ``|f| = 1``. The floor is
positive and fixed for the run, so a bounded objective still allows only a
bounded number of successes, as the restart guarantee needs. Until a curvature
pair scales the inverse Hessian, each line search of a finite-difference
descent starts with a trial whose largest move is 5% of a side whatever the
gradient's size, where a small gradient used to shorten it. Multiplying the
objective by a power of two now leaves a run unchanged until the surrogate or
the restart arm plays, whose temperatures are absolute; adding a constant to it
still coarsens the threshold.
