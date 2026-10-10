A ``qn`` arm for values-only portfolio runs: projected BFGS on
finite-difference gradients from the incumbent, with a projected backtracking
line search and stencils that step inward at the bounds. The initial inverse
Hessian is the box metric ``diag(w^2)``, so sides that span decades descend as
fast as unit ones. A descent that begins at an incumbent already evaluated,
``x0`` included, reuses its value. Forward differences give way to central ones
when they stall, a converged descent is kicked from the incumbent with draws
independent of the CMA-ES arm's, and every stencil point and trial is a single
charged evaluation.
