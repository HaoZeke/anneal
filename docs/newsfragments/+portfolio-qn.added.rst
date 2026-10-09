A ``qn`` portfolio arm: projected BFGS on finite-difference gradients from the
incumbent, with a projected backtracking line search and stencils that step
inward at the bounds. A descent that begins at an incumbent already evaluated,
``x0`` included, reuses its value. Forward differences give way to central ones
when they stall, and every stencil point and trial is a single charged
evaluation.
