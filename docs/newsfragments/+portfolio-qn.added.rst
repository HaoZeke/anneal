A ``qn`` portfolio arm: projected BFGS on finite-difference gradients from the
incumbent, with a projected backtracking line search and stencils that step
inward at the bounds. Forward differences give way to central ones when they
stall, and every stencil point and trial is a single charged evaluation.
