``xtb_getGradient`` is dE/dx. Storing that buffer as the rgpot force
made every descent step walk uphill, because rgpot negates the force
to build the objective gradient. The callback stores the negated
buffer, so the objective gradient is the energy slope.
