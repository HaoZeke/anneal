In values-only portfolio runs the GSA arm starts one chain at the incumbent,
ends its first temperature step with a finite-difference local search from its
best point, and follows every later record with another, as dual_annealing
does with L-BFGS-B, until a descent from the incumbent has stopped paying (the
end of the opening or of the descent's turn). From then on its records are
left to the descent and the closing polish. The search stops on L-BFGS-B's
tests: a decrease below ``1e7 eps max(|f|, 1)`` or a projected gradient below
``1e-5``. The DE arm's population starts with the incumbent, as SciPy's
differential_evolution places ``x0``, and its first slice evolves the
population once it is seeded.
