``gle_langevin_objective``, ``gle_langevin_preconditioned_objective``,
``global_optimize_objective`` and ``qmc_polish_objective`` read the gradient
of a ``PyObjective`` as a contiguous float64 vector of its dimension. A
``grad_fn`` that returned a strided view, a list, a column or another dtype,
or that raised, had made them panic with an exception that ``except
Exception`` does not catch, and a gradient of another length went
unnoticed, except that one too long made the two GLE functions panic.
``grad_fn`` may now return any array-like of the right length; a gradient
of another length ends the run with a ``TypeError`` giving it, and an
exception from ``grad_fn`` ends the run and is raised as itself.
