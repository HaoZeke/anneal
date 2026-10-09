In values-only portfolio runs the GSA arm starts one chain at the incumbent,
as dual_annealing starts from ``x0``, and anneals without a local search. The
DE arm's population starts with the incumbent, as SciPy's
differential_evolution places ``x0``, and its first slice evolves the
population once it is seeded.
