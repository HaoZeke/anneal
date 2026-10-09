``global_optimize`` only calls ``grad_fn`` inside the box. A point an arm
proposes outside is mirror-reflected before the objective sees it, and the
gradient is now taken at the same reflected point, its sign flipped along
every coordinate the fold reverses, which is the gradient of the function
the portfolio actually evaluates. Before, the gradient was called at the
unreflected point, up to 0.34 outside a ``[-3, 3]`` box. No arm hands the
caller's objective or gradient a non-finite coordinate: the population arm's
success-history mean weighed a success from a walker at ``+inf`` as
infinite, its scale factors became NaN, and up to 15% of a run's calls went
to NaN points when part of the box was infeasible.
