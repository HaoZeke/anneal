``run`` and ``global_optimize`` refused a box with a coordinate whose two
ends are equal, with ``low[i] must be strictly less than high[i]``, though
the documentation said such a coordinate stays fixed, as ``minimize`` and
``run_qmc`` kept it. ``run``, ``run_qmc`` and ``global_optimize`` now hold
it as ``minimize`` does: the run makes the calls of the same run on the
other coordinates, and ``obj_fn``, ``grad_fn`` and ``eval_batch`` receive
every point at full length, with the fixed values. With every coordinate
fixed, a run is one call, at ``x0`` or at ``low``; ``global_optimize`` then
returns the keys of one controller, its arm counts empty, whatever the
number of replicas. ``low`` above ``high`` is still refused, now with
``low[i] must be less than or equal to high[i]``, and a run on a box with
no fixed coordinate makes the same calls as before. ``chemfit_box`` still
refuses a zero-width bound, and its message no longer says that the
drivers need ``lower < upper``.
